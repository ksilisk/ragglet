from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import yaml
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ragglet.config.errors import FieldError
from ragglet.core.issues import Issue, ValidationError

log = logging.getLogger("ragglet.api")


class ProblemIssue(BaseModel):
    code: str  # "config.value_error", "index.not_green", "library.conflict"
    severity: Literal["error", "warning"]
    path: str | None = None  # поле конфига: "retrieval.sources[1].params"
    where: str | None = None  # место вне конфига: "qrels/test.tsv:7", "target default (nfcorpus_v1)"
    message: str


class Problem(BaseModel):
    """Тело любого ответа не 2xx."""

    title: str
    hint: str | None = None
    issues: list[ProblemIssue]


class ProblemError(Exception):
    def __init__(self, status: int, problem: Problem):
        super().__init__(problem.title)
        self.status = status
        self.problem = problem


def problem_error(status: int, code: str, title: str, message: str, *, where: str | None = None,
                  path: str | None = None) -> ProblemError:
    issue = ProblemIssue(code=code, severity="error", path=path, where=where, message=message)
    return ProblemError(status, Problem(title=title, issues=[issue]))


def issue_out(issue: Issue) -> ProblemIssue:
    return ProblemIssue(
        code=issue.code, severity=issue.severity.value, path=issue.path, where=issue.where, message=issue.message
    )


def issues_out(issues: Sequence[Issue]) -> list[ProblemIssue]:
    return [issue_out(issue) for issue in issues]


def from_validation_error(e: ValidationError) -> Problem:
    return Problem(title=e.title, hint=e.hint, issues=issues_out(e.issues))


def loc_to_path(loc: Sequence[Any]) -> str | None:
    """('retrieval', 'sources', 1, 'params') -> 'retrieval.sources[1].params'."""
    path = ""
    for part in loc:
        path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else str(part))
    return path or None


def _pydantic_issue(err: Any, prefix: str) -> ProblemIssue:
    path = loc_to_path(err["loc"])
    cause = (err.get("ctx") or {}).get("error")
    if isinstance(cause, FieldError):  # model-валидатор сам назвал поле
        path = f"{path}.{cause.path}" if path else cause.path
    message = err["msg"].removeprefix("Value error, ")
    return ProblemIssue(code=f"{prefix}.{err['type']}", severity="error", path=path, message=message)


def from_pydantic(e: PydanticValidationError, title: str) -> Problem:
    return Problem(title=title, issues=[_pydantic_issue(err, "config") for err in e.errors()])


def from_yaml(e: yaml.YAMLError, title: str) -> Problem:
    mark = getattr(e, "problem_mark", None)
    where = f"line {mark.line + 1}, column {mark.column + 1}" if mark is not None else None
    problem, context = getattr(e, "problem", None), getattr(e, "context", None)
    message = f"{problem} ({context})" if problem and context else str(problem or e)
    return Problem(title=title, issues=[ProblemIssue(code="config.yaml", severity="error", where=where,
                                                     message=message)])


def load_mapping(text: str, title: str) -> dict[str, Any]:
    """YAML-текст -> mapping; битый YAML или не mapping в корне — 422 config.yaml."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ProblemError(422, from_yaml(e, title)) from e
    if not isinstance(data, dict):
        raise problem_error(422, "config.yaml", title, "YAML root must be a mapping")
    return data


def parse_config[M: BaseModel](text: str, model: type[M], title: str) -> M:
    """YAML-текст -> конфиг; любая ошибка разбора и схемы — 422 Problem."""
    data = load_mapping(text, title)
    try:
        return model.model_validate(data)
    except PydanticValidationError as e:
        raise ProblemError(422, from_pydantic(e, title)) from e


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProblemError)
    async def _problem(request: Request, e: ProblemError) -> JSONResponse:
        return JSONResponse(e.problem.model_dump(), status_code=e.status)

    @app.exception_handler(ValidationError)
    async def _core_validation(request: Request, e: ValidationError) -> JSONResponse:
        return JSONResponse(from_validation_error(e).model_dump(), status_code=422)

    @app.exception_handler(RequestValidationError)
    async def _request(request: Request, e: RequestValidationError) -> JSONResponse:
        issues = []
        for err in e.errors():
            loc = list(err["loc"])
            issues.append(_pydantic_issue({**err, "loc": loc[1:] if loc[:1] == ["body"] else loc}, "request"))
        return JSONResponse(Problem(title="Request is invalid", issues=issues).model_dump(), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, e: StarletteHTTPException) -> JSONResponse:
        code = "http.not_found" if e.status_code == 404 else f"http.{e.status_code}"
        problem = Problem(title=str(e.detail), issues=[
            ProblemIssue(code=code, severity="error", where=request.url.path, message=str(e.detail)),
        ])
        return JSONResponse(problem.model_dump(), status_code=e.status_code, headers=getattr(e, "headers", None))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, e: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        problem = Problem(title="Internal error", issues=[
            ProblemIssue(code="server.error", severity="error", message=f"{type(e).__name__}: {e}"),
        ])
        return JSONResponse(problem.model_dump(), status_code=500)
