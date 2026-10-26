from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import APIRouter, Query

from ragglet.api.deps import Svc
from ragglet.api.problems import Problem, problem_error
from ragglet.api.routers.runs import _record
from ragglet.api.schemas import QueryDetail, QueryPageOut, ResultChunk, TruthDoc
from ragglet.registry.per_query import result_chunks, truth_docs
from ragglet.registry.protocol import QueryFilter, RegistryError

router = APIRouter(tags=["runs"])

TITLES = {404: "Not found", 409: "Conflict", 422: "Request is invalid"}
ERRORS = {404: {"model": Problem}, 409: {"model": Problem}, 422: {"model": Problem}}


@contextmanager
def problems(run_id: str) -> Iterator[None]:
    """Ожидаемые ошибки реестра — Problem с их кодом и статусом."""
    try:
        yield
    except RegistryError as e:
        raise problem_error(e.status, e.code, TITLES.get(e.status, "Error"), e.message,
                            where=f"runs/{run_id}", path=e.path) from e


@router.get("/runs/{run_id}/queries", responses=ERRORS)
async def list_queries(
        run_id: str,
        svc: Svc,
        repeat: Annotated[int, Query(ge=0)] = 0,
        outcome: Annotated[str | None, Query(description="ok | timeout | error")] = None,
        miss_at: Annotated[int | None, Query(ge=1, description="промах при k: первый релевантный дальше k")] = None,
        min_total_ms: Annotated[float | None, Query(ge=0)] = None,
        q: Annotated[str | None, Query(description="подстрока в qid или тексте запроса")] = None,
        sort: Annotated[str | None, Query(description="колонка или first_relevant_rank, '-' — по убыванию")] = None,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> QueryPageOut:
    """Строки per-query повтора: parquet или CSV, NaN -> null; списки и длинные тексты — в детали запроса."""
    await _record(svc, run_id)
    with problems(run_id):
        page = await svc.registry.queries(run_id, QueryFilter(
            repeat=repeat, outcome=outcome, offset=offset, limit=limit, miss_at=miss_at,
            min_total_ms=min_total_ms, text=q, sort=sort,
        ))
    return QueryPageOut(repeat=page.repeat, repeats=page.repeats, source=page.source, columns=page.columns,
                        total=page.total, items=page.items)


@router.get("/runs/{run_id}/queries/{qid}", responses=ERRORS)
async def get_query(run_id: str, qid: str, svc: Svc, repeat: Annotated[int, Query(ge=0)] = 0) -> QueryDetail:
    """Один запрос: строка per-query, релевантные документы и финальная выдача с рангами."""
    record = await _record(svc, run_id)
    with problems(run_id):
        row = await svc.registry.query_row(run_id, repeat, qid)
    if row is None:
        raise problem_error(404, "query.not_found", "Not found", f"query {qid!r} is not in repeat {repeat}",
                            where=f"runs/{run_id}")
    context_enabled = bool((record.scenario.get("context") or {}).get("enabled"))
    return QueryDetail(
        qid=qid,
        repeat=repeat,
        query=str(row.get("query") or ""),
        row=row,
        truth=[TruthDoc(**doc) for doc in truth_docs(row)],
        results=[ResultChunk(**chunk) for chunk in result_chunks(row, context_enabled)],
    )
