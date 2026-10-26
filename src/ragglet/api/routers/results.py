from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO, Annotated

import anyio
from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from ragglet.api.deps import Svc
from ragglet.api.problems import Problem, problem_error
from ragglet.api.routers.runs import _record
from ragglet.api.schemas import (
    ArtifactOut,
    Availability,
    QueryDetail,
    QueryPageOut,
    ResultChunk,
    TruthDoc,
)
from ragglet.registry.files import zip_artifacts
from ragglet.registry.inspect import QueryInspector
from ragglet.registry.per_query import result_chunks, truth_docs
from ragglet.registry.protocol import QueryFilter, RegistryError

router = APIRouter(tags=["runs"])

TITLES = {404: "Not found", 409: "Conflict", 422: "Request is invalid"}
MEDIA_TYPES = {
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".csv": "text/csv",
    ".parquet": "application/vnd.apache.parquet",
    ".log": "text/plain; charset=utf-8",
}
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
    """
    Один запрос: строка per-query, релевантные документы и финальная выдача с рангами. Оценки и
    заголовки — из датасета прогона, тексты чанков — из Qdrant, если источники те же, что у
    прогона; иначе grades и texts говорят, чего нет и почему.
    """
    record = await _record(svc, run_id)
    with problems(run_id):
        row = await svc.registry.query_row(run_id, repeat, qid)
    if row is None:
        raise problem_error(404, "query.not_found", "Not found", f"query {qid!r} is not in repeat {repeat}",
                            where=f"runs/{run_id}")
    context_enabled = bool((record.scenario.get("context") or {}).get("enabled"))
    chunks = result_chunks(row, context_enabled)

    inspector = svc.query_inspector = svc.query_inspector or QueryInspector(svc.settings.probe_timeout_s)
    manifest = await svc.registry.manifest(run_id)
    dataset = await inspector.dataset(record, manifest)
    payloads, texts = await inspector.texts(record, manifest, [c["chunk_id"] for c in chunks])
    qrels = dataset.qrels.get(qid, {}) if dataset.qrels is not None else None
    titles = dataset.titles or {}

    def grade(doc_id: str) -> int | None:
        return None if qrels is None else qrels.get(doc_id, 0)

    return QueryDetail(
        qid=qid,
        repeat=repeat,
        query=str(row.get("query") or ""),
        row=row,
        truth=[TruthDoc(**doc, grade=grade(doc["doc_id"]), title=titles.get(doc["doc_id"]))
               for doc in truth_docs(row)],
        results=[
            ResultChunk(
                **chunk,
                grade=grade(chunk["doc_id"]),
                title=(payloads.get(chunk["chunk_id"]) or {}).get("title") or titles.get(chunk["doc_id"]),
                text=(payloads.get(chunk["chunk_id"]) or {}).get("text"),
            )
            for chunk in chunks
        ],
        grades=Availability(**vars(dataset.grades)),
        texts=Availability(**vars(texts)),
    )


@router.get("/runs/{run_id}/artifacts", responses=ERRORS)
async def list_artifacts(run_id: str, svc: Svc) -> list[ArtifactOut]:
    """Файлы каталога прогона как есть, с run.json и task.log; служебные файлы менеджера скрыты."""
    await _record(svc, run_id)
    return [ArtifactOut(name=a.name, size=a.size, modified_at=a.modified_at)
            for a in await svc.registry.artifacts(run_id)]


@router.get("/runs/{run_id}/artifacts.zip", responses=ERRORS, response_class=StreamingResponse)
async def download_zip(run_id: str, svc: Svc) -> StreamingResponse:
    """Весь каталог одним архивом: `<id>/<файл>`."""
    record = await _record(svc, run_id)
    names = [a.name for a in await svc.registry.artifacts(run_id)]
    archive = await anyio.to_thread.run_sync(zip_artifacts, record.run_dir, names)
    return StreamingResponse(_chunks(archive), media_type="application/zip", background=BackgroundTask(archive.close),
                             headers={"Content-Disposition": f'attachment; filename="{run_id}.zip"'})


@router.get("/runs/{run_id}/artifacts/{name}", responses=ERRORS, response_class=FileResponse)
async def download_artifact(run_id: str, name: str, svc: Svc) -> FileResponse:
    """Один файл как есть. Путь строится только из имён листинга: «..» и чужие пути не проходят."""
    record = await _record(svc, run_id)
    if name not in {a.name for a in await svc.registry.artifacts(run_id)}:
        raise problem_error(404, "artifact.not_found", "Not found", f"run {run_id!r} has no file {name!r}",
                            where=f"runs/{run_id}")
    path = record.run_dir / name
    return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix, "application/octet-stream"), filename=name)


def _chunks(archive: IO[bytes], size: int = 1 << 20) -> Iterator[bytes]:
    while chunk := archive.read(size):
        yield chunk
