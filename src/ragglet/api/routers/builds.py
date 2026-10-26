from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Annotated

import anyio
import yaml
from fastapi import APIRouter, Query, Response

from ragglet.api import probes
from ragglet.api.core_calls import check_build
from ragglet.api.deps import Services, Svc
from ragglet.api.library import etag_of
from ragglet.api.problems import Problem, issues_out, parse_config, problem_error
from ragglet.api.refs import library_users
from ragglet.api.routers.runs import check_library_ref, check_sort
from ragglet.api.routers.tasks import cancel, parse_include, read_log
from ragglet.api.schemas import (
    BuildAccepted,
    BuildCheck,
    BuildItem,
    BuildList,
    BuildRequest,
    BuildStatus,
    BuildTargetCheck,
    CollectionRef,
    LogChunkOut,
    live_fields,
)
from ragglet.config.index_profile import IndexProfileConfig
from ragglet.indexing.metadata import RAGGLET_METADATA_KEY
from ragglet.jobs.protocol import TaskKind, TaskRef, TaskSpec, TaskState
from ragglet.registry.listing import BUILD_SORT_KEYS, duration_s
from ragglet.registry.protocol import BuildQuery, BuildRecord, NewBuild

if TYPE_CHECKING:  # build_index тянет torch: в API — только при проверке
    from ragglet.indexing.build_index import BuildResult

router = APIRouter(tags=["builds"])

TITLE = "Index profile config is invalid"
INCLUDE = {"profile"}


async def build_check(svc: Services, cfg: IndexProfileConfig, result: BuildResult) -> BuildCheck:
    """
    BuildResult ядра плюс то, чего ядро не знает: dry-run сборки к Qdrant не подключается.
    Это данные, а не проверки: недоступный endpoint — reachable: false, а не 422.
    """
    timeout = svc.settings.probe_timeout_s
    states = await probes.gather([partial(probes.target, t.endpoint, t.collection, timeout) for t in result.targets])
    scenario_users, _ = await anyio.to_thread.run_sync(library_users, svc)
    targets = []
    for target, state in zip(result.targets, states, strict=True):
        info = state.info
        meta = ((info.metadata or {}).get(RAGGLET_METADATA_KEY) or {}) if info else {}
        targets.append(BuildTargetCheck(
            id=target.id,
            endpoint=target.endpoint,
            collection=target.collection,
            reachable=state.reachable,
            error=state.error,
            exists=(info is not None) if state.reachable else None,
            status=info.status if info else None,
            points=info.points if info else None,
            built_at=meta.get("built_at"),
            index_profile_id=meta.get("index_profile_id"),
            used_by=scenario_users.get((probes.normalize_endpoint(target.endpoint), target.collection), []),
        ))
    return BuildCheck(profile=cfg.id, dataset=result.dataset, corpus_sha256=result.corpus_sha256,
                      documents=result.documents, targets=targets, warnings=issues_out(result.warnings))


@router.post(
    "/builds",
    status_code=202,
    response_model=None,
    responses={200: {"model": BuildCheck}, 202: {"model": BuildAccepted}, 422: {"model": Problem}},
)
async def post_build(req: BuildRequest, response: Response, svc: Svc) -> BuildCheck | BuildAccepted:
    """dry_run — что будет пересоздано (200 или 422); иначе проверка и постановка в очередь (202)."""
    cfg = parse_config(req.profile, IndexProfileConfig, TITLE)
    check_library_ref(req.profile_id, cfg.id, "profile_id")
    result = await check_build(cfg)  # ValidationError — 422
    if req.dry_run:
        response.status_code = 200
        return await build_check(svc, cfg, result)

    record = await svc.registry.create_build(NewBuild(
        name=cfg.id,
        profile_id=req.profile_id,
        profile=cfg.model_dump(),
        etag=etag_of(req.profile),
        dataset=result.dataset,
        targets=result.targets,
        warnings=result.warnings,
    ))
    status = await svc.jobs.submit(TaskSpec(ref=TaskRef(TaskKind.BUILD, record.id), name=cfg.id,
                                            config=cfg.model_dump(by_alias=True), run_dir=record.build_dir))
    response.headers["Location"] = f"/v1/builds/{record.id}"
    return BuildAccepted(id=record.id, state=status.state, position=status.position,
                         warnings=issues_out(result.warnings))


def build_item(record: BuildRecord) -> BuildItem:
    documents = (record.result or {}).get("documents")
    return BuildItem(
        id=record.id,
        profile=record.name,
        profile_id=record.profile_id,
        state=record.state,
        enqueued_at=record.enqueued_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        duration_s=duration_s(record),
        dataset=record.dataset,
        targets=[CollectionRef(endpoint=t.endpoint, collection=t.collection) for t in record.targets],
        documents=documents if isinstance(documents, int) else None,
    )


@router.get("/builds", responses={422: {"model": Problem}})
async def list_builds(
        svc: Svc,
        state: TaskState | None = None,
        profile_id: str | None = None,
        dataset: Annotated[str | None, Query(description="source:id — любая ревизия; с @ревизией — точная")] = None,
        collection: Annotated[str | None, Query(description="среди целей сборки")] = None,
        since: datetime | None = None,
        until: datetime | None = None,
        q: str | None = None,
        sort: Annotated[str, Query(description="ключ сортировки, '-' — по убыванию; null всегда в конце")] = "-enqueued_at",
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> BuildList:
    page = await svc.registry.list_builds(BuildQuery(
        profile_id=profile_id, state=state, offset=offset, limit=limit, dataset=dataset, collection=collection,
        since=since, until=until, text=q, sort=check_sort(sort, BUILD_SORT_KEYS),
    ))
    return BuildList(total=page.total, items=[build_item(record) for record in page.items])


async def _record(svc: Services, build_id: str) -> BuildRecord:
    record = await svc.registry.get_build(build_id)
    if record is None:
        raise problem_error(404, "build.not_found", "No such build", f"build {build_id!r} not found",
                            where=f"builds/{build_id}")
    return record


async def build_status(svc: Services, record: BuildRecord, include: Collection[str] = ()) -> BuildStatus:
    out = BuildStatus(
        id=record.id,
        profile=record.name,
        profile_id=record.profile_id,
        etag=record.etag,
        dataset=record.dataset,
        targets=[CollectionRef(endpoint=t.endpoint, collection=t.collection) for t in record.targets],
        warnings=issues_out(record.warnings),
        result=record.result,
        **live_fields(record, await svc.jobs.status(TaskRef(TaskKind.BUILD, record.id))),
    )
    if "profile" in include:
        out.profile_yaml = yaml.safe_dump(record.profile, allow_unicode=True, sort_keys=False)
    return out


@router.get("/builds/{build_id}", responses={404: {"model": Problem}})
async def get_build(build_id: str, svc: Svc, include: Annotated[str | None, Query()] = None) -> BuildStatus:
    return await build_status(svc, await _record(svc, build_id), parse_include(include, INCLUDE))


@router.post("/builds/{build_id}/cancel", responses={404: {"model": Problem}, 409: {"model": Problem}})
async def cancel_build(build_id: str, svc: Svc) -> BuildStatus:
    record = await _record(svc, build_id)
    await cancel(svc.jobs, TaskRef(TaskKind.BUILD, build_id), record.state)
    return await build_status(svc, await _record(svc, build_id))


@router.get("/builds/{build_id}/log", responses={404: {"model": Problem}})
async def build_log(build_id: str, svc: Svc, offset: Annotated[int, Query(ge=0)] = 0) -> LogChunkOut:
    await _record(svc, build_id)
    return await read_log(svc.jobs, TaskRef(TaskKind.BUILD, build_id), offset)
