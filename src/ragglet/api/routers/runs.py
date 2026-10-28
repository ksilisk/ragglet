from __future__ import annotations

from collections.abc import Collection
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated

import anyio
from fastapi import APIRouter, Query, Response

from ragglet.api.core_calls import check_run
from ragglet.api.deps import Services, Svc
from ragglet.api.library import etag_of
from ragglet.api.problems import Problem, issues_out, parse_config, problem_error
from ragglet.api.refs import snapshot_yaml
from ragglet.api.routers.tasks import cancel, parse_include, read_log
from ragglet.api.schemas import (
    DatasetCheck,
    LogChunkOut,
    RunAccepted,
    RunCheck,
    RunItem,
    RunList,
    RunPlanOut,
    RunRequest,
    RunStatus,
    RunSummaryOut,
    ScenarioFile,
    TargetCheck,
    live_fields,
)
from ragglet.bench.plan import run_plan
from ragglet.bench.preflight import Preflight
from ragglet.config.scenario import ScenarioConfig
from ragglet.datasets.sources import describe_dataset
from ragglet.jobs.protocol import TaskKind, TaskRef, TaskSpec, TaskState
from ragglet.registry.listing import RUN_SORT_KEYS, duration_s, run_tags, summary_column
from ragglet.registry.protocol import NewRun, RunQuery, RunRecord
from ragglet.storage.layout import build_storage_layout

router = APIRouter(tags=["runs"])

TITLE = "Scenario config is invalid"
INCLUDE = {"summary", "manifest", "scenario"}


def check_library_ref(ref: str | None, declared: str, field: str) -> None:
    """Ссылка на библиотеку обязана совпадать с id в содержимом: id файла = name сценария / id профиля."""
    if ref is not None and ref != declared:
        raise problem_error(422, "request.library_ref", "Request is invalid",
                            f"{field}={ref!r} does not match {declared!r} in the config", path=field)


def with_artifacts_dir(cfg: ScenarioConfig, artifacts_dir: Path) -> ScenarioConfig:
    return cfg.model_copy(update={"run": cfg.run.model_copy(update={"artifacts_dir": str(artifacts_dir)})})


async def library_etag(svc: Services, req: RunRequest, cfg: ScenarioConfig) -> str:
    """
    Версия сценария для `scenario_file`. «Повторить» присылает snapshot, а не текст файла:
    если конфиг совпадает с файлом библиотеки, прогон поставлен с этой версией файла.
    """
    etag = etag_of(req.scenario)
    if req.scenario_id is None:
        return etag
    entry = await anyio.to_thread.run_sync(svc.scenarios.find, req.scenario_id)
    if entry is None or entry.etag == etag or entry.config is None:
        return etag
    same = with_artifacts_dir(entry.config, svc.settings.artifacts_dir) == cfg
    return entry.etag if same else etag


def run_check(cfg: ScenarioConfig, checked: Preflight) -> RunCheck:
    dataset = checked.dataset
    targets = [
        TargetCheck(id=t.id, endpoint=t.endpoint, collection=t.collection, status=info.status, points=info.points,
                    indexed_vectors=info.indexed_vectors)
        for t in build_storage_layout(cfg.storage).targets
        if (info := checked.collections.get(t.id)) is not None
    ]
    return RunCheck(
        scenario=cfg.name,
        dataset=DatasetCheck(
            label=describe_dataset(cfg.dataset.source, cfg.dataset.id, dataset.revision),
            revision=dataset.revision,
            split=cfg.dataset.split,
            queries=len(dataset.items),
            corpus_sha256=dataset.corpus_sha256,
            eval_sha256=dataset.eval_sha256,
        ),
        index=checked.index,
        params=checked.params,
        targets=targets,
        plan=RunPlanOut(**asdict(run_plan(cfg, len(dataset.items)))),
        warnings=issues_out(checked.warnings),
    )


@router.post(
    "/runs",
    status_code=202,
    response_model=None,
    responses={200: {"model": RunCheck}, 202: {"model": RunAccepted}, 422: {"model": Problem}},
)
async def post_run(req: RunRequest, response: Response, svc: Svc) -> RunCheck | RunAccepted:
    """dry_run — синхронная проверка (200 или 422); иначе preflight и постановка в очередь (202)."""
    cfg = parse_config(req.scenario, ScenarioConfig, TITLE)
    check_library_ref(req.scenario_id, cfg.name, "scenario_id")
    # серверный путь: его назначает сервер, а не клиент
    cfg = with_artifacts_dir(cfg, svc.settings.artifacts_dir)
    checked = (await check_run(cfg)).preflight  # preflight №1; ValidationError — 422
    check = run_check(cfg, checked)
    if req.dry_run:
        response.status_code = 200
        return check

    record = await svc.registry.create_run(NewRun(
        name=cfg.name,
        scenario_id=req.scenario_id,
        scenario=cfg.model_dump(),  # ровно то, что ляжет в scenario.snapshot
        etag=await library_etag(svc, req, cfg),
        tags=req.tags,
        dataset=check.dataset.label,
        collection=build_storage_layout(cfg.storage).primary.collection,
        plan=run_plan(cfg, check.dataset.queries),
        warnings=checked.warnings,
    ))
    status = await svc.jobs.submit(TaskSpec(
        ref=TaskRef(TaskKind.RUN, record.id),
        name=cfg.name,
        config=cfg.model_dump(by_alias=True),
        run_dir=record.run_dir,
    ))
    response.headers["Location"] = f"/v1/runs/{record.id}"
    return RunAccepted(id=record.id, state=status.state, position=status.position, warnings=check.warnings)


async def _record(svc: Services, run_id: str) -> RunRecord:
    record = await svc.registry.get_run(run_id)
    if record is None:
        raise problem_error(404, "run.not_found", "No such run", f"run {run_id!r} not found", where=f"runs/{run_id}")
    return record


def check_sort(sort: str, allowed: Collection[str]) -> str:
    if sort.removeprefix("-") not in allowed:
        raise problem_error(422, "request.sort", "Request is invalid",
                            f"unknown sort key {sort!r}; allowed: {sorted(allowed)}, '-' for descending", path="sort")
    return sort


def run_item(record: RunRecord, k: int) -> RunItem:
    return RunItem(
        id=record.id,
        scenario=record.name,
        scenario_id=record.scenario_id,
        origin=record.origin,
        run_dir=str(record.run_dir),
        state=record.state,
        enqueued_at=record.enqueued_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        duration_s=duration_s(record),
        dataset=record.dataset,
        collection=record.collection,
        tags=run_tags(record),
        repeats=record.metrics.repeats if record.metrics else None,
        ndcg=summary_column(record, "ndcg", k),
        recall=summary_column(record, "recall", k),
        p50_ms=summary_column(record, "p50", k),
        p95_ms=summary_column(record, "p95", k),
        qps=summary_column(record, "qps", k),
        timeout_rate=summary_column(record, "timeout_rate", k),
    )


@router.get("/runs", responses={422: {"model": Problem}})
async def list_runs(
        svc: Svc,
        state: TaskState | None = None,
        scenario: Annotated[str | None, Query(description="имя сценария: история сценария")] = None,
        scenario_id: str | None = None,
        dataset: Annotated[str | None, Query(description="source:id — любая ревизия; с @ревизией — точная")] = None,
        collection: str | None = None,
        tag: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        q: Annotated[str | None, Query(description="подстрока в id, имени, описании, тегах, датасете, коллекции")] = None,
        sort: Annotated[str, Query(description="ключ сортировки, '-' — по убыванию; null всегда в конце")] = "-enqueued_at",
        k: Annotated[int, Query(ge=1, description="k для ndcg и recall")] = 10,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> RunList:
    """Реестр прогонов с колонками сводки; per-query не читается."""
    page = await svc.registry.list_runs(RunQuery(
        scenario_id=scenario_id, name=scenario, state=state, tag=tag, since=since, until=until,
        offset=offset, limit=limit, dataset=dataset, collection=collection, text=q,
        sort=check_sort(sort, RUN_SORT_KEYS), k=k,
    ))
    return RunList(k=k, total=page.total, items=[run_item(record, k) for record in page.items])


async def scenario_file(svc: Services, record: RunRecord) -> ScenarioFile | None:
    """Файл сценария библиотеки против версии, с которой прогон поставлен."""
    if record.scenario_id is None or record.etag is None:
        return None
    entry = await anyio.to_thread.run_sync(svc.scenarios.find, record.scenario_id)
    if entry is None:
        return "deleted"
    return "unchanged" if entry.etag == record.etag else "changed"


async def run_status(svc: Services, record: RunRecord, include: Collection[str] = ()) -> RunStatus:
    out = RunStatus(
        id=record.id,
        scenario=record.name,
        scenario_id=record.scenario_id,
        etag=record.etag,
        origin=record.origin,
        run_dir=str(record.run_dir),
        scenario_file=await scenario_file(svc, record),
        plan=RunPlanOut(**asdict(record.plan)) if record.plan else None,
        dataset=record.dataset,
        collection=record.collection,
        tags=record.tags,
        warnings=issues_out(record.warnings),
        **live_fields(record, await svc.jobs.status(TaskRef(TaskKind.RUN, record.id))),
    )
    if "summary" in include and (summary := await svc.registry.summary(record.id)) is not None:
        out.summary = RunSummaryOut(repeats=summary.repeats, aggregate=summary.aggregate)
    if "manifest" in include:
        out.manifest = await svc.registry.manifest(record.id)
    if "scenario" in include:
        out.scenario_yaml = snapshot_yaml(record.scenario)
    return out


@router.get("/runs/{run_id}", responses={404: {"model": Problem}})
async def get_run(run_id: str, svc: Svc, include: Annotated[str | None, Query()] = None) -> RunStatus:
    """Лёгкий статус для опроса раз в секунду; тяжёлое — через include=summary,manifest,scenario."""
    return await run_status(svc, await _record(svc, run_id), parse_include(include, INCLUDE))


@router.post("/runs/{run_id}/cancel", responses={404: {"model": Problem}, 409: {"model": Problem}})
async def cancel_run(run_id: str, svc: Svc) -> RunStatus:
    record = await _record(svc, run_id)
    await cancel(svc.jobs, TaskRef(TaskKind.RUN, run_id), record.state)
    return await run_status(svc, await _record(svc, run_id))


@router.get("/runs/{run_id}/log", responses={404: {"model": Problem}})
async def run_log(run_id: str, svc: Svc, offset: Annotated[int, Query(ge=0)] = 0) -> LogChunkOut:
    record = await _record(svc, run_id)
    return await read_log(svc.jobs, TaskRef(TaskKind.RUN, run_id), offset, record.run_dir)
