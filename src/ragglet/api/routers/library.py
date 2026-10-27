from __future__ import annotations

from typing import Annotated, Any

import anyio
from fastapi import APIRouter, Path, Query, Response

from ragglet.api.deps import Svc
from ragglet.api.library import ConfigLibrary, LibraryEntry
from ragglet.api.problems import Problem, problem_error
from ragglet.api.refs import collection_refs, snapshot_yaml
from ragglet.api.schemas import (
    ID_PATTERN,
    BuildBrief,
    ConfigDocument,
    ConfigWrite,
    IndexProfileCreate,
    IndexProfileItem,
    Pipeline,
    RunBrief,
    ScenarioCreate,
    ScenarioItem,
)
from ragglet.config.index_profile import IndexProfileConfig
from ragglet.config.scenario import ScenarioConfig
from ragglet.datasets.sources import describe_dataset
from ragglet.registry.listing import summary_column
from ragglet.registry.protocol import BuildQuery, RunQuery

router = APIRouter()

EntryId = Annotated[str, Path(pattern=ID_PATTERN)]
ERRORS: dict[int | str, dict[str, Any]] = {404: {"model": Problem}, 409: {"model": Problem}, 422: {"model": Problem}}


def document(entry: LibraryEntry) -> ConfigDocument:
    return ConfigDocument(id=entry.id, yaml=entry.text, updated_at=entry.updated_at, etag=entry.etag,
                          valid=entry.config is not None, problem=entry.problem)


async def _write(library: ConfigLibrary, entry_id: str, body: ConfigWrite, response: Response) -> ConfigDocument:
    entry, created = await anyio.to_thread.run_sync(library.write, entry_id, body.yaml, body.etag)
    if created:
        response.status_code = 201
    return document(entry)

# -----------------------
# Сценарии
# -----------------------


def scenario_item(entry: LibraryEntry, last_run: RunBrief | None) -> ScenarioItem:
    item = ScenarioItem(id=entry.id, updated_at=entry.updated_at, etag=entry.etag, valid=entry.config is not None,
                        last_run=last_run)
    cfg: ScenarioConfig | None = entry.config
    if cfg is None:
        return item
    return item.model_copy(update={
        "description": cfg.description,
        "tags": cfg.tags,
        "dataset": describe_dataset(cfg.dataset.source, cfg.dataset.id, cfg.dataset.revision),
        "collections": collection_refs(cfg),
        "pipeline": Pipeline(
            strategy=cfg.retrieval.strategy,
            sources=len(cfg.retrieval.sources) or 1,
            rerank=cfg.rerank.enabled,
            context=cfg.context.enabled,
            generation=cfg.generation.enabled,
            cache=cfg.cache.enabled,
            concurrency=cfg.run.load.params.get("concurrency"),
            repeats=cfg.run.repeats,
        ),
    })


@router.get("/scenarios", tags=["scenarios"])
async def list_scenarios(svc: Svc) -> list[ScenarioItem]:
    """Библиотека сценариев: все *.yaml каталога, невалидные — с valid: false."""
    items = []
    for entry in await anyio.to_thread.run_sync(svc.scenarios.list):
        page = await svc.registry.list_runs(RunQuery(scenario_id=entry.id, limit=1))
        last = page.items[0] if page.items else None
        items.append(scenario_item(entry, RunBrief(id=last.id, state=last.state, enqueued_at=last.enqueued_at,
                                                   ndcg_at_10=summary_column(last, "ndcg", 10))
                                   if last else None))
    return items


@router.post("/scenarios", tags=["scenarios"], status_code=201, responses=ERRORS)
async def create_scenario(body: ScenarioCreate, svc: Svc) -> ConfigDocument:
    """Новый сценарий из YAML (id — его name) или из snapshot прогона под новым name."""
    text = body.yaml
    if body.from_run is not None:
        record = await svc.registry.get_run(body.from_run)
        if record is None:
            raise problem_error(422, "run.not_found", "Request is invalid", f"run {body.from_run!r} not found",
                                path="from_run")
        run = {k: v for k, v in record.scenario["run"].items() if k != "artifacts_dir"}  # серверный путь
        text = snapshot_yaml(record.scenario, name=body.name, run=run)
    assert text is not None  # ScenarioCreate требует ровно один источник
    return document(await anyio.to_thread.run_sync(svc.scenarios.create, text))


@router.get("/scenarios/{scenario_id}", tags=["scenarios"], responses=ERRORS)
async def get_scenario(scenario_id: EntryId, svc: Svc) -> ConfigDocument:
    return document(await anyio.to_thread.run_sync(svc.scenarios.get, scenario_id))


@router.put("/scenarios/{scenario_id}", tags=["scenarios"], responses=ERRORS)
async def put_scenario(scenario_id: EntryId, body: ConfigWrite, response: Response, svc: Svc) -> ConfigDocument:
    """Сохранить: невалидное по схеме — черновиком (valid: false), битый YAML — 422, чужая правка — 409."""
    return await _write(svc.scenarios, scenario_id, body, response)


@router.delete("/scenarios/{scenario_id}", tags=["scenarios"], status_code=204, responses=ERRORS)
async def delete_scenario(scenario_id: EntryId, svc: Svc, etag: Annotated[str | None, Query()] = None) -> None:
    await anyio.to_thread.run_sync(svc.scenarios.delete, scenario_id, etag)

# -----------------------
# Профили индексов
# -----------------------


def profile_item(entry: LibraryEntry, last_build: BuildBrief | None) -> IndexProfileItem:
    item = IndexProfileItem(id=entry.id, updated_at=entry.updated_at, etag=entry.etag,
                            valid=entry.config is not None, last_build=last_build)
    cfg: IndexProfileConfig | None = entry.config
    if cfg is None:
        return item
    chunking: str = cfg.chunking.method
    if cfg.chunking.chunk_size is not None:
        chunking += f" {cfg.chunking.chunk_size}/{cfg.chunking.chunk_overlap}"
    return item.model_copy(update={
        "dataset": describe_dataset(cfg.dataset.source, cfg.dataset.id, cfg.dataset.revision),
        "chunking": chunking,
        "embedding_model": cfg.embedding.model,
        "targets": collection_refs(cfg),
    })


@router.get("/index-profiles", tags=["index profiles"])
async def list_profiles(svc: Svc) -> list[IndexProfileItem]:
    items = []
    for entry in await anyio.to_thread.run_sync(svc.profiles.list):
        page = await svc.registry.list_builds(BuildQuery(profile_id=entry.id, limit=1))
        last = page.items[0] if page.items else None
        items.append(profile_item(entry, BuildBrief(id=last.id, state=last.state, enqueued_at=last.enqueued_at)
                                  if last else None))
    return items


@router.post("/index-profiles", tags=["index profiles"], status_code=201, responses=ERRORS)
async def create_profile(body: IndexProfileCreate, svc: Svc) -> ConfigDocument:
    return document(await anyio.to_thread.run_sync(svc.profiles.create, body.yaml))


@router.get("/index-profiles/{profile_id}", tags=["index profiles"], responses=ERRORS)
async def get_profile(profile_id: EntryId, svc: Svc) -> ConfigDocument:
    return document(await anyio.to_thread.run_sync(svc.profiles.get, profile_id))


@router.put("/index-profiles/{profile_id}", tags=["index profiles"], responses=ERRORS)
async def put_profile(profile_id: EntryId, body: ConfigWrite, response: Response, svc: Svc) -> ConfigDocument:
    return await _write(svc.profiles, profile_id, body, response)


@router.delete("/index-profiles/{profile_id}", tags=["index profiles"], status_code=204, responses=ERRORS)
async def delete_profile(profile_id: EntryId, svc: Svc, etag: Annotated[str | None, Query()] = None) -> None:
    await anyio.to_thread.run_sync(svc.profiles.delete, profile_id, etag)
