from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragglet.bench.dataset import QueryItem, load_jsonl_dataset
from ragglet.config.scenario import ScenarioConfig
from ragglet.storage.layout import StorageTarget, build_storage_layout
from ragglet.stores.qdrant_store_async import AsyncQdrantStore


@dataclass(frozen=True)
class MissingTruthId:
    qid: str
    truth_id: str


@dataclass(frozen=True)
class DatasetValidationReport:
    scenario: str
    dataset_path: str
    backend_type: str
    collection: str
    storage_mode: str
    backend_counts: dict[str, int]
    queries_checked: int
    truth_ids_checked: int
    indexed_external_ids: int
    missing: list[MissingTruthId]
    missing_by_backend: dict[str, list[MissingTruthId]]
    duplicate_external_ids: int

    @property
    def ok(self) -> bool:
        return not self.missing and not self.missing_by_backend

    @property
    def missing_count(self) -> int:
        return len(self.missing)


def find_missing_truth_ids(
    items: list[QueryItem],
    indexed_external_ids: set[str],
) -> list[MissingTruthId]:
    missing: list[MissingTruthId] = []
    for item in items:
        for truth_id in item.truth_ids:
            if truth_id not in indexed_external_ids:
                missing.append(MissingTruthId(qid=item.qid, truth_id=truth_id))
    return missing


async def _qdrant_external_ids(target: StorageTarget, batch_size: int = 1024) -> set[str]:
    store = AsyncQdrantStore(target.endpoint, target.collection)
    external_ids: set[str] = set()
    offset = None

    try:
        while True:
            payloads, offset = await store.scroll_payloads(
                limit=batch_size,
                offset=offset,
                with_payload=["external_id"],
            )
            for payload in payloads:
                external_id = payload.get("external_id")
                if external_id is not None:
                    external_ids.add(str(external_id))
            if offset is None:
                break
    finally:
        await store.close()

    return external_ids


async def load_index_external_ids_by_backend(cfg: ScenarioConfig) -> dict[str, set[str]]:
    layout = build_storage_layout(cfg.storage)
    if any(target.type != "qdrant" for target in layout.targets):
        raise NotImplementedError("Dataset validation is implemented only for qdrant backends")
    return {target.id: await _qdrant_external_ids(target) for target in layout.targets}


async def load_index_external_ids(cfg: ScenarioConfig) -> set[str]:
    by_backend = await load_index_external_ids_by_backend(cfg)
    return set().union(*by_backend.values()) if by_backend else set()


async def validate_scenario_dataset(cfg: ScenarioConfig) -> DatasetValidationReport:
    items = load_jsonl_dataset(
        Path(cfg.dataset.id),
        cfg.dataset.query_field,
        cfg.dataset.truth_field,
        cfg.dataset.limit,
    )
    layout = build_storage_layout(cfg.storage)
    by_backend = await load_index_external_ids_by_backend(cfg)
    indexed_external_ids = set().union(*by_backend.values()) if by_backend else set()
    missing = find_missing_truth_ids(items, indexed_external_ids)
    truth_ids_checked = sum(len(item.truth_ids) for item in items)
    backend_counts = {backend_id: len(ids) for backend_id, ids in by_backend.items()}
    duplicate_external_ids = sum(len(ids) for ids in by_backend.values()) - len(indexed_external_ids)

    missing_by_backend: dict[str, list[MissingTruthId]] = {}
    if layout.mode == "replicated":
        for backend_id, ids in by_backend.items():
            backend_missing = find_missing_truth_ids(items, ids)
            if backend_missing:
                missing_by_backend[backend_id] = backend_missing

    return DatasetValidationReport(
        scenario=cfg.name,
        dataset_path=cfg.dataset.id,
        backend_type=cfg.storage.backend.type,
        collection=", ".join(f"{target.id}:{target.collection}" for target in layout.targets),
        storage_mode=layout.mode,
        backend_counts=backend_counts,
        queries_checked=len(items),
        truth_ids_checked=truth_ids_checked,
        indexed_external_ids=len(indexed_external_ids),
        missing=missing,
        missing_by_backend=missing_by_backend,
        duplicate_external_ids=duplicate_external_ids,
    )
