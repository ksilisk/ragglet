from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragglet.bench.dataset import QueryItem, load_jsonl_dataset
from ragglet.config.scenario import ScenarioConfig
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
    queries_checked: int
    truth_ids_checked: int
    indexed_external_ids: int
    missing: list[MissingTruthId]

    @property
    def ok(self) -> bool:
        return not self.missing

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


async def _qdrant_external_ids(cfg: ScenarioConfig, batch_size: int = 1024) -> set[str]:
    store = AsyncQdrantStore(cfg.storage.backend.endpoint, cfg.storage.backend.collection)
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


async def load_index_external_ids(cfg: ScenarioConfig) -> set[str]:
    if cfg.storage.backend.type == "qdrant":
        return await _qdrant_external_ids(cfg)
    raise NotImplementedError(
        f"Dataset validation is not implemented yet for backend '{cfg.storage.backend.type}'"
    )


async def validate_scenario_dataset(cfg: ScenarioConfig) -> DatasetValidationReport:
    items = load_jsonl_dataset(
        Path(cfg.dataset.id),
        cfg.dataset.query_field,
        cfg.dataset.truth_field,
        cfg.dataset.limit,
    )
    indexed_external_ids = await load_index_external_ids(cfg)
    missing = find_missing_truth_ids(items, indexed_external_ids)
    truth_ids_checked = sum(len(item.truth_ids) for item in items)

    return DatasetValidationReport(
        scenario=cfg.name,
        dataset_path=cfg.dataset.id,
        backend_type=cfg.storage.backend.type,
        collection=cfg.storage.backend.collection,
        queries_checked=len(items),
        truth_ids_checked=truth_ids_checked,
        indexed_external_ids=len(indexed_external_ids),
        missing=missing,
    )
