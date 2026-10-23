from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from ragglet.bench.dataset import EvalDataset, load_eval_dataset
from ragglet.config.scenario import ScenarioConfig
from ragglet.core.issues import Issue, error, raise_on_errors, warning
from ragglet.datasets.sources import describe_dataset
from ragglet.indexing.metadata import extract_index_metadata
from ragglet.storage.layout import StorageLayout, StorageTarget, build_storage_layout
from ragglet.stores.qdrant_store_async import AsyncQdrantStore, CollectionInfo

REBUILD_HINT = "Rebuild the index: ragglet index build <index_profile.yaml>"


@dataclass(frozen=True)
class IndexContext:
    cfg: ScenarioConfig
    layout: StorageLayout
    dataset: EvalDataset
    collections: dict[str, CollectionInfo]  # target id -> состояние коллекции
    metadata: dict[str, dict[str, Any] | None]  # target id -> метаданные ragglet

    def complete_targets(self) -> Iterator[tuple[StorageTarget, dict[str, Any]]]:
        for target in self.layout.targets:
            meta = self.metadata[target.id]
            if meta is not None:
                yield target, meta


IndexCheck = Callable[[IndexContext], Iterable[Issue]]


def _where(target: StorageTarget) -> str:
    return f"target {target.id} ({target.collection})"


def index_metadata_present(ctx: IndexContext) -> Iterator[Issue]:
    for target in ctx.layout.targets:
        if ctx.metadata[target.id] is None:
            yield error(
                "index.no_metadata",
                _where(target),
                "no ragglet metadata: the build did not finish or predates the current contract",
            )


def index_same_build(ctx: IndexContext) -> Iterator[Issue]:
    builds = sorted({meta["built_at"] for _, meta in ctx.complete_targets()})
    if len(builds) > 1:
        yield error("index.mixed_builds", "storage", f"targets come from different builds: {', '.join(builds)}")


def index_storage_layout(ctx: IndexContext) -> Iterator[Issue]:
    expected = (ctx.layout.mode, len(ctx.layout.targets))
    for target, meta in ctx.complete_targets():
        actual = (meta["storage_mode"], meta["storage_backend_count"])
        if actual != expected:
            yield error(
                "index.storage_layout",
                _where(target),
                f"index is {actual[0]} with {actual[1]} backend(s), "
                f"scenario expects {expected[0]} with {expected[1]}",
            )


def index_corpus(ctx: IndexContext) -> Iterator[Issue]:
    scenario = describe_dataset(ctx.cfg.dataset.source, ctx.cfg.dataset.id, ctx.dataset.revision)
    for target, meta in ctx.complete_targets():
        if meta["corpus_sha256"] != ctx.dataset.corpus_sha256:
            yield error(
                "index.corpus",
                _where(target),
                f"corpus differs: index built from {meta['dataset_source']}:{meta['dataset_id']} "
                f"({meta['corpus_sha256'][:12]}), scenario dataset {scenario} "
                f"({ctx.dataset.corpus_sha256[:12]})",
            )


def index_embedding(ctx: IndexContext) -> Iterator[Issue]:
    expected = {
        "embedding_model": ctx.cfg.embedding.model,
        "embedding_normalize": ctx.cfg.embedding.normalize,
    }
    for target, meta in ctx.complete_targets():
        for field, value in expected.items():
            if meta[field] != value:
                yield error(
                    "index.embedding",
                    _where(target),
                    f"{field}: index has {meta[field]!r}, scenario expects {value!r}",
                )


def index_optimized(ctx: IndexContext) -> Iterator[Issue]:
    for target in ctx.layout.targets:
        status = ctx.collections[target.id].status
        if status != "green":
            yield warning(
                "index.not_green",
                _where(target),
                f"collection status is {status}: optimization is running or pending, "
                "latency and recall may not be representative",
            )


INDEX_CHECKS: list[IndexCheck] = [
    index_metadata_present,
    index_same_build,
    index_storage_layout,
    index_corpus,
    index_embedding,
    index_optimized,
]


@dataclass(frozen=True)
class Preflight:
    dataset: EvalDataset
    index: dict[str, Any]  # метаданные ragglet primary target (у всех targets сборки они равны)
    params: dict[str, Any]  # параметры коллекции primary target
    warnings: list[Issue]  # датасета и индекса

    def manifest(self) -> dict[str, Any]:
        """Только то, чего нет в scenario.snapshot."""
        return {
            "dataset": {
                "revision": self.dataset.revision,
                "corpus_sha256": self.dataset.corpus_sha256,
                "eval_sha256": self.dataset.eval_sha256,
            },
            "index": {
                "profile_id": self.index["index_profile_id"],
                "built_at": self.index["built_at"],
                "embedding_revision": self.index["embedding_revision"],
                "chunking": {
                    "method": self.index["chunking_method"],
                    "chunk_size": self.index["chunk_size"],
                    "chunk_overlap": self.index["chunk_overlap"],
                },
                "params": self.params,
                "build_env": self.index["build_env"],
            },
        }


async def preflight(cfg: ScenarioConfig, stores: Mapping[str, Any]) -> Preflight:
    """Датасет сценария + совместимость с индексом в `stores` (target id -> store)."""
    layout = build_storage_layout(cfg.storage)
    dataset = load_eval_dataset(cfg.dataset)
    collections = {target.id: await stores[target.id].collection_info() for target in layout.targets}
    metadata = {target_id: extract_index_metadata(info.metadata) for target_id, info in collections.items()}
    ctx = IndexContext(cfg=cfg, layout=layout, dataset=dataset, collections=collections, metadata=metadata)
    index_warnings = raise_on_errors(
        f"Index is not compatible with scenario {cfg.name!r}",
        (issue for check in INDEX_CHECKS for issue in check(ctx)),
        hint=REBUILD_HINT,
    )
    primary = layout.primary.id
    return Preflight(
        dataset=dataset,
        index=metadata[primary],
        params=collections[primary].params,
        warnings=[*dataset.warnings, *index_warnings],
    )


async def validate_scenario(cfg: ScenarioConfig) -> Preflight:
    """Preflight без прогона запросов (`bench validate`)."""
    layout = build_storage_layout(cfg.storage)
    if any(target.type != "qdrant" for target in layout.targets):
        raise NotImplementedError("Validation is implemented only for qdrant backends")
    stores = {target.id: AsyncQdrantStore(target.endpoint, target.collection) for target in layout.targets}
    try:
        return await preflight(cfg, stores)
    finally:
        for store in stores.values():
            await store.close()
