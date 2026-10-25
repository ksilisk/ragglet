from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from ragglet.bench.dataset import EvalDataset, load_eval_dataset
from ragglet.cache.embedding_redis import redis_version
from ragglet.config.scenario import RetrievalSourceConfig, ScenarioConfig
from ragglet.core.errors import ServiceUnavailable
from ragglet.core.issues import Issue, error, raise_on_errors, warning
from ragglet.datasets.sources import describe_dataset
from ragglet.indexing.metadata import extract_index_metadata
from ragglet.retrieval.engine import IMPLEMENTED_STRATEGIES
from ragglet.storage.layout import (
    StorageLayout,
    StorageTarget,
    build_storage_layout,
    unimplemented_targets,
)
from ragglet.stores.qdrant_store_async import CollectionInfo

REBUILD_HINT = "Rebuild the index: ragglet index build <index_profile.yaml>"
SERVICE_TIMEOUT_S = 2.0

# -----------------------
# Проверки только по конфигу: до клиентов, датасета и моделей
# -----------------------

ScenarioCheck = Callable[[ScenarioConfig, StorageLayout], Iterable[Issue]]


def _used_sources(cfg: ScenarioConfig) -> list[tuple[int, RetrievalSourceConfig]]:
    sources = list(enumerate(cfg.retrieval.sources))
    return sources[:1] if cfg.retrieval.strategy == "single" else sources  # single берёт sources[0]


def retrieval_strategy_implemented(cfg: ScenarioConfig, layout: StorageLayout) -> Iterator[Issue]:
    if cfg.retrieval.strategy not in IMPLEMENTED_STRATEGIES:
        yield error(
            "retrieval.not_implemented",
            "retrieval.strategy",
            f"strategy {cfg.retrieval.strategy!r} is not implemented, use one of {list(IMPLEMENTED_STRATEGIES)}",
            path="retrieval.strategy",
        )


def retrieval_one_keyword_source(cfg: ScenarioConfig, layout: StorageLayout) -> Iterator[Issue]:
    keyword = [i for i, source in enumerate(cfg.retrieval.sources) if source.kind == "keyword"]
    for i in keyword[1:]:
        yield error(
            "retrieval.keyword_sources",
            f"retrieval.sources[{i}]",
            "only one keyword source is supported: the run builds one BM25 index",
            path=f"retrieval.sources[{i}]",
        )


def retrieval_known_backend_refs(cfg: ScenarioConfig, layout: StorageLayout) -> Iterator[Issue]:
    known = [target.id for target in layout.targets]
    for i, source in _used_sources(cfg):
        ref = source.params.get("backend_ref")
        if source.kind == "vector" and ref is not None and str(ref) not in known:
            path = f"retrieval.sources[{i}].params.backend_ref"
            yield error("retrieval.backend_ref", path, f"unknown backend_ref {ref!r}; storage targets: {known}",
                        path=path)


SCENARIO_CHECKS: list[ScenarioCheck] = [
    retrieval_strategy_implemented,
    retrieval_one_keyword_source,
    retrieval_known_backend_refs,
]


def check_scenario(cfg: ScenarioConfig) -> StorageLayout:
    """То, что схема пропускает, а прогон не умеет: форма storage, реализованные backends и стратегии."""
    layout = build_storage_layout(cfg.storage)
    raise_on_errors(
        f"Scenario {cfg.name!r} uses unsupported settings",
        [*unimplemented_targets(layout), *(issue for check in SCENARIO_CHECKS for issue in check(cfg, layout))],
    )
    return layout

# -----------------------
# Совместимость с индексом
# -----------------------


@dataclass(frozen=True)
class IndexContext:
    cfg: ScenarioConfig
    layout: StorageLayout
    dataset: EvalDataset
    collections: dict[str, CollectionInfo | None]  # target id -> состояние коллекции, None — её нет
    metadata: dict[str, dict[str, Any] | None]  # target id -> метаданные ragglet

    def complete_targets(self) -> Iterator[tuple[StorageTarget, dict[str, Any]]]:
        for target in self.layout.targets:
            meta = self.metadata[target.id]
            if meta is not None:
                yield target, meta


IndexCheck = Callable[[IndexContext], Iterable[Issue]]


def _where(target: StorageTarget) -> str:
    return f"target {target.id} ({target.collection})"


def index_collection_exists(ctx: IndexContext) -> Iterator[Issue]:
    for target in ctx.layout.targets:
        if ctx.collections[target.id] is None:
            yield error("index.no_collection", _where(target), "collection does not exist: the index was not built")


def index_metadata_present(ctx: IndexContext) -> Iterator[Issue]:
    for target in ctx.layout.targets:
        if ctx.collections[target.id] is not None and ctx.metadata[target.id] is None:
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
        for key, value in expected.items():
            if meta[key] != value:
                yield error(
                    "index.embedding",
                    _where(target),
                    f"{key}: index has {meta[key]!r}, scenario expects {value!r}",
                )


def index_optimized(ctx: IndexContext) -> Iterator[Issue]:
    for target in ctx.layout.targets:
        info = ctx.collections[target.id]
        if info is not None and info.status != "green":
            yield warning(
                "index.not_green",
                _where(target),
                f"collection status is {info.status}: optimization is running or pending, "
                "latency and recall may not be representative",
            )


INDEX_CHECKS: list[IndexCheck] = [
    index_collection_exists,
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
    collections: dict[str, CollectionInfo] = field(default_factory=dict)  # target id -> состояние коллекции

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


async def _reach_services(
        cfg: ScenarioConfig, layout: StorageLayout, stores: Mapping[str, Any]
) -> dict[str, CollectionInfo | None]:
    """Состояние коллекций целей; недоступный Qdrant или Redis — ошибка конфига, до побочных эффектов."""
    collections: dict[str, CollectionInfo | None] = {}
    issues: list[Issue] = []
    for target in layout.targets:
        try:
            collections[target.id] = await stores[target.id].collection_info()
        except ServiceUnavailable as e:
            issues.append(error("storage.unreachable", f"target {target.id} ({target.endpoint})",
                                f"Qdrant is not reachable: {e.reason}", path=f"{target.path}.endpoint"))
    emb = cfg.cache.embedding
    if cfg.cache.enabled and emb is not None:
        try:
            await redis_version(emb.endpoint, SERVICE_TIMEOUT_S)
        except ServiceUnavailable as e:
            issues.append(error("cache.unreachable", f"cache.embedding ({emb.endpoint})",
                                f"Redis is not reachable: {e.reason}", path="cache.embedding.endpoint"))
    raise_on_errors("Services are not reachable", issues, hint="Check the endpoints in the scenario")
    return collections


async def preflight(cfg: ScenarioConfig, stores: Mapping[str, Any]) -> Preflight:
    """Конфиг, датасет сценария, доступность сервисов и совместимость с индексом в `stores` (target id -> store)."""
    layout = check_scenario(cfg)
    dataset = load_eval_dataset(cfg.dataset)
    collections = await _reach_services(cfg, layout, stores)
    metadata = {
        target_id: extract_index_metadata(info.metadata) if info is not None else None
        for target_id, info in collections.items()
    }
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
        collections={tid: info for tid, info in collections.items() if info is not None},
    )
