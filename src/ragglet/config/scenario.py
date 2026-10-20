from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from ragglet.bench.metrics import METRICS
from ragglet.datasets.sources import check_source


# -----------------------
# Small helpers
# -----------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Scenario YAML root must be a mapping/object")
    return data


# -----------------------
# Schema blocks
# -----------------------

class TimeoutsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_ms: int = 6000
    retrieval_ms: int = 1500
    rerank_ms: int = 800
    llm_ms: int = 3500


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repeats: int = Field(1, ge=1, description="How many times to repeat the whole dataset run")
    shuffle: bool = Field(True, description="Shuffle dataset items before running")
    save_per_query: bool = Field(True, description="Save per-query raw results")
    artifacts_dir: str = Field("results/runs", description="Directory for run artifacts")


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["parquet", "csv", "both"] = "parquet"
    include_debug_fields: bool = False


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = "local"
    id: str
    revision: str | None = None
    split: str = "test"
    limit: int = Field(2000, ge=1)

    @model_validator(mode="after")
    def _known_source(self) -> DatasetConfig:
        check_source(self.source, self.revision)
        return self


class StorageBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    type: Literal["qdrant", "pgvector", "milvus", "weaviate"]
    endpoint: str
    collection: str


class StorageRoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: Literal["chunk_id", "doc_id"] = "chunk_id"


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["single", "sharded", "replicated"] = "single"
    backend: StorageBackendConfig
    backends: list[StorageBackendConfig] = Field(default_factory=list)
    routing: StorageRoutingConfig = Field(default_factory=StorageRoutingConfig)

    @model_validator(mode="after")
    def _validate_storage_shape(self) -> "StorageConfig":
        if self.mode != "single" and not self.backends:
            raise ValueError("storage.backends must be provided when storage.mode is not 'single'")
        return self


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "all-MiniLM-L6-v2"
    normalize: bool = True


class RetrievalSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["vector", "keyword"] = "vector"
    weight: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)


class RetrievalMergeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["rrf", "weighted_sum", "interleave"] = "rrf"
    rrf_k: int = 60
    deduplicate: bool = True


class RetrievalConfig(BaseModel):
    """
    Note: YAML uses field name `async`, but in Python it's reserved.
    We store it as `async_enabled` with alias="async".
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    strategy: Literal["single", "fanout_merge", "cascade"] = "single"
    top_k: int = Field(30, ge=1)
    async_enabled: bool = Field(True, alias="async")
    sources: list[RetrievalSourceConfig] = Field(default_factory=list)
    merge: RetrievalMergeConfig = Field(default_factory=RetrievalMergeConfig)

    @field_validator("sources")
    @classmethod
    def _sources_required_for_fanout(cls, v: list[RetrievalSourceConfig], info):
        # For fanout_merge/cascade we expect >=1 sources; for single we allow empty => implicit default source.
        strategy = info.data.get("strategy", "single")
        if strategy != "single" and len(v) == 0:
            raise ValueError("retrieval.sources must be provided when strategy != 'single'")
        return v

    @model_validator(mode="after")
    def _single_keyword_not_supported(self) -> "RetrievalConfig":
        if self.strategy == "single" and any(s.kind == "keyword" for s in self.sources):
            raise ValueError(
                "retrieval.strategy='single' supports only vector sources; "
                "keyword single retrieval is not implemented yet"
            )
        return self


class RerankConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: Literal["cross_encoder", "noop"] = "cross_encoder"
    model: str | None = None
    input_top_k: int = 30
    output_top_n: int = 10

    @model_validator(mode="after")
    def _model_required_if_enabled(self) -> "RerankConfig":
        if self.enabled and self.provider == "cross_encoder" and not self.model:
            raise ValueError("rerank.model is required when rerank.enabled=true and provider='cross_encoder'")
        return self


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_tokens: int = 1200
    policy: Literal["top_n", "adaptive", "threshold"] = "top_n"
    threshold: float = 0.0
    include_metadata: bool = True


class GenerationSaveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: bool = False
    answer: bool = True
    raw_response: bool = False


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: Literal["noop", "openai_compatible"] = "noop"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 300
    save: GenerationSaveConfig = Field(default_factory=GenerationSaveConfig)

    @model_validator(mode="after")
    def _model_required_if_enabled(self) -> "GenerationConfig":
        if self.enabled and not self.model:
            raise ValueError("generation.model is required when generation.enabled=true")
        return self


class CacheEmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["redis"] = "redis"
    endpoint: str
    ttl_seconds: int = 86400


class CacheRetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["memory_lru"] = "memory_lru"
    size: int = 10_000


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    embedding: CacheEmbeddingConfig | None = None
    retrieval: CacheRetrievalConfig | None = None

    @model_validator(mode="after")
    def _at_least_one_layer_if_enabled(self) -> "CacheConfig":
        if self.enabled and self.embedding is None and self.retrieval is None:
            raise ValueError("At least one cache layer must be set when cache.enabled=true")
        return self


class MetricsQualityConfig(BaseModel):
    """
    Каждая метрика уровня считается для каждого k из `k`.
    doc — уникальные документы (как в BEIR); chunk — уникальные чанки финального списка,
    релевантность наследуется от документа. chunk-level по умолчанию выключен.
    """
    model_config = ConfigDict(extra="forbid")

    k: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    doc: list[str] = Field(default_factory=lambda: ["ndcg", "recall", "hit", "mrr"])
    chunk: list[str] = Field(default_factory=list)

    @field_validator("k")
    @classmethod
    def _k_positive_unique(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("metrics.quality.k must contain at least one value")
        if any(k < 1 for k in v):
            raise ValueError("metrics.quality.k values must be >= 1")
        if len(set(v)) != len(v):
            raise ValueError("metrics.quality.k values must be unique")
        return sorted(v)

    @field_validator("doc", "chunk")
    @classmethod
    def _known_metrics(cls, v: list[str], info) -> list[str]:
        level = info.field_name
        unknown = [name for name in v if name not in METRICS]
        if unknown:
            raise ValueError(
                f"metrics.quality.{level}: unknown metrics {unknown}; available: {sorted(METRICS)}"
            )
        if len(set(v)) != len(v):
            raise ValueError(f"metrics.quality.{level} values must be unique")
        if level == "chunk":
            doc_only = [name for name in v if METRICS[name].needs_ideal]
            if doc_only:
                raise ValueError(
                    f"metrics.quality.chunk: {doc_only} need the total number of relevant units, "
                    "which is unknown for chunks with document-level qrels; use them in "
                    "metrics.quality.doc"
                )
        return v


class MetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latency_steps: list[str] = Field(default_factory=lambda: ["embed", "retrieve", "total"])
    percentiles: list[int] = Field(default_factory=lambda: [50, 95, 99])
    quality: MetricsQualityConfig = Field(default_factory=MetricsQualityConfig)

    @field_validator("percentiles")
    @classmethod
    def _percentiles_in_range(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("metrics.percentiles must contain at least one value")
        invalid = [p for p in v if p < 0 or p > 100]
        if invalid:
            raise ValueError("metrics.percentiles values must be between 0 and 100")
        return v


# -----------------------
# Root config
# -----------------------

class ScenarioConfig(BaseModel):
    """
    Final Scenario schema (v1).
    """
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)

    seed: int = 42
    timeouts: TimeoutsConfig = Field(default_factory=TimeoutsConfig)

    run: RunConfig = Field(default_factory=RunConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    dataset: DatasetConfig
    storage: StorageConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @model_validator(mode="after")
    def _generation_requires_context(self) -> "ScenarioConfig":
        if self.generation.enabled and not self.context.enabled:
            raise ValueError("context.enabled must be true when generation.enabled=true")
        return self

    @model_validator(mode="after")
    def _quality_k_within_final_depth(self) -> "ScenarioConfig":
        # иначе @k тихо считается по более короткому списку
        depth, source = self.retrieval.top_k, "retrieval.top_k"
        if self.rerank.enabled:
            for value, name in (
                (self.rerank.input_top_k, "rerank.input_top_k"),
                (self.rerank.output_top_n, "rerank.output_top_n"),
            ):
                if value < depth:
                    depth, source = value, name
        max_k = max(self.metrics.quality.k)
        if max_k > depth:
            raise ValueError(
                f"metrics.quality.k={max_k} exceeds final list depth {depth} ({source})"
            )
        return self


def load_scenario(path: Path) -> ScenarioConfig:
    data = _load_yaml(path)
    cfg = ScenarioConfig.model_validate(data)
    return cfg
