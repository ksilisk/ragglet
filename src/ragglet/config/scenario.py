from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ConfigDict, field_validator


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

    id: str
    limit: int = Field(2000, ge=1)
    query_field: str = "question"
    truth_field: str = "relevant"


class StorageBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["qdrant", "pgvector", "milvus", "weaviate"]
    endpoint: str
    collection: str


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["single", "sharded", "replicated"] = "single"
    backend: StorageBackendConfig


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "all-MiniLM-L6-v2"
    normalize: bool = True


class RetrievalSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["vector", "keyword", "hybrid"] = "vector"
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


class RerankConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: str | None = None
    input_top_k: int = 30
    output_top_n: int = 10

    @field_validator("model")
    @classmethod
    def _model_required_if_enabled(cls, v: str | None, info):
        if info.data.get("enabled") and not v:
            raise ValueError("rerank.model is required when rerank.enabled=true")
        return v


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = 1200
    policy: Literal["top_n", "adaptive", "threshold"] = "top_n"
    threshold: float = 0.0
    include_metadata: bool = True


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: str = "openai_compatible"
    model: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 300

    @field_validator("model")
    @classmethod
    def _model_required_if_enabled(cls, v: str | None, info):
        if info.data.get("enabled") and not v:
            raise ValueError("generation.model is required when generation.enabled=true")
        return v


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

    @field_validator("embedding")
    @classmethod
    def _embedding_required_if_enabled(cls, v: CacheEmbeddingConfig | None, info):
        # Optional: enforce if cache enabled
        if info.data.get("enabled") and v is None:
            # Not strictly required, but helps keep configs meaningful
            raise ValueError("cache.embedding must be set when cache.enabled=true (at least one layer)")
        return v


class MetricsQualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recall_at_k: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 30])
    mrr_at_k: int = 30


class MetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results_path: str = "results/{name}_{timestamp}.parquet"
    latency_steps: list[str] = Field(default_factory=lambda: ["embed", "retrieve", "total"])
    percentiles: list[int] = Field(default_factory=lambda: [50, 95, 99])
    quality: MetricsQualityConfig = Field(default_factory=MetricsQualityConfig)


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


def load_scenario(path: Path) -> ScenarioConfig:
    data = _load_yaml(path)
    cfg = ScenarioConfig.model_validate(data)
    return cfg