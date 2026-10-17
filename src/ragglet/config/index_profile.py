from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Index profile YAML root must be a mapping/object")
    return data


class ChunkingConfig(BaseModel):
    """
    method="none": документ индексируется целиком одной точкой (стандартная BEIR-постановка).
    chunk_size/chunk_overlap для него запрещены, для "recursive" по умолчанию 800/120.
    """
    model_config = ConfigDict(extra="forbid")

    method: Literal["recursive", "none"] = "recursive"
    chunk_size: int | None = Field(None, gt=0)
    chunk_overlap: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _validate_sizes(self) -> "ChunkingConfig":
        if self.method == "none":
            if self.chunk_size is not None or self.chunk_overlap is not None:
                raise ValueError(
                    "chunking.chunk_size/chunk_overlap are not allowed when chunking.method='none'"
                )
            return self
        if self.chunk_size is None:
            self.chunk_size = 800
        if self.chunk_overlap is None:
            self.chunk_overlap = 120
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    normalize: bool = True
    batch_size: int = 64


class HNSWConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    m: int = 16
    ef_construct: int = 128


class IndexParamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)


class StorageBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    type: Literal["qdrant", "pgvector", "milvus", "weaviate"] = "qdrant"
    endpoint: str
    collection: str


class StorageRoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: Literal["chunk_id", "doc_id"] = "chunk_id"


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["single", "sharded", "replicated"] = "single"
    type: Literal["qdrant", "pgvector", "milvus", "weaviate"] = "qdrant"
    endpoint: str | None = None
    collection: str | None = None
    backend: StorageBackendConfig | None = None
    backends: list[StorageBackendConfig] = Field(default_factory=list)
    routing: StorageRoutingConfig = Field(default_factory=StorageRoutingConfig)

    @model_validator(mode="after")
    def _validate_storage_shape(self) -> "StorageConfig":
        if self.backend is None and (self.endpoint is None or self.collection is None):
            raise ValueError("storage.endpoint and storage.collection are required when storage.backend is not set")
        if self.mode != "single" and not self.backends:
            raise ValueError("storage.backends must be provided when storage.mode is not 'single'")
        return self


class DatasetSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["local", "hf"] = "local"
    id: str


class IndexProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    dataset: DatasetSourceConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig
    storage: StorageConfig
    index_params: IndexParamsConfig = Field(default_factory=IndexParamsConfig)


def load_index_profile(path: Path) -> IndexProfileConfig:
    return IndexProfileConfig.model_validate(_load_yaml(path))
