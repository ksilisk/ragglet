from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Index profile YAML root must be a mapping/object")
    return data


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["recursive"] = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 120


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


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["qdrant", "pgvector", "milvus", "weaviate"] = "qdrant"
    endpoint: str
    collection: str


class IndexProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    corpus: str
    parser: Literal["plain_text"] = "plain_text"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig
    storage: StorageConfig
    index_params: IndexParamsConfig = Field(default_factory=IndexParamsConfig)


def load_index_profile(path: Path) -> IndexProfileConfig:
    return IndexProfileConfig.model_validate(_load_yaml(path))