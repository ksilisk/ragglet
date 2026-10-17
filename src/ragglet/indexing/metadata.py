from __future__ import annotations

from typing import Any

from ragglet.config.index_profile import IndexProfileConfig

RAGGLET_METADATA_KEY = "ragglet"

REQUIRED_FIELDS = (
    "index_profile_id",
    "built_at",
    "dataset_source",
    "dataset_id",
    "corpus_sha256",
    "embedding_model",
    "embedding_revision",
    "embedding_normalize",
    "chunking_method",
    "chunk_size",
    "chunk_overlap",
    "storage_mode",
    "storage_backend_count",
)


def build_index_metadata(
    cfg: IndexProfileConfig,
    *,
    built_at: str,
    corpus_sha256: str,
    embedding_revision: str | None,
    storage_mode: str,
    storage_backend_count: int,
) -> dict[str, Any]:
    return {
        RAGGLET_METADATA_KEY: {
            "index_profile_id": cfg.id,
            "built_at": built_at,
            "dataset_source": cfg.dataset.source,
            "dataset_id": cfg.dataset.id,
            "corpus_sha256": corpus_sha256,
            "embedding_model": cfg.embedding.model,
            "embedding_revision": embedding_revision,
            "embedding_normalize": cfg.embedding.normalize,
            "chunking_method": cfg.chunking.method,
            "chunk_size": cfg.chunking.chunk_size,
            "chunk_overlap": cfg.chunking.chunk_overlap,
            "storage_mode": storage_mode,
            "storage_backend_count": storage_backend_count,
        }
    }


def extract_index_metadata(collection_metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Метаданные ragglet коллекции. Пишутся одним вызовом после последнего upsert, поэтому
    None означает, что сборка не завершилась или индекс собран по старому контракту.
    """
    metadata = (collection_metadata or {}).get(RAGGLET_METADATA_KEY)
    if not isinstance(metadata, dict) or any(f not in metadata for f in REQUIRED_FIELDS):
        return None
    return metadata
