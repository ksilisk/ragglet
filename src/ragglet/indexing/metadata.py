from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragglet.config.index_profile import IndexProfileConfig
from ragglet.config.scenario import ScenarioConfig


RAGGLET_METADATA_KEY = "ragglet"


@dataclass(frozen=True)
class MetadataMismatch:
    field: str
    expected: Any
    actual: Any


class IndexMetadataError(ValueError):
    pass


class MissingIndexMetadataError(IndexMetadataError):
    pass


class IndexMetadataMismatchError(IndexMetadataError):
    def __init__(self, mismatches: list[MetadataMismatch]):
        self.mismatches = mismatches
        details = ", ".join(
            f"{m.field}: expected {m.expected!r}, index has {m.actual!r}" for m in mismatches
        )
        super().__init__(f"Index metadata does not match scenario config: {details}")


def build_index_metadata(cfg: IndexProfileConfig) -> dict[str, Any]:
    return {
        RAGGLET_METADATA_KEY: {
            "index_profile_id": cfg.id,
            "embedding_model": cfg.embedding.model,
            "embedding_normalize": cfg.embedding.normalize,
            "chunking_method": cfg.chunking.method,
            "chunk_size": cfg.chunking.chunk_size,
            "chunk_overlap": cfg.chunking.chunk_overlap,
        }
    }


def extract_index_metadata(collection_metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not collection_metadata or RAGGLET_METADATA_KEY not in collection_metadata:
        raise MissingIndexMetadataError(
            "Qdrant collection has no Ragglet index metadata. Rebuild the index with "
            "`ragglet index build` before running benchmarks."
        )
    metadata = collection_metadata[RAGGLET_METADATA_KEY]
    if not isinstance(metadata, dict):
        raise MissingIndexMetadataError("Qdrant collection Ragglet metadata has invalid format.")
    return metadata


def validate_index_metadata_for_scenario(
    collection_metadata: dict[str, Any] | None,
    cfg: ScenarioConfig,
) -> None:
    metadata = extract_index_metadata(collection_metadata)
    expected = {
        "embedding_model": cfg.embedding.model,
        "embedding_normalize": cfg.embedding.normalize,
    }
    mismatches = [
        MetadataMismatch(field=field, expected=expected_value, actual=metadata.get(field))
        for field, expected_value in expected.items()
        if metadata.get(field) != expected_value
    ]
    if mismatches:
        raise IndexMetadataMismatchError(mismatches)
