from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


DISTANCE_MAP = {
    "cosine": qm.Distance.COSINE,
    "dot": qm.Distance.DOT,
    "euclid": qm.Distance.EUCLID,
}


@dataclass(frozen=True)
class QdrantIndexSpec:
    endpoint: str
    collection: str
    vector_size: int
    distance: str
    hnsw_m: int
    hnsw_ef_construct: int


class QdrantIndexer:
    def __init__(self, spec: QdrantIndexSpec):
        self.client = QdrantClient(url=spec.endpoint)
        self.spec = spec

    def recreate_collection(self) -> None:
        # drop if exists
        collections = {c.name for c in self.client.get_collections().collections}
        if self.spec.collection in collections:
            self.client.delete_collection(self.spec.collection)

        self.client.create_collection(
            collection_name=self.spec.collection,
            vectors_config=qm.VectorParams(
                size=self.spec.vector_size,
                distance=DISTANCE_MAP[self.spec.distance],
            ),
            hnsw_config=qm.HnswConfigDiff(
                m=self.spec.hnsw_m,
                ef_construct=self.spec.hnsw_ef_construct,
            ),
        )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        points = [
            qm.PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.spec.collection, points=points)