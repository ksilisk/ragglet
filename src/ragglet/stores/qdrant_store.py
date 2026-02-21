from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

@dataclass(frozen=True)
class Candidate:
    id: str
    score_raw: float
    payload: dict[str, Any] | None

class QdrantStore:
    def __init__(self, endpoint: str, collection: str):
        self.client = QdrantClient(url=endpoint)
        self.collection = collection

    def search(
            self,
            vector: list[float],
            top_k: int,
            qfilter: qm.Filter | None = None,
            params: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        params = params or {}
        ef_search = params.get("ef_search")

        search_params = None
        if ef_search is not None:
            search_params = qm.SearchParams(hnsw_ef=int(ef_search))

        res = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=qfilter,
            search_params=search_params,
        )

        points = res.points if hasattr(res, "points") else []
        out: list[Candidate] = []
        for p in points:
            out.append(Candidate(id=str(p.id), score_raw=float(p.score), payload=p.payload))
        return out