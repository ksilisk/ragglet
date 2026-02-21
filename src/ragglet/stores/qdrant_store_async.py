from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm


@dataclass(frozen=True)
class Candidate:
    id: str
    score_raw: float
    payload: dict[str, Any] | None


class AsyncQdrantStore:
    def __init__(self, endpoint: str, collection: str):
        self.client = AsyncQdrantClient(url=endpoint)
        self.collection = collection

    async def search(self, vector: list[float], top_k: int, params: dict[str, Any] | None = None) -> list[Candidate]:
        params = params or {}
        ef_search = params.get("ef_search")

        search_params = None
        if ef_search is not None:
            search_params = qm.SearchParams(hnsw_ef=int(ef_search))

        res = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            search_params=search_params,
        )

        points = res.points if hasattr(res, "points") else []
        return [Candidate(id=str(p.id), score_raw=float(p.score), payload=p.payload) for p in points]

    async def close(self) -> None:
        await self.client.close()