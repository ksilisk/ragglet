from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException

from ragglet.core.errors import ServiceUnavailable
from ragglet.indexing.metadata import chunk_point_id


@dataclass(frozen=True)
class Candidate:
    id: str
    score_raw: float
    payload: dict[str, Any] | None


@dataclass(frozen=True)
class CollectionInfo:
    metadata: dict[str, Any] | None
    params: dict[str, Any] | None  # фактические параметры коллекции; None — не один безымянный вектор
    status: str  # green | yellow (оптимизация идёт) | grey (ждёт запуска) | red
    points: int | None
    indexed_vectors: int | None  # 0 — HNSW не построен, поиск полным перебором
    segments: int


class AsyncQdrantStore:
    def __init__(self, endpoint: str, collection: str):
        self.client = AsyncQdrantClient(url=endpoint)
        self.endpoint = endpoint
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

    async def collection_info(self) -> CollectionInfo | None:
        """None — коллекции нет; ServiceUnavailable — Qdrant не ответил."""
        if not await _reachable(self.endpoint, self.client.collection_exists(self.collection)):
            return None
        return await _reachable(self.endpoint, _collection_info(self.client, self.collection))

    async def describe(self) -> dict[str, Any]:
        server = await self.client.info()
        info = await self.collection_info()
        if info is None:
            raise ValueError(f"Collection {self.collection!r} not found")
        return {
            "type": "qdrant",
            "version": server.version,
            "status": info.status,
            "points": info.points,
            "indexed_vectors": info.indexed_vectors,
            "segments": info.segments,
        }

    async def payloads(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """payload чанков по chunk_id, без векторов; ServiceUnavailable — Qdrant не ответил."""
        points = await _reachable(self.endpoint, self.client.retrieve(
            self.collection, ids=[chunk_point_id(c) for c in chunk_ids], with_payload=True, with_vectors=False,
        ))
        return {str(p.payload["chunk_id"]): p.payload for p in points if p.payload and "chunk_id" in p.payload}

    async def scroll_payloads(self, limit: int = 256, offset=None, with_payload: Any = True):
        res = await self.client.scroll(
            collection_name=self.collection,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False,
        )
        points, next_offset = res
        payloads = [p.payload for p in points if p.payload]
        return payloads, next_offset


@dataclass(frozen=True)
class ServerCollections:
    version: str
    collections: dict[str, CollectionInfo]  # имя -> состояние


async def list_collections(endpoint: str) -> ServerCollections:
    """Все коллекции сервера одним клиентом; ServiceUnavailable — Qdrant не ответил."""
    client = AsyncQdrantClient(url=endpoint)
    try:
        server = await _reachable(endpoint, client.info())
        names = [c.name for c in (await _reachable(endpoint, client.get_collections())).collections]
        infos = {name: await _reachable(endpoint, _collection_info(client, name)) for name in sorted(names)}
        return ServerCollections(version=server.version, collections=infos)
    finally:
        await client.close()


async def _collection_info(client: AsyncQdrantClient, name: str) -> CollectionInfo:
    info = await client.get_collection(name)
    vectors = info.config.params.vectors
    params = None
    if isinstance(vectors, qm.VectorParams):  # иначе коллекция не ragglet: именованные векторы
        hnsw = info.config.hnsw_config
        params = {
            "vector_size": vectors.size,
            "distance": vectors.distance.value,
            "hnsw": {"m": hnsw.m, "ef_construct": hnsw.ef_construct},
        }
    return CollectionInfo(
        metadata=info.config.metadata,
        params=params,
        status=info.status.value,
        points=info.points_count,
        indexed_vectors=info.indexed_vectors_count,
        segments=info.segments_count,
    )


async def _reachable[T](endpoint: str, call: Awaitable[T]) -> T:
    # клиент Qdrant заворачивает ошибки транспорта в ResponseHandlingException
    try:
        return await call
    except (ResponseHandlingException, httpx.TransportError) as e:
        cause = e.source if isinstance(e, ResponseHandlingException) else e
        raise ServiceUnavailable(endpoint, f"{type(cause).__name__}: {cause}") from e
