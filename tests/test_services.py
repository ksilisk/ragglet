from __future__ import annotations

from qdrant_client.http import models as qm

from ragglet.cache.embedding_redis import RedisEmbeddingCache
from ragglet.stores.qdrant_store_async import AsyncQdrantStore


class _QdrantClient:
    async def get_collection(self, name):
        return qm.CollectionInfo(
            status=qm.CollectionStatus.YELLOW,
            optimizer_status=qm.OptimizersStatusOneOf.OK,
            points_count=53,
            indexed_vectors_count=0,
            segments_count=2,
            payload_schema={},
            config=qm.CollectionConfig(
                params=qm.CollectionParams(vectors=qm.VectorParams(size=384, distance=qm.Distance.COSINE)),
                hnsw_config=qm.HnswConfig(m=16, ef_construct=128, full_scan_threshold=10000),
                optimizer_config=qm.OptimizersConfig(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=0,
                    flush_interval_sec=5,
                ),
                metadata={"ragglet": {"index_profile_id": "demo"}},
            ),
        )

    async def info(self):
        return qm.VersionInfo(title="qdrant", version="1.17.0")


async def test_qdrant_store_reports_actual_collection_state():
    store = AsyncQdrantStore("http://q:6333", "chunks_v1")
    store.client = _QdrantClient()

    info = await store.collection_info()
    assert info.metadata == {"ragglet": {"index_profile_id": "demo"}}
    assert info.params == {"vector_size": 384, "distance": "Cosine", "hnsw": {"m": 16, "ef_construct": 128}}
    assert await store.describe() == {
        "type": "qdrant",
        "version": "1.17.0",
        "status": "yellow",
        "points": 53,
        "indexed_vectors": 0,
        "segments": 2,
    }


class _Redis:
    async def info(self, section):
        assert section == "server"
        return {"redis_version": "7.4.1", "redis_mode": "standalone"}


async def test_redis_cache_reports_server_version():
    cache = RedisEmbeddingCache("redis://localhost:6379/0", 60, "m", "rev", True)
    cache.redis = _Redis()

    assert await cache.describe() == {"type": "redis", "version": "7.4.1"}
