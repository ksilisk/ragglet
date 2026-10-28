from __future__ import annotations

import uuid
from typing import Any, Literal

from ragglet.cache.embedding_redis import RedisEmbeddingCache
from ragglet.cache.retrieval_lru import LRURetrievalCache, RetrievalCacheEntry
from ragglet.config.scenario import ScenarioConfig
from ragglet.retrieval.merge import RetrievedItem

Lookup = Literal["hit", "miss"] | None  # None — к слою не обращались: он выключен


class CacheManager:
    """
    Каждый повтор начинается с кэша в том состоянии, что на старте прогона. Без
    cache.embedding.namespace ключи Redis принадлежат прогону и удаляются перед каждым
    повтором и в конце. С namespace кэш общий для прогонов, за его состояние отвечает
    исследователь, и ragglet его не чистит.
    """

    def __init__(self, cfg: ScenarioConfig, embedding_revision: str | None = None):
        self.cfg = cfg

        self.emb_cache: RedisEmbeddingCache | None = None
        self.ret_cache: LRURetrievalCache | None = None
        self._owns_emb_keys = False

        if cfg.cache.enabled and cfg.cache.embedding:
            namespace = cfg.cache.embedding.namespace
            self._owns_emb_keys = namespace is None
            self.emb_cache = RedisEmbeddingCache(
                endpoint=cfg.cache.embedding.endpoint,
                ttl_seconds=cfg.cache.embedding.ttl_seconds,
                model_name=cfg.embedding.model,
                revision=embedding_revision,
                normalize=cfg.embedding.normalize,
                namespace=namespace or f"run-{uuid.uuid4().hex}",
            )

        if cfg.cache.enabled and cfg.cache.retrieval:
            self.ret_cache = LRURetrievalCache(size=cfg.cache.retrieval.size)

    async def reset(self) -> None:
        """Перед повтором: убрать всё, что записал прогон."""
        if self.ret_cache is not None:
            self.ret_cache.clear()
        if self.emb_cache is not None and self._owns_emb_keys:
            await self.emb_cache.delete_written()

    async def close(self) -> None:
        if self.emb_cache is None:
            return
        try:
            if self._owns_emb_keys:
                await self.emb_cache.delete_written()  # при падении остаток истечёт по TTL
        finally:
            await self.emb_cache.close()

    async def get_embedding(self, query: str) -> tuple[list[float] | None, Lookup]:
        if self.emb_cache is None:
            return None, None
        vec = await self.emb_cache.get(query)
        return (vec, "hit") if vec is not None else (None, "miss")

    async def set_embedding(self, query: str, vec: list[float]) -> None:
        if self.emb_cache is not None:
            await self.emb_cache.set(query, vec)

    def _sources_sig(self) -> list[dict[str, Any]]:
        sig = []
        for s in self.cfg.retrieval.sources or []:
            sig.append({"name": s.name, "kind": s.kind, "weight": s.weight, "params": s.params})
        return sig

    def _merge_sig(self) -> dict[str, Any]:
        return {
            "method": self.cfg.retrieval.merge.method,
            "rrf_k": self.cfg.retrieval.merge.rrf_k,
            "dedup": self.cfg.retrieval.merge.deduplicate,
        }

    def _retrieval_key(self, query: str) -> str:
        # без вектора: кэш можно проверить до эмбеддинга
        return LRURetrievalCache.make_key(
            scenario_name=self.cfg.name,
            query_text=query,
            top_k=self.cfg.retrieval.top_k,
            strategy=self.cfg.retrieval.strategy,
            sources_sig=self._sources_sig(),
            merge_sig=self._merge_sig(),
        )

    def get_retrieval(self, query: str) -> tuple[list[RetrievedItem] | None, Lookup]:
        if self.ret_cache is None:
            return None, None
        entry = self.ret_cache.get(self._retrieval_key(query))
        return (entry.retrieved_items, "hit") if entry is not None else (None, "miss")

    def set_retrieval(self, query: str, retrieved_items: list[RetrievedItem]) -> None:
        if self.ret_cache is not None:
            self.ret_cache.set(self._retrieval_key(query), RetrievalCacheEntry(retrieved_items=retrieved_items))
