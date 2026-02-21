from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragglet.config.scenario import ScenarioConfig
from ragglet.cache.embedding_redis import RedisEmbeddingCache
from ragglet.cache.retrieval_lru import LRURetrievalCache, RetrievalCacheEntry


@dataclass
class CacheStats:
    emb_hits: int = 0
    emb_misses: int = 0
    ret_hits: int = 0
    ret_misses: int = 0

    def hit_rate(self, hits: int, misses: int) -> float:
        d = hits + misses
        return float(hits) / d if d else 0.0

    @property
    def emb_hit_rate(self) -> float:
        return self.hit_rate(self.emb_hits, self.emb_misses)

    @property
    def ret_hit_rate(self) -> float:
        return self.hit_rate(self.ret_hits, self.ret_misses)


class CacheManager:
    def __init__(self, cfg: ScenarioConfig):
        self.cfg = cfg
        self.stats = CacheStats()

        self.emb_cache: RedisEmbeddingCache | None = None
        self.ret_cache: LRURetrievalCache | None = None

        if cfg.cache.enabled and cfg.cache.embedding:
            self.emb_cache = RedisEmbeddingCache(
                endpoint=cfg.cache.embedding.endpoint,
                ttl_seconds=cfg.cache.embedding.ttl_seconds,
                model_name=cfg.embedding.model,
                normalize=cfg.embedding.normalize,
            )

        if cfg.cache.enabled and cfg.cache.retrieval:
            self.ret_cache = LRURetrievalCache(size=cfg.cache.retrieval.size)

    async def close(self) -> None:
        if self.emb_cache is not None:
            await self.emb_cache.close()

    async def get_or_compute_embedding(self, query: str, embed_fn) -> list[float]:
        """
        embed_fn: sync callable(query)->vector or callable producing vector.
        """
        vec = None
        if self.emb_cache is not None:
            vec = await self.emb_cache.get(query)
            if vec is not None:
                self.stats.emb_hits += 1
            else:
                self.stats.emb_misses += 1

        if vec is None:
            vec = embed_fn(query)
            if self.emb_cache is not None:
                await self.emb_cache.set(query, vec)

        return vec

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

    def make_retrieval_key(self, query: str) -> str | None:
        if self.ret_cache is None:
            return None
        return self.ret_cache.make_key(
            scenario_name=self.cfg.name,
            query_text=query,
            top_k=self.cfg.retrieval.top_k,
            strategy=self.cfg.retrieval.strategy,
            sources_sig=self._sources_sig(),
            merge_sig=self._merge_sig(),
        )

    def get_retrieval(self, key: str | None) -> RetrievalCacheEntry | None:
        if self.ret_cache is None or key is None:
            return None
        entry = self.ret_cache.get(key)
        if entry is not None:
            self.stats.ret_hits += 1
        else:
            self.stats.ret_misses += 1
        return entry

    def set_retrieval(self, key: str | None, retrieved_ids: list[str]) -> None:
        if self.ret_cache is None or key is None:
            return
        self.ret_cache.set(key, RetrievalCacheEntry(retrieved_ids=retrieved_ids))

    def reset_stats(self) -> None:
        self.stats = CacheStats()