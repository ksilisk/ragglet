from __future__ import annotations

import time
from typing import Any

import pandas as pd

from ragglet.config.scenario import ScenarioConfig
from ragglet.bench.metrics import recall_at_k, mrr_at_k
from ragglet.cache.manager import CacheManager
from ragglet.retrieval.engine import retrieve_ids
from ragglet.stores.qdrant_store import QdrantStore
from ragglet.modules.embedders.st_embedder import SentenceTransformerEmbedder


async def run_queries_once(
    cfg: ScenarioConfig,
    items,
    store: QdrantStore,
    embedder: SentenceTransformerEmbedder,
    caches: CacheManager,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def embed_one(q: str) -> list[float]:
        return embedder.embed_batch([q])[0]

    for it in items:
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        vec = await caches.get_or_compute_embedding(it.query, embed_one)
        t2 = time.perf_counter()

        # retrieval cache
        key = caches.make_retrieval_key(it.query)
        cached = caches.get_retrieval(key)
        if cached is not None:
            retrieved_ids = cached.retrieved_ids
        else:
            retrieved_ids = await retrieve_ids(cfg, store, vec)
            caches.set_retrieval(key, retrieved_ids)

        t3 = time.perf_counter()

        truth = set(it.truth_ids)
        rec = {f"recall@{k}": recall_at_k(retrieved_ids, truth, k) for k in cfg.metrics.quality.recall_at_k}
        rr = mrr_at_k(retrieved_ids, truth, cfg.metrics.quality.mrr_at_k)

        rows.append(
            {
                "qid": it.qid,
                "query": it.query,
                "truth_ids": it.truth_ids,
                "retrieved_ids": retrieved_ids,
                "mrr": rr,
                **rec,
                "lat_embed_ms": (t2 - t1) * 1000.0,
                "lat_retrieve_ms": (t3 - t2) * 1000.0,
                "lat_total_ms": (t3 - t0) * 1000.0,
                "emb_hit_rate": caches.stats.emb_hit_rate,
                "ret_hit_rate": caches.stats.ret_hit_rate,
            }
        )

    return pd.DataFrame(rows)