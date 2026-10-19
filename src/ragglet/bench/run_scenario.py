from __future__ import annotations

import random

import pandas as pd

from ragglet.bench.artifacts import make_artifact_paths, save_config_snapshot, save_manifest
from ragglet.bench.preflight import preflight
from ragglet.bench.query_loop import run_queries_once
from ragglet.bench.summarize import summarize_repeat, aggregate_over_repeats
from ragglet.cache.manager import CacheManager
from ragglet.config.scenario import ScenarioConfig
from ragglet.core.environment import capture_env
from ragglet.generation.engine import build_generator
from ragglet.modules.embedders.st_embedder import SentenceTransformerEmbedder
from ragglet.rerank.engine import build_reranker
from ragglet.retrieval.backends import BackendRegistry
from ragglet.storage.layout import build_storage_layout
from ragglet.stores.qdrant_store_async import AsyncQdrantStore


async def run_scenario(cfg: ScenarioConfig) -> dict:
    layout = build_storage_layout(cfg.storage)
    if any(target.type != "qdrant" for target in layout.targets):
        raise NotImplementedError("Only qdrant backend is implemented in the first iteration")

    stores = {
        target.id: AsyncQdrantStore(target.endpoint, target.collection)
        for target in layout.targets
    }
    async_store = stores[layout.primary.id]
    caches: CacheManager | None = None

    try:
        checked = await preflight(cfg, stores)
        items = checked.dataset.items

        backends = await BackendRegistry.build(cfg, async_store, stores)
        # ровно та ревизия модели, которой собран индекс
        embedder = SentenceTransformerEmbedder(
            cfg.embedding.model,
            normalize=cfg.embedding.normalize,
            revision=checked.index["embedding_revision"],
        )
        reranker = build_reranker(cfg)
        generator = build_generator(cfg)
        caches = CacheManager(cfg, embedding_revision=embedder.revision)

        # каталог прогона — только когда всё загружено, но до запросов
        paths = make_artifact_paths(cfg.run.artifacts_dir, cfg.name)
        save_config_snapshot(paths.run_dir, cfg)
        save_manifest(paths.run_dir, {**checked.manifest(), "env": capture_env(embedder.compute_info())})

        all_summaries: list[dict] = []
        base_rng = random.Random(cfg.seed)

        for r in range(cfg.run.repeats):
            run_items = list(items)
            if cfg.run.shuffle:
                seed_r = base_rng.randint(0, 2 ** 31 - 1)
                random.Random(seed_r).shuffle(run_items)

            caches.reset_stats()
            df = await run_queries_once(cfg, run_items, backends, embedder, caches, reranker, generator)

            summary = summarize_repeat(cfg, df)
            summary["repeat_idx"] = r
            summary["retrieval_parallel"] = bool(cfg.retrieval.async_enabled)
            summary["retrieval_strategy"] = cfg.retrieval.strategy
            summary["sources_count"] = len(cfg.retrieval.sources) if cfg.retrieval.sources else 1
            all_summaries.append(summary)

            if cfg.run.save_per_query:
                if cfg.output.format in ("csv", "both"):
                    df.to_csv(paths.run_dir / f"per_query.repeat{r}.csv", index=False)
                if cfg.output.format in ("parquet", "both"):
                    df.to_parquet(paths.run_dir / f"per_query.repeat{r}.parquet", index=False)

        df_sum = pd.DataFrame(all_summaries)
        (paths.run_dir / "summary.json").write_text(df_sum.to_json(orient="records", indent=2), encoding="utf-8")
        (paths.run_dir / "summary.csv").write_text(df_sum.to_csv(index=False), encoding="utf-8")

        agg = aggregate_over_repeats(df_sum, cfg.name, cfg.run.repeats)
        (paths.run_dir / "summary.aggregate.json").write_text(pd.Series(agg).to_json(indent=2), encoding="utf-8")
        pd.DataFrame([agg]).to_csv(paths.run_dir / "summary.aggregate.csv", index=False)

        return {"summary": agg, "run_dir": str(paths.run_dir), "repeats": all_summaries}

    finally:
        if caches is not None:
            await caches.close()
        for store in stores.values():
            await store.close()
