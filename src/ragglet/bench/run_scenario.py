from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ragglet.bench.artifacts import make_artifact_paths, save_config_snapshot, save_manifest
from ragglet.bench.plan import run_plan
from ragglet.bench.preflight import Preflight, check_scenario, preflight
from ragglet.bench.query_loop import run_queries_once
from ragglet.bench.summarize import summarize_repeat, aggregate_over_repeats
from ragglet.cache.manager import CacheManager
from ragglet.config.scenario import ScenarioConfig
from ragglet.core.environment import Service, capture_env, describe_services
from ragglet.core.progress import Progress, no_progress
from ragglet.generation.engine import build_generator
from ragglet.modules.embedders.st_embedder import SentenceTransformerEmbedder
from ragglet.rerank.engine import build_reranker
from ragglet.retrieval.backends import BackendRegistry
from ragglet.stores.qdrant_store_async import AsyncQdrantStore


@dataclass(frozen=True)
class RunResult:
    preflight: Preflight
    run_dir: Path | None = None  # у dry_run прогона нет
    summary: dict[str, Any] | None = None  # агрегат по повторам
    repeats: list[dict[str, Any]] = field(default_factory=list)


async def run_scenario(
        cfg: ScenarioConfig,
        *,
        dry_run: bool = False,
        progress: Progress = no_progress,
        run_id: str | None = None,
) -> RunResult:
    """
    Прогон сценария; `dry_run` — только preflight: без моделей, каталога прогона и запросов.
    `run_id` — имя каталога прогона, его выдаёт реестр; None — `<name>_<UTC сейчас>`.
    """
    layout = check_scenario(cfg)

    stores = {
        target.id: AsyncQdrantStore(target.endpoint, target.collection)
        for target in layout.targets
    }
    async_store = stores[layout.primary.id]
    caches: CacheManager | None = None

    try:
        checked = await preflight(cfg, stores)
        if dry_run:
            return RunResult(preflight=checked)
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

        services: dict[str, Mapping[str, Service]] = {"storage": stores}
        if caches.emb_cache is not None:
            services["cache"] = {"embedding": caches.emb_cache}
        env = capture_env(embedder.compute_info())
        env["services"] = await describe_services(services)

        # каталог прогона — только когда всё загружено, но до запросов
        paths = make_artifact_paths(cfg.run.artifacts_dir, cfg.name, run_id)
        save_config_snapshot(paths.run_dir, cfg)
        save_manifest(
            paths.run_dir,
            {
                **checked.manifest(),
                "rerank": {"revision": reranker.revision} if cfg.rerank.enabled else None,
                "env": env,
            },
        )

        all_summaries: list[dict] = []
        base_rng = random.Random(cfg.seed)
        done, total = 0, run_plan(cfg, len(items)).total

        def query_done() -> None:
            nonlocal done
            done += 1
            progress("queries", done, total)

        for r in range(cfg.run.repeats):
            run_items = list(items)
            if cfg.run.shuffle:
                seed_r = base_rng.randint(0, 2 ** 31 - 1)
                random.Random(seed_r).shuffle(run_items)

            await caches.reset()
            df = await run_queries_once(
                cfg, run_items, backends, embedder, caches, reranker, generator, on_done=query_done
            )

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

        return RunResult(preflight=checked, run_dir=paths.run_dir, summary=agg, repeats=all_summaries)

    finally:
        if caches is not None:
            await caches.close()
        for store in stores.values():
            await store.close()
