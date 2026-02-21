from __future__ import annotations

from pathlib import Path
import random
import time
from typing import Any
import anyio
from ragglet.retrieval.fanout_rrf import SourceSpec, RetrievedItem, rrf_merge, run_sources

import pandas as pd

from ragglet.config.scenario import ScenarioConfig
from ragglet.bench.dataset import load_jsonl_dataset
from ragglet.bench.metrics import recall_at_k, mrr_at_k
from ragglet.bench.artifacts import make_artifact_paths, save_config_snapshot
from ragglet.modules.embedders.st_embedder import SentenceTransformerEmbedder
from ragglet.stores.qdrant_store import QdrantStore

def _to_retrieved(candidate, source_name: str) -> RetrievedItem:
    payload = candidate.payload or {}
    ext = payload.get("external_id")
    # если external_id вдруг отсутствует — fallback на point id
    external_id = str(ext) if ext is not None else str(candidate.id)
    return RetrievedItem(
        point_id=str(candidate.id),
        external_id=external_id,
        score_raw=float(candidate.score_raw),
        source=source_name,
        payload=candidate.payload,
    )

def _extract_retrieved_ids(candidates) -> list[str]:
    # сравниваем именно по external_id (payload), иначе метрики бессмысленны
    ids: list[str] = []
    for c in candidates:
        ext = (c.payload or {}).get("external_id")
        ids.append(str(ext) if ext is not None else str(c.id))
    return ids


async def _run_once(cfg: ScenarioConfig, items, store: QdrantStore, embedder: SentenceTransformerEmbedder) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for it in items:
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        vec = embedder.embed_batch([it.query])[0]
        t2 = time.perf_counter()

        strategy = cfg.retrieval.strategy
        top_k = cfg.retrieval.top_k

        if strategy == "single":
            # один источник: либо первый из sources, либо дефолт "vec"
            src = None
            if cfg.retrieval.sources:
                s0 = cfg.retrieval.sources[0]
                src = SourceSpec(name=s0.name, kind=s0.kind, weight=s0.weight, params=s0.params)
            else:
                src = SourceSpec(name="vec", kind="vector", weight=1.0, params={})

            if src.kind != "vector":
                raise NotImplementedError("Only vector source is implemented in baseline")

            candidates = store.search(vec, top_k=top_k, params=src.params)
            retrieved_items = [_to_retrieved(c, src.name) for c in candidates]

        elif strategy == "fanout_merge":
            # готовим источники
            sources: list[SourceSpec] = []
            for s in cfg.retrieval.sources:
                sources.append(SourceSpec(name=s.name, kind=s.kind, weight=s.weight, params=s.params))

            # сейчас реализуем только vector источники (keyword позже)
            for s in sources:
                if s.kind != "vector":
                    raise NotImplementedError(f"Source kind '{s.kind}' is not implemented yet")

            # fanout tasks (параллельно)
            async def _query_source(s: SourceSpec):
                # можно добавить per-source timeout, пока используем общий retrieval_ms на весь fanout
                return store.search(vec, top_k=top_k, params=s.params)

            tasks = [(s.name, lambda s=s: _query_source(s)) for s in sources]

            # timeout на retrieval fanout
            with anyio.fail_after(cfg.timeouts.retrieval_ms / 1000.0):
                parallel = bool(cfg.retrieval.async_enabled)  # поле в модели с alias="async"
                res_map = await run_sources(tasks, parallel=parallel)

            per_source_ranked: dict[str, list[RetrievedItem]] = {}
            weights: dict[str, float] = {}
            for s in sources:
                weights[s.name] = s.weight
                per_source_ranked[s.name] = [_to_retrieved(c, s.name) for c in res_map.get(s.name, [])]

            merged = rrf_merge(
                per_source_ranked=per_source_ranked,
                weights=weights,
                rrf_k=cfg.retrieval.merge.rrf_k,
                top_k=top_k,
                deduplicate=cfg.retrieval.merge.deduplicate,
            )
            retrieved_items = merged

        else:
            raise NotImplementedError(f"retrieval.strategy='{strategy}' is not implemented")

        t3 = time.perf_counter()

        retrieved_ids = [it.external_id for it in retrieved_items]
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
            }
        )

    return pd.DataFrame(rows)


def _summarize(cfg: ScenarioConfig, df: pd.DataFrame) -> dict[str, Any]:
    s: dict[str, Any] = {
        "scenario": cfg.name,
        "n": int(len(df)),
        "mrr_mean": float(df["mrr"].mean()) if len(df) else 0.0,
        "lat_total_p50": float(df["lat_total_ms"].quantile(0.50)) if len(df) else 0.0,
        "lat_total_p95": float(df["lat_total_ms"].quantile(0.95)) if len(df) else 0.0,
    }
    for k in cfg.metrics.quality.recall_at_k:
        col = f"recall@{k}"
        s[f"{col}_mean"] = float(df[col].mean()) if len(df) else 0.0
    return s


def run_scenario(cfg: ScenarioConfig) -> dict[str, Any]:
    # 1) Артефакты
    paths = make_artifact_paths(cfg.run.artifacts_dir, cfg.name)
    save_config_snapshot(paths.run_dir, cfg)

    # 2) Датасет
    ds_path = Path(cfg.dataset.id)
    items = load_jsonl_dataset(ds_path, cfg.dataset.query_field, cfg.dataset.truth_field, cfg.dataset.limit)

    # 3) Компоненты (пока baseline: qdrant + embedder)
    if cfg.storage.backend.type != "qdrant":
        raise NotImplementedError("Only qdrant backend is implemented in the first iteration")

    store = QdrantStore(cfg.storage.backend.endpoint, cfg.storage.backend.collection)
    embedder = SentenceTransformerEmbedder(cfg.embedding.model, normalize=cfg.embedding.normalize)

    # 4) Repeats + shuffle
    all_summaries: list[dict[str, Any]] = []
    per_repeat_paths: list[str] = []

    base_rng = random.Random(cfg.seed)

    for r in range(cfg.run.repeats):
        run_items = list(items)

        if cfg.run.shuffle:
            # детерминированно: на каждый repeat новый seed
            seed_r = base_rng.randint(0, 2**31 - 1)
            random.Random(seed_r).shuffle(run_items)

        df = anyio.run(_run_once,cfg, run_items, store, embedder)
        summary = _summarize(cfg, df)
        summary["repeat_idx"] = r
        summary["retrieval_parallel"] = bool(cfg.retrieval.async_enabled)
        summary["retrieval_strategy"] = cfg.retrieval.strategy
        summary["sources_count"] = len(cfg.retrieval.sources) if cfg.retrieval.sources else 1
        all_summaries.append(summary)

        # 5) Сохранение per-query
        if cfg.run.save_per_query:
            if cfg.output.format in ("csv", "both"):
                p = paths.run_dir / f"per_query.repeat{r}.csv"
                df.to_csv(p, index=False)
            if cfg.output.format in ("parquet", "both"):
                p = paths.run_dir / f"per_query.repeat{r}.parquet"
                df.to_parquet(p, index=False)

        per_repeat_paths.append(str(paths.run_dir))

    # 6) Сохранение summary (всегда)
    df_sum = pd.DataFrame(all_summaries)
    (paths.run_dir / "summary.json").write_text(df_sum.to_json(orient="records", indent=2), encoding="utf-8")
    (paths.run_dir / "summary.csv").write_text(df_sum.to_csv(index=False), encoding="utf-8")

    # 7) Aggregate summary (среднее по repeat)
    # 7) Aggregate summary (mean over repeats) -> отдельные файлы
    agg: dict[str, Any] = {"scenario": cfg.name, "repeats": cfg.run.repeats}

    # какие колонки агрегировать: все числовые (кроме repeat_idx)
    numeric_cols = [c for c in df_sum.columns if
                    c not in {"scenario", "repeat_idx"} and pd.api.types.is_numeric_dtype(df_sum[c])]

    for col in numeric_cols:
        agg[f"{col}_mean_over_repeats"] = float(df_sum[col].mean())
        agg[f"{col}_std_over_repeats"] = float(df_sum[col].std(ddof=1)) if cfg.run.repeats > 1 else 0.0
        agg[f"{col}_min_over_repeats"] = float(df_sum[col].min())
        agg[f"{col}_max_over_repeats"] = float(df_sum[col].max())

    agg_path_json = paths.run_dir / "summary.aggregate.json"
    agg_path_csv = paths.run_dir / "summary.aggregate.csv"

    agg_path_json.write_text(pd.Series(agg).to_json(indent=2), encoding="utf-8")
    pd.DataFrame([agg]).to_csv(agg_path_csv, index=False)

    return {
        "summary": agg,
        "run_dir": str(paths.run_dir),
        "repeats": all_summaries,
    }