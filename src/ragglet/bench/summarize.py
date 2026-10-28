from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ragglet.bench.evaluation import quality_columns
from ragglet.config.scenario import ScenarioConfig

_LATENCY_COLUMNS = {
    "embed": "lat_embed_ms",
    "retrieve": "lat_retrieve_ms",
    "rerank": "lat_rerank_ms",
    "context": "lat_context_ms",
    "generation": "lat_generation_ms",
    "total": "lat_total_ms",
    "merge": "lat_merge_ms",
}

OUTCOMES = ("ok", "timeout", "error")

CACHE_LAYERS = {"emb": "embedding", "ret": "retrieval"}  # префикс колонки -> слой в cfg.cache


def _latency_percentiles(cfg: ScenarioConfig, df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for step in cfg.metrics.latency_steps:
        col = _LATENCY_COLUMNS.get(step)
        if col is None or col not in df.columns:
            continue

        for percentile in cfg.metrics.percentiles:
            # пустые значения (этап не завершился или выключен) quantile пропускает
            out[f"lat_{step}_p{percentile}"] = float(df[col].quantile(percentile / 100.0))
    return out


def _outcomes(df: pd.DataFrame) -> dict[str, float]:
    unknown = set(df["outcome"]) - set(OUTCOMES)
    if unknown:
        raise ValueError(f"unknown query outcomes {sorted(unknown)}; known: {list(OUTCOMES)}")
    counts = df["outcome"].value_counts()
    out: dict[str, float] = {}
    for outcome in OUTCOMES:
        out[f"{outcome}_count"] = int(counts.get(outcome, 0))
        out[f"{outcome}_rate"] = out[f"{outcome}_count"] / len(df)
    return out


def _cache_summary(cfg: ScenarioConfig, df: pd.DataFrame) -> dict[str, float]:
    """Только включённые слои: у выключенного полей нет, а не нули."""
    out: dict[str, float] = {}
    if not cfg.cache.enabled:
        return out
    for layer, field in CACHE_LAYERS.items():
        if getattr(cfg.cache, field) is None:
            continue
        lookup = df[f"{layer}_cache"]
        hits, misses = int((lookup == "hit").sum()), int((lookup == "miss").sum())
        out[f"{layer}_hits"] = hits
        out[f"{layer}_misses"] = misses
        out[f"{layer}_hit_rate"] = hits / (hits + misses) if hits + misses else math.nan
        for state in ("hit", "miss"):
            lat = df.loc[lookup == state, "lat_total_ms"]
            for percentile in cfg.metrics.percentiles:
                out[f"lat_total_{layer}_{state}_p{percentile}"] = float(lat.quantile(percentile / 100.0))
    return out


def summarize_repeat(cfg: ScenarioConfig, df: pd.DataFrame) -> dict[str, Any]:
    """
    Итоги одного повтора, только из строк per_query. Правила (docs/metrics.md):
    - wall_ms — от начала повтора до конца последнего запроса, t_start_ms + lat_total_ms;
    - throughput_qps — только ответы (outcome=ok) в секунду: таймаут и ошибка пользователю
      ничего не дали;
    - lat_total — по всем запросам, задержка этапа — по строкам, где этап завершился;
    - запрос без ответа — промах: качество 0 ставит query_loop, в среднее он входит;
    - доля попаданий кэша — из счётчиков отметок hit/miss, не среднее долей.
    """
    n = int(len(df))
    outcomes = _outcomes(df)
    wall_ms = float((df["t_start_ms"] + df["lat_total_ms"]).max())

    stage_counts: dict[str, int] = {}
    if outcomes["timeout_count"]:
        stages = df.loc[df["outcome"] == "timeout", "timeout_stage"].fillna("unknown")
        stage_counts = {str(k): int(v) for k, v in stages.value_counts().items()}
    stage_rates = {k: v / n for k, v in stage_counts.items()}  # доля от всех запросов

    s: dict[str, Any] = {
        "scenario": cfg.name,
        "n": n,
        **outcomes,
        "wall_ms": wall_ms,
        "throughput_qps": outcomes["ok_count"] / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
        **_latency_percentiles(cfg, df),
        **_cache_summary(cfg, df),
        "context_tokens_mean": float(df["context_tokens"].mean()) if "context_tokens" in df.columns and len(df) else 0.0,
        "context_chunks_mean": float(df["context_chunks"].mean()) if "context_chunks" in df.columns and len(df) else 0.0,
        "gen_input_tokens_mean": float(df["gen_input_tokens"].mean()) if "gen_input_tokens" in df.columns and len(df) else 0.0,
        "gen_output_tokens_mean": float(df["gen_output_tokens"].mean()) if "gen_output_tokens" in df.columns and len(df) else 0.0,
        "gen_total_tokens_mean": float(df["gen_total_tokens"].mean()) if "gen_total_tokens" in df.columns and len(df) else 0.0,
        "timeout_stage_counts": stage_counts,
        "timeout_stage_rates": stage_rates,
    }
    s.update(_quality_summary(cfg, df))
    return s


def _quality_summary(cfg: ScenarioConfig, df: pd.DataFrame) -> dict[str, float]:
    quality = cfg.metrics.quality
    out: dict[str, float] = {}
    for col in [*quality_columns(quality), "n_chunks", "n_docs", "dup_chunks"]:
        out[f"{col}_mean"] = float(df[col].mean()) if len(df) else 0.0
    if quality.doc:
        # доля запросов, где doc-level @k посчитан по списку короче k документов
        for k in quality.k:
            out[f"doc_short@{k}_rate"] = float((df["n_docs"] < k).mean()) if len(df) else 0.0
    return out


def aggregate_over_repeats(df_sum: pd.DataFrame, scenario: str, repeats: int) -> dict[str, Any]:
    agg: dict[str, Any] = {"scenario": scenario, "repeats": repeats}
    numeric_cols = [
        c for c in df_sum.columns
        if c not in {"scenario", "repeat_idx"} and pd.api.types.is_numeric_dtype(df_sum[c])
    ]
    for col in numeric_cols:
        agg[f"{col}_mean_over_repeats"] = float(df_sum[col].mean())
        agg[f"{col}_std_over_repeats"] = float(df_sum[col].std(ddof=1)) if repeats > 1 else 0.0
        agg[f"{col}_min_over_repeats"] = float(df_sum[col].min())
        agg[f"{col}_max_over_repeats"] = float(df_sum[col].max())
    return agg
