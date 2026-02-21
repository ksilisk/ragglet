from __future__ import annotations

from typing import Any
import pandas as pd
from ragglet.config.scenario import ScenarioConfig


def summarize_repeat(cfg: ScenarioConfig, df: pd.DataFrame) -> dict[str, Any]:
    n = int(len(df))

    # timeout aggregates
    if n and "timed_out" in df.columns:
        timed_out_count = int(df["timed_out"].fillna(False).sum())
        timeout_rate = float(timed_out_count) / n
    else:
        timed_out_count = 0
        timeout_rate = 0.0

    stage_rates: dict[str, float] = {}
    stage_counts: dict[str, int] = {}
    if n and timed_out_count and "timeout_stage" in df.columns:
        # считаем только для таймаутнутых
        stages = df.loc[df["timed_out"] == True, "timeout_stage"].fillna("unknown")
        counts = stages.value_counts()
        stage_counts = {str(k): int(v) for k, v in counts.items()}
        stage_rates = {k: v / n for k, v in stage_counts.items()}  # rate по всем запросам

    s: dict[str, Any] = {
        "scenario": cfg.name,
        "n": int(len(df)),
        "mrr_mean": float(df["mrr"].mean()) if len(df) else 0.0,
        "lat_total_p50": float(df["lat_total_ms"].quantile(0.50)) if len(df) else 0.0,
        "lat_total_p95": float(df["lat_total_ms"].quantile(0.95)) if len(df) else 0.0,
        "emb_hit_rate": float(df["emb_hit_rate"].mean()) if len(df) else 0.0,
        "ret_hit_rate": float(df["ret_hit_rate"].mean()) if len(df) else 0.0,
        "timeout_count": timed_out_count,
        "timeout_rate": timeout_rate,
        "timeout_stage_counts": stage_counts,
        "timeout_stage_rates": stage_rates,
    }
    for k in cfg.metrics.quality.recall_at_k:
        col = f"recall@{k}"
        s[f"{col}_mean"] = float(df[col].mean()) if len(df) else 0.0
    return s


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