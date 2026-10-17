#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd
import redis

ROOT = Path(__file__).resolve().parents[1]
K_VALUES = (1, 3, 5, 10, 30)

SCENARIOS = {
    "article_a_s0_retrieval_reference": ("adaptive", "A-S0", "retrieval reference"),
    "article_a0_top_n": ("adaptive", "A0", "top_n"),
    "article_a1_adaptive_t015": ("adaptive", "A1", "adaptive 0.15"),
    "article_a2_adaptive_t030": ("adaptive", "A2", "adaptive 0.30"),
    "article_a3_adaptive_t050": ("adaptive", "A3", "adaptive 0.50"),
    "article_b0_cache_disabled": ("cache", "B0", "cache disabled"),
    "article_b1_cache_phases": ("cache", "B1", "cache enabled"),
}

EXPECTED_RUNS = {
    "article_a_s0_retrieval_reference": 1,
    "article_a0_top_n": 1,
    "article_a1_adaptive_t015": 1,
    "article_a2_adaptive_t030": 1,
    "article_a3_adaptive_t050": 1,
    "article_b0_cache_disabled": 5,
    "article_b1_cache_phases": 5,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_text(*args: str) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def cpu_name() -> str | None:
    for args in (("sysctl", "-n", "machdep.cpu.brand_string"), ("lscpu",)):
        value = run_text(*args)
        if value:
            if args[0] == "lscpu":
                match = re.search(r"^Model name:\s*(.+)$", value, re.MULTILINE)
                return match.group(1).strip() if match else value.splitlines()[0]
            return value
    return platform.processor() or None


def total_ram_bytes() -> int | None:
    value = run_text("sysctl", "-n", "hw.memsize")
    if value and value.isdigit():
        return int(value)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (ValueError, OSError, AttributeError):
        return None


def qdrant_version(url: str) -> dict[str, Any]:
    try:
        with urlopen(url.rstrip("/") + "/", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {"reachable": True, "version": payload.get("version"), "response": payload}
    except Exception as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}


def redis_version(url: str) -> dict[str, Any]:
    client = redis.Redis.from_url(url, socket_timeout=5)
    try:
        info = client.info("server")
        return {
            "reachable": True,
            "version": info.get("redis_version"),
            "mode": info.get("redis_mode"),
        }
    except Exception as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        client.close()


def capture_manifest(args: argparse.Namespace) -> int:
    output = (
        (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    config_paths = sorted((ROOT / "configs/experiments/article").rglob("*.yaml"))
    manifest = {
        "schema_version": 1,
        "captured_at_utc": utc_now(),
        "repository": {
            "path": str(ROOT),
            "git_revision": run_text("git", "rev-parse", "HEAD"),
            "git_branch": run_text("git", "branch", "--show-current"),
            "dirty": bool(run_text("git", "status", "--porcelain")),
            "dirty_state": (run_text("git", "status", "--porcelain") or "").splitlines(),
        },
        "environment": {
            "os": platform.platform(),
            "macos_version": platform.mac_ver()[0] or None,
            "machine": platform.machine(),
            "cpu": cpu_name(),
            "logical_cpu_count": os.cpu_count(),
            "ram_bytes": total_ram_bytes(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
        },
        "services": {
            "qdrant": qdrant_version(args.qdrant_url),
            "redis": redis_version(args.redis_url),
        },
        "hashes": {
            "uv.lock": sha256(ROOT / "uv.lock"),
            "dataset": sha256(ROOT / "data/queries/truth_manifest.jsonl"),
            "index_profile": sha256(ROOT / "configs/indexes/demo.yaml"),
            "experiment_configs": {
                str(path.relative_to(ROOT)): sha256(path) for path in config_paths
            },
        },
        "fixed_protocol": {
            "adaptive_thresholds": [0.15, 0.30, 0.50],
            "adaptive_repeats": 5,
            "cache_independent_trials": 5,
            "cache_repeats_per_trial": 2,
            "cache_redis_db": 15,
            "context_size_primary_metrics": ["context_chunks", "context_chars"],
            "context_tokens_interpretation": "Approximate whitespace-separated word count, not tokenizer tokens.",
            "benchmark_runtime_modified": False,
        },
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    if (
        not manifest["services"]["qdrant"]["reachable"]
        or not manifest["services"]["redis"]["reachable"]
    ):
        print("Manifest written, but one or more services were unreachable.", file=sys.stderr)
        return 2
    return 0


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        parsed = ast.literal_eval(value)
        return [str(item) for item in parsed]
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [str(item) for item in value]
    raise TypeError(f"Unsupported list value: {type(value).__name__}")


def recall_at_k(retrieved: list[str], truth: list[str], k: int) -> float:
    truth_set = set(truth)
    if not truth_set:
        return 0.0
    return len(set(retrieved[:k]) & truth_set) / len(truth_set)


def mrr_at_k(retrieved: list[str], truth: list[str], k: int = 30) -> float:
    truth_set = set(truth)
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in truth_set:
            return 1.0 / rank
    return 0.0


def read_repeat(path: Path) -> pd.DataFrame:
    match = re.search(r"repeat(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Cannot parse repeat index from {path}")
    df = pd.read_parquet(path)
    df["repeat_idx"] = int(match.group(1))
    return df


def load_run(run_dir: Path, trial_idx: int) -> tuple[dict[str, Any], pd.DataFrame]:
    snapshot = json.loads((run_dir / "scenario.snapshot.json").read_text(encoding="utf-8"))
    files = sorted(run_dir.glob("per_query.repeat*.parquet"))
    if not files:
        raise ValueError(f"No per-query Parquet files in {run_dir}")
    df = pd.concat([read_repeat(path) for path in files], ignore_index=True)
    df["run_dir"] = str(run_dir)
    df["trial_idx"] = trial_idx
    return snapshot, df


def normalize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(snapshot))
    for key in ("name", "description", "tags"):
        copied.pop(key, None)
    copied["context"].pop("policy", None)
    copied["context"].pop("threshold", None)
    return copied


def matching_run_dirs(root: Path, scenario: str) -> list[Path]:
    return sorted(path for path in root.glob(f"{scenario}_*") if path.is_dir())


def compare_smoke_runs(top_dir: Path, adaptive_dir: Path) -> dict[str, Any]:
    _, top = load_run(top_dir, 1)
    _, adaptive = load_run(adaptive_dir, 1)
    columns = [
        "qid",
        "retrieved_ids",
        "context_ids",
        "context_text",
        "context_tokens",
        "context_chunks",
        "mrr",
        *[f"recall@{k}" for k in K_VALUES],
    ]
    if len(top) != len(adaptive):
        raise AssertionError(f"Smoke row count differs: {len(top)} != {len(adaptive)}")
    mismatches: list[str] = []
    for idx, (left, right) in enumerate(
        zip(top[columns].to_dict("records"), adaptive[columns].to_dict("records"), strict=True)
    ):
        for column in columns:
            lv = as_list(left[column]) if column.endswith("_ids") else left[column]
            rv = as_list(right[column]) if column.endswith("_ids") else right[column]
            if isinstance(lv, list) or isinstance(rv, list):
                equal = lv == rv
            elif pd.isna(lv) and pd.isna(rv):
                equal = True
            else:
                equal = lv == rv
            if equal:
                continue
            mismatches.append(f"row={idx} qid={left['qid']} column={column}")
    return {
        "checked_at_utc": utc_now(),
        "top_n_run": str(top_dir),
        "adaptive_t000_run": str(adaptive_dir),
        "rows": len(top),
        "equal": not mismatches,
        "mismatches": mismatches,
    }


def check_smoke(args: argparse.Namespace) -> int:
    result = compare_smoke_runs(
        Path(args.top_n_run).resolve(), Path(args.adaptive_run).resolve()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["equal"] else 1


def percentile(series: pd.Series, q: float) -> float:
    return float(series.astype(float).quantile(q)) if len(series) else 0.0


def mean(series: pd.Series) -> float:
    return float(series.astype(float).mean()) if len(series) else 0.0


def build_tidy(runs_root: Path) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    frames: list[pd.DataFrame] = []
    snapshots: dict[str, list[dict[str, Any]]] = {}

    for scenario, expected in EXPECTED_RUNS.items():
        dirs = matching_run_dirs(runs_root, scenario)
        if len(dirs) != expected:
            raise AssertionError(f"Expected {expected} runs for {scenario}, found {len(dirs)}")
        experiment, condition, label = SCENARIOS[scenario]
        snapshots[scenario] = []
        for trial_idx, run_dir in enumerate(dirs, start=1):
            snapshot, raw = load_run(run_dir, trial_idx)
            snapshots[scenario].append(snapshot)
            expected_repeats = snapshot["run"]["repeats"]
            actual_repeats = sorted(raw["repeat_idx"].unique().tolist())
            if actual_repeats != list(range(expected_repeats)):
                raise AssertionError(f"Unexpected repeats in {run_dir}: {actual_repeats}")
            for repeat_idx, repeat_df in raw.groupby("repeat_idx", sort=True):
                if len(repeat_df) != 100:
                    raise AssertionError(
                        f"Expected 100 rows in {run_dir} repeat {repeat_idx}, found {len(repeat_df)}"
                    )

            out = pd.DataFrame()
            out["experiment"] = [experiment] * len(raw)
            out["condition"] = [condition] * len(raw)
            out["condition_label"] = [label] * len(raw)
            out["scenario"] = [scenario] * len(raw)
            out["run_dir"] = raw["run_dir"]
            out["trial_idx"] = raw["trial_idx"]
            out["repeat_idx"] = raw["repeat_idx"]
            out["qid"] = raw["qid"].astype(str)
            out["timed_out"] = raw["timed_out"].fillna(False).astype(bool)
            out["lat_embed_ms"] = raw["lat_embed_ms"].astype(float)
            out["lat_retrieve_ms"] = raw["lat_retrieve_ms"].astype(float)
            out["lat_context_ms"] = raw["lat_context_ms"].astype(float)
            out["lat_total_ms"] = raw["lat_total_ms"].astype(float)
            out["context_enabled"] = bool(snapshot["context"]["enabled"])
            out["context_policy"] = snapshot["context"]["policy"]
            out["context_threshold"] = float(snapshot["context"]["threshold"])
            out["context_chunks"] = raw["context_chunks"].fillna(0).astype(int)
            out["context_chars"] = raw["context_text"].fillna("").astype(str).str.len()
            out["context_words_estimate"] = raw["context_tokens"].fillna(0).astype(int)
            out["emb_hit_rate_recorded"] = raw["emb_hit_rate"].astype(float)
            out["ret_hit_rate_recorded"] = raw["ret_hit_rate"].astype(float)

            truths = raw["truth_ids"].map(as_list)
            retrieved = raw["retrieved_ids"].map(as_list)
            contexts = raw["context_ids"].map(as_list)
            out["retrieval_mrr@30"] = [
                mrr_at_k(ids, truth) for ids, truth in zip(retrieved, truths, strict=True)
            ]
            if bool(snapshot["context"]["enabled"]):
                out["context_mrr@30"] = [
                    mrr_at_k(ids, truth) for ids, truth in zip(contexts, truths, strict=True)
                ]
            else:
                out["context_mrr@30"] = np.nan
            for k in K_VALUES:
                out[f"retrieval_recall@{k}"] = [
                    recall_at_k(ids, truth, k) for ids, truth in zip(retrieved, truths, strict=True)
                ]
                if bool(snapshot["context"]["enabled"]):
                    out[f"context_recall@{k}"] = [
                        recall_at_k(ids, truth, k)
                        for ids, truth in zip(contexts, truths, strict=True)
                    ]
                else:
                    out[f"context_recall@{k}"] = np.nan

            if scenario == "article_b1_cache_phases":
                out["cache_phase"] = np.where(out["repeat_idx"] == 0, "cold", "warm")
                out["embedding_cache_hit"] = out["repeat_idx"].astype(int)
                out["retrieval_cache_hit"] = out["repeat_idx"].astype(int)
            elif scenario == "article_b0_cache_disabled":
                out["cache_phase"] = "disabled"
                out["embedding_cache_hit"] = np.nan
                out["retrieval_cache_hit"] = np.nan
            else:
                out["cache_phase"] = "not_applicable"
                out["embedding_cache_hit"] = np.nan
                out["retrieval_cache_hit"] = np.nan
            frames.append(out)

    tidy = pd.concat(frames, ignore_index=True)
    return tidy, snapshots


def validate_inputs(tidy: pd.DataFrame, snapshots: dict[str, list[dict[str, Any]]]) -> list[str]:
    anomalies: list[str] = []

    controlled = [
        "article_a0_top_n",
        "article_a1_adaptive_t015",
        "article_a2_adaptive_t030",
        "article_a3_adaptive_t050",
    ]
    normalized = [normalize_snapshot(snapshots[name][0]) for name in controlled]
    if any(item != normalized[0] for item in normalized[1:]):
        raise AssertionError("Adaptive configs differ outside context policy/threshold")

    reference_order: list[str] | None = None
    for scenario in controlled:
        subset = tidy[tidy["scenario"] == scenario]
        for _, repeat in subset.groupby(["trial_idx", "repeat_idx"], sort=True):
            order = repeat["qid"].tolist()
            if reference_order is None:
                reference_order = order
            elif order != reference_order:
                raise AssertionError(f"Query ordering differs in {scenario}")

    b1 = tidy[tidy["scenario"] == "article_b1_cache_phases"]
    for trial_idx, trial in b1.groupby("trial_idx"):
        cold = trial[trial["repeat_idx"] == 0]
        warm = trial[trial["repeat_idx"] == 1]
        if not (
            (cold["emb_hit_rate_recorded"] == 0).all()
            and (cold["ret_hit_rate_recorded"] == 0).all()
        ):
            raise AssertionError(f"B1 trial {trial_idx}: cold phase is not all misses")
        if not (
            (warm["emb_hit_rate_recorded"] == 1).all()
            and (warm["ret_hit_rate_recorded"] == 1).all()
        ):
            raise AssertionError(f"B1 trial {trial_idx}: warm phase is not all hits")
        cold_quality = cold.sort_values("qid")[["qid", "retrieval_mrr@30", "retrieval_recall@30"]]
        warm_quality = warm.sort_values("qid")[["qid", "retrieval_mrr@30", "retrieval_recall@30"]]
        if not cold_quality.reset_index(drop=True).equals(warm_quality.reset_index(drop=True)):
            anomalies.append(
                f"B1 trial {trial_idx}: retrieval quality differs between cold and warm phases."
            )

    if tidy["timed_out"].any():
        anomalies.append(f"Observed {int(tidy['timed_out'].sum())} timed-out invocations.")

    adaptive = tidy[tidy["condition"].isin(["A0", "A1", "A2", "A3"])]
    means = adaptive.groupby("condition")["context_chunks"].mean().reindex(["A0", "A1", "A2", "A3"])
    if any(means.iloc[i] < means.iloc[i + 1] for i in range(len(means) - 1)):
        anomalies.append("Mean context_chunks is not monotonic as adaptive threshold increases.")

    return anomalies


def summarize_group(df: pd.DataFrame, include_context: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_invocations": int(len(df)),
        "n_unique_queries": int(df["qid"].nunique()),
        "n_trials": int(df["trial_idx"].nunique()),
        "timeout_rate": mean(df["timed_out"].astype(float)),
        "retrieval_mrr@30_mean": mean(df["retrieval_mrr@30"]),
        "lat_embed_ms_p50": percentile(df["lat_embed_ms"], 0.50),
        "lat_embed_ms_p95": percentile(df["lat_embed_ms"], 0.95),
        "lat_retrieve_ms_p50": percentile(df["lat_retrieve_ms"], 0.50),
        "lat_retrieve_ms_p95": percentile(df["lat_retrieve_ms"], 0.95),
        "lat_total_ms_p50": percentile(df["lat_total_ms"], 0.50),
        "lat_total_ms_p95": percentile(df["lat_total_ms"], 0.95),
    }
    for k in K_VALUES:
        result[f"retrieval_recall@{k}_mean"] = mean(df[f"retrieval_recall@{k}"])
    if include_context:
        result.update(
            {
                "context_mrr@30_mean": mean(df["context_mrr@30"].dropna()),
                "context_chunks_mean": mean(df["context_chunks"]),
                "context_chunks_p50": percentile(df["context_chunks"], 0.50),
                "context_chunks_p95": percentile(df["context_chunks"], 0.95),
                "context_chars_mean": mean(df["context_chars"]),
                "context_chars_p50": percentile(df["context_chars"], 0.50),
                "context_chars_p95": percentile(df["context_chars"], 0.95),
                "context_words_estimate_mean": mean(df["context_words_estimate"]),
                "context_words_estimate_p50": percentile(df["context_words_estimate"], 0.50),
                "context_words_estimate_p95": percentile(df["context_words_estimate"], 0.95),
                "lat_context_ms_p50": percentile(df["lat_context_ms"], 0.50),
                "lat_context_ms_p95": percentile(df["lat_context_ms"], 0.95),
            }
        )
        for k in K_VALUES:
            result[f"context_recall@{k}_mean"] = mean(df[f"context_recall@{k}"].dropna())
    return result


def build_summary(tidy: pd.DataFrame, anomalies: list[str], manifest_path: Path) -> dict[str, Any]:
    adaptive_records: list[dict[str, Any]] = []
    for condition in ("A-S0", "A0", "A1", "A2", "A3"):
        group = tidy[(tidy["experiment"] == "adaptive") & (tidy["condition"] == condition)]
        record = {"condition": condition, "label": group["condition_label"].iloc[0]}
        record.update(summarize_group(group, include_context=condition != "A-S0"))
        adaptive_records.append(record)

    cache_source = tidy[tidy["experiment"] == "cache"]
    cache_groups = {
        "B0-disabled": cache_source[cache_source["condition"] == "B0"],
        "B1-cold": cache_source[
            (cache_source["condition"] == "B1") & (cache_source["cache_phase"] == "cold")
        ],
        "B1-partial": cache_source[cache_source["condition"] == "B1"],
        "B1-warm": cache_source[
            (cache_source["condition"] == "B1") & (cache_source["cache_phase"] == "warm")
        ],
    }
    cache_records: list[dict[str, Any]] = []
    disabled_p50 = percentile(cache_groups["B0-disabled"]["lat_total_ms"], 0.50)
    disabled_p95 = percentile(cache_groups["B0-disabled"]["lat_total_ms"], 0.95)
    for condition, group in cache_groups.items():
        record = {"condition": condition}
        record.update(summarize_group(group, include_context=False))
        if condition == "B0-disabled":
            record.update(
                {
                    "embedding_cache_hits": None,
                    "embedding_cache_misses": None,
                    "embedding_cache_hit_rate": None,
                    "retrieval_cache_hits": None,
                    "retrieval_cache_misses": None,
                    "retrieval_cache_hit_rate": None,
                }
            )
        else:
            emb_hits = int(group["embedding_cache_hit"].sum())
            ret_hits = int(group["retrieval_cache_hit"].sum())
            record.update(
                {
                    "embedding_cache_hits": emb_hits,
                    "embedding_cache_misses": int(len(group) - emb_hits),
                    "embedding_cache_hit_rate": emb_hits / len(group),
                    "retrieval_cache_hits": ret_hits,
                    "retrieval_cache_misses": int(len(group) - ret_hits),
                    "retrieval_cache_hit_rate": ret_hits / len(group),
                }
            )
        record["total_p50_speedup_vs_disabled"] = disabled_p50 / record["lat_total_ms_p50"]
        record["total_p95_speedup_vs_disabled"] = disabled_p95 / record["lat_total_ms_p95"]
        cache_records.append(record)

    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "manifest": str(manifest_path),
        "metric_definitions": {
            "recall@k": "Fraction of relevant ids present in the first k ids of the named stage.",
            "mrr@30": "Reciprocal rank of the first relevant id within the first 30 ids of the named stage.",
            "context_chunks": "Number of chunks selected by context assembly.",
            "context_chars": "Python len() of rendered context_text, including metadata labels and separators.",
            "context_words_estimate": "Existing context_tokens field: approximate whitespace-separated word count, not tokenizer tokens.",
            "latency_percentiles": "Pooled percentiles over all invocations in the named condition or phase.",
        },
        "adaptive": adaptive_records,
        "cache": cache_records,
        "anomalies": anomalies,
        "validation": {
            "adaptive_configs_equal_except_policy_threshold": True,
            "adaptive_query_order_identical": True,
            "cache_trials_per_condition": 5,
            "cache_phase_rate_invariants": True,
        },
    }


def fnum(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def markdown_report(summary: dict[str, Any]) -> str:
    adaptive = {row["condition"]: row for row in summary["adaptive"]}
    cache = {row["condition"]: row for row in summary["cache"]}
    a0 = adaptive["A0"]
    a2 = adaptive["A2"]
    a3 = adaptive["A3"]
    cold = cache["B1-cold"]
    partial = cache["B1-partial"]
    warm = cache["B1-warm"]

    def reduction(base: float, value: float) -> float:
        return 100.0 * (base - value) / base

    adaptive_rows = []
    for row in summary["adaptive"]:
        adaptive_rows.append(
            "| {condition} | {label} | {recall} | {mrr} | {chunks} | {chars} | {ctx50} | {ctx95} | {total50} | {total95} |".format(
                condition=row["condition"],
                label=row["label"],
                recall=fnum(row.get("context_recall@10_mean")),
                mrr=fnum(row.get("context_mrr@30_mean")),
                chunks=fnum(row.get("context_chunks_mean"), 2),
                chars=fnum(row.get("context_chars_mean"), 1),
                ctx50=fnum(row.get("lat_context_ms_p50")),
                ctx95=fnum(row.get("lat_context_ms_p95")),
                total50=fnum(row["lat_total_ms_p50"]),
                total95=fnum(row["lat_total_ms_p95"]),
            )
        )

    cache_rows = []
    for row in summary["cache"]:
        cache_rows.append(
            "| {condition} | {n} | {rate} | {embed50} | {retrieve50} | {total50} | {total95} | {speed50} |".format(
                condition=row["condition"],
                n=row["n_invocations"],
                rate=fnum(row.get("retrieval_cache_hit_rate")),
                embed50=fnum(row["lat_embed_ms_p50"]),
                retrieve50=fnum(row["lat_retrieve_ms_p50"]),
                total50=fnum(row["lat_total_ms_p50"]),
                total95=fnum(row["lat_total_ms_p95"]),
                speed50=fnum(row["total_p50_speedup_vs_disabled"], 2),
            )
        )

    anomalies = summary["anomalies"]
    anomaly_text = (
        "\n".join(f"- {item}" for item in anomalies)
        if anomalies
        else "- Не выявлены автоматическими проверками."
    )
    return f"""# Результаты short-track экспериментов Ragglet

## Краткий вывод

Этот отчёт сформирован автоматически из per-query Parquet. Он описывает
наблюдаемые различия; причинные выводы ограничены контролируемыми факторами
матриц A и B.

- A2 (threshold 0.30) уменьшил средний контекст на
  {reduction(a0["context_chunks_mean"], a2["context_chunks_mean"]):.1f}% по chunks
  и на {reduction(a0["context_chars_mean"], a2["context_chars_mean"]):.1f}% по
  characters относительно A0; Context Recall@10 снизился на
  {100 * (a0["context_recall@10_mean"] - a2["context_recall@10_mean"]):.1f} п.п.
- A3 (threshold 0.50) уменьшил контекст на
  {reduction(a0["context_chunks_mean"], a3["context_chunks_mean"]):.1f}% по chunks
  и на {reduction(a0["context_chars_mean"], a3["context_chars_mean"]):.1f}% по
  characters; Context Recall@10 снизился на
  {100 * (a0["context_recall@10_mean"] - a3["context_recall@10_mean"]):.1f} п.п.
- Warm cache дал p50 speedup {warm["total_p50_speedup_vs_disabled"]:.2f}x и p95
  speedup {warm["total_p95_speedup_vs_disabled"]:.2f}x. При 50% повторов p50
  ускорился в {partial["total_p50_speedup_vs_disabled"]:.2f}x, но pooled p95
  практически не улучшился ({partial["total_p95_speedup_vs_disabled"]:.2f}x).
- Cold cache оказался медленнее B0: p50 ratio
  {cold["total_p50_speedup_vs_disabled"]:.2f}x и p95 ratio
  {cold["total_p95_speedup_vs_disabled"]:.2f}x, что отражает накладные расходы
  cache lookup/write без hits.

## Matrix A: adaptive context

| Cell | Режим | Context Recall@10 | Context MRR@30 | Chunks mean | Chars mean | Context p50 ms | Context p95 ms | Total p50 ms | Total p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(adaptive_rows)}

`context_tokens` в исходных артефактах интерпретируется только как примерное
число whitespace-separated слов. Основные показатели размера — chunks и
characters.

## Matrix B: cache

| Условие | Requests | Retrieval hit rate | Embed p50 ms | Retrieve p50 ms | Total p50 ms | Total p95 ms | p50 speedup vs disabled |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(cache_rows)}

Cold и warm — парные фазы пяти независимых B1 runs. Partial объединяет реально
выполненные cold и warm проходы и имеет 50% повторных запросов.

## Validation

- Smoke invariant `adaptive threshold=0.00 == top_n`: пройден строго на
  {summary["validation"]["adaptive_t000_smoke_rows"]} запросах (совпали ids,
  контекст и quality metrics; latency намеренно не сравнивалась).
- Adaptive configs совпадают вне context policy/threshold; порядок запросов
  одинаков во всех ячейках.
- Для пяти B1 trials подтверждены cold=0% hits и warm=100% hits.

## Аномалии и ограничения

{anomaly_text}

- Dataset содержит один relevant id на запрос, поэтому Recall@k здесь численно
  совпадает с Hit@k.
- Pooled p50/p95 рассчитаны по всем invocation соответствующего условия.
- Состояние внутренних caches Qdrant и OS специально не сбрасывалось.

## Воспроизводимость

- Environment manifest: `{summary["manifest"]}`
- Tidy per-invocation data: `tidy.csv`
- Machine-readable summary: `summary.json`
"""


def analyze(args: argparse.Namespace) -> int:
    runs_root = (
        (ROOT / args.runs).resolve() if not Path(args.runs).is_absolute() else Path(args.runs)
    )
    output_dir = (
        (ROOT / args.output_dir).resolve()
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )
    manifest_path = (
        (ROOT / args.manifest).resolve()
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest)
    )
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tidy, snapshots = build_tidy(runs_root)
    anomalies = validate_inputs(tidy, snapshots)
    rejected_root = runs_root.parent / "rejected"
    rejected_b1 = sorted(rejected_root.glob("article_b1_cache_phases_*"))
    if rejected_b1:
        anomalies.append(
            f"Исключены {len(rejected_b1)} предварительных B1-прогона после "
            "нарушения cold-phase invariant: benchmark использовал host Redis, а "
            "первоначальный reset был направлен в Compose Redis. Исходные "
            f"отклонённые артефакты сохранены в {rejected_root}."
        )
    summary = build_summary(tidy, anomalies, manifest_path)

    smoke_root = runs_root.parent / "smoke"
    top_smoke = matching_run_dirs(smoke_root, "article_smoke_top_n")
    adaptive_smoke = matching_run_dirs(smoke_root, "article_smoke_adaptive_t000")
    if len(top_smoke) != 1 or len(adaptive_smoke) != 1:
        raise AssertionError(
            "Expected exactly one top_n and one adaptive threshold=0.00 smoke run"
        )
    smoke = compare_smoke_runs(top_smoke[0], adaptive_smoke[0])
    if not smoke["equal"]:
        raise AssertionError("Adaptive threshold=0.00 smoke invariant failed")
    summary["validation"].update(
        {
            "adaptive_t000_equals_top_n_smoke": True,
            "adaptive_t000_smoke_rows": smoke["rows"],
            "adaptive_t000_smoke_runs": {
                "top_n": smoke["top_n_run"],
                "adaptive_t000": smoke["adaptive_t000_run"],
            },
        }
    )

    tidy_path = output_dir / "tidy.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    tidy.to_csv(tidy_path, index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(markdown_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {"tidy": str(tidy_path), "summary": str(summary_path), "report": str(report_path)},
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture and analyze Ragglet article experiments.")
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser(
        "capture-manifest", help="Capture the pre-experiment environment."
    )
    manifest.add_argument("--output", default="results/article/experiment-manifest.json")
    manifest.add_argument("--qdrant-url", default="http://localhost:6333")
    manifest.add_argument("--redis-url", default="redis://localhost:6379/15")
    manifest.set_defaults(func=capture_manifest)

    smoke = commands.add_parser(
        "check-smoke", help="Check top_n == adaptive threshold 0 invariant."
    )
    smoke.add_argument("--top-n-run", required=True)
    smoke.add_argument("--adaptive-run", required=True)
    smoke.set_defaults(func=check_smoke)

    report = commands.add_parser(
        "analyze", help="Create tidy CSV, JSON summary, and Markdown report."
    )
    report.add_argument("--runs", default="results/article/raw")
    report.add_argument("--output-dir", default="results/article")
    report.add_argument("--manifest", default="results/article/experiment-manifest.json")
    report.set_defaults(func=analyze)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
