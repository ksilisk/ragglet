"""Синтетические каталоги прогонов для тестов реестра: форматы как у ядра, без моделей и Qdrant."""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from api_support import DEMO, MODEL_SHA, PARAMS, QDRANT

from ragglet.bench.artifacts import save_config_snapshot, save_manifest
from ragglet.bench.dataset import load_eval_dataset
from ragglet.bench.summarize import aggregate_over_repeats, summarize_repeat
from ragglet.config.scenario import ScenarioConfig

RUN_ID = "demo_run_20261024T180917Z"
BUILT_AT = "2026-10-20T12:00:00Z"  # как в api_support.index_meta()
QIDS = ("q001", "q003", "q007")
TRUTH = {  # из datasets/demo/qrels/test.tsv
    "q001": "rag_overview_definition_and_pipeline",
    "q003": "rag_retrieval_vs_generation_failure_modes",
    "q007": "retrieval_recall_at_k_mrr_ndcg_explained",
}
TEXT = {
    "q001": "What is Retrieval-Augmented Generation (RAG) and what are the main stages of its pipeline?",
    "q003": "What are common failure modes in RAG systems?",
    "q007": "[demo] recall, mrr and ndcg explained",  # начинается со скобки: CSV-разбор не должен его трогать
}


def demo_scenario(name: str = "demo_run", **overrides) -> ScenarioConfig:
    data = {
        "name": name,
        "tags": ["demo"],
        "dataset": {"id": str(DEMO), "limit": 100},
        "storage": {"backend": {"type": "qdrant", "endpoint": QDRANT, "collection": "chunks_v1"}},
        "embedding": {"model": "all-MiniLM-L6-v2", "normalize": True},
        "rerank": {"enabled": True, "model": "cross-encoder/x"},
        "context": {"enabled": True},
        "run": {"repeats": 2, "shuffle": False},
        "metrics": {"quality": {"k": [1, 3], "doc": ["ndcg", "recall", "hit"], "chunk": []},
                    "percentiles": [50, 95]},
        **overrides,
    }
    return ScenarioConfig.model_validate(data)


def _rows(qids: Sequence[str], repeat: int) -> list[dict]:
    """
    Первый запрос: релевантный документ на 1-м месте двумя чанками подряд. Второй: на 2-м.
    Третий: таймаут на реранке — выдачи нет, у незавершённых стадий NaN.
    """
    nan = math.nan
    a, b, c = qids
    ta, tb, tc = TRUTH["q001"], TRUTH["q003"], TRUTH["q007"]
    other = "rag_evaluation_overview_metrics_and_protocols"
    noise = "chunking_overlap_tradeoffs_and_guidelines"
    shift = 10.0 * repeat
    return [
        {
            "qid": a, "query": TEXT["q001"], "truth_ids": [ta],
            "retrieved_ids": [f"{ta}#0", f"{other}#0", f"{ta}#1"], "retrieved_doc_ids": [ta, other],
            "reranked_ids": [f"{ta}#0", f"{ta}#1", f"{other}#0"], "context_ids": [f"{ta}#0", f"{ta}#1"],
            "context_text": "chunk text", "answer": "",
            "final_ids": [f"{ta}#0", f"{ta}#1", f"{other}#0"], "final_doc_ids": [ta, other],
            "rerank_scores": [0.9, 0.8, 0.1],
            "doc_ndcg@1": 1.0, "doc_ndcg@3": 1.0, "doc_recall@1": 1.0, "doc_recall@3": 1.0,
            "doc_hit@1": 1.0, "doc_hit@3": 1.0, "n_chunks": 3, "n_docs": 2, "dup_chunks": 0,
            "t_start_ms": 0.0, "lat_embed_ms": 5.0, "lat_retrieve_ms": 8.0, "lat_rerank_ms": 50.0 + shift,
            "lat_context_ms": 1.0, "lat_generation_ms": nan, "lat_total_ms": 64.0 + shift,
            "emb_cache": None, "ret_cache": None, "outcome": "ok", "timeout_stage": None,
        },
        {
            "qid": b, "query": TEXT["q003"], "truth_ids": [tb],
            "retrieved_ids": [f"{noise}#0", f"{tb}#2"], "retrieved_doc_ids": [noise, tb],
            "reranked_ids": [f"{noise}#0", f"{tb}#2"], "context_ids": [f"{noise}#0"],
            "context_text": "chunk text", "answer": "",
            "final_ids": [f"{noise}#0", f"{tb}#2"], "final_doc_ids": [noise, tb],
            "rerank_scores": [0.7, 0.6],
            "doc_ndcg@1": 0.0, "doc_ndcg@3": 1 / math.log2(3), "doc_recall@1": 0.0, "doc_recall@3": 1.0,
            "doc_hit@1": 0.0, "doc_hit@3": 1.0, "n_chunks": 2, "n_docs": 2, "dup_chunks": 0,
            "t_start_ms": 64.0, "lat_embed_ms": 5.0, "lat_retrieve_ms": 9.0, "lat_rerank_ms": 300.0 + shift,
            "lat_context_ms": 1.0, "lat_generation_ms": nan, "lat_total_ms": 315.0 + shift,
            "emb_cache": None, "ret_cache": None, "outcome": "ok", "timeout_stage": None,
        },
        {
            "qid": c, "query": TEXT["q007"], "truth_ids": [tc],
            "retrieved_ids": [f"{tc}#0"], "retrieved_doc_ids": [tc],
            "reranked_ids": [], "context_ids": [], "context_text": "", "answer": "",
            "final_ids": [], "final_doc_ids": [], "rerank_scores": [],
            "doc_ndcg@1": 0.0, "doc_ndcg@3": 0.0, "doc_recall@1": 0.0, "doc_recall@3": 0.0,
            "doc_hit@1": 0.0, "doc_hit@3": 0.0, "n_chunks": 0, "n_docs": 0, "dup_chunks": 0,
            "t_start_ms": 379.0, "lat_embed_ms": 5.0, "lat_retrieve_ms": 9.0, "lat_rerank_ms": nan,
            "lat_context_ms": nan, "lat_generation_ms": nan, "lat_total_ms": 900.0,
            "emb_cache": None, "ret_cache": None, "outcome": "timeout", "timeout_stage": "rerank",
        },
    ]


def make_run_dir(
        root: Path,
        run_id: str = RUN_ID,
        *,
        repeats: int = 2,
        formats: Sequence[str] = ("parquet", "csv"),
        complete: bool = True,
        manifest: bool = True,
        qids: Sequence[str] = QIDS,
        cfg: ScenarioConfig | None = None,
) -> Path:
    """Каталог прогона текущего формата, как его пишет run_scenario; complete=False — упал до summary."""
    cfg = cfg or demo_scenario(run={"repeats": repeats, "shuffle": False})
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    save_config_snapshot(run_dir, cfg)
    if manifest:
        dataset = load_eval_dataset(cfg.dataset)
        save_manifest(run_dir, {
            "dataset": {"revision": None, "corpus_sha256": dataset.corpus_sha256, "eval_sha256": dataset.eval_sha256},
            "index": {"profile_id": "demo", "built_at": BUILT_AT, "embedding_revision": MODEL_SHA,
                      "chunking": {"method": "recursive", "chunk_size": 800, "chunk_overlap": 120},
                      "params": PARAMS, "build_env": {"commit": "c" * 40}},
            "rerank": {"revision": "r" * 40},
            "env": {"commit": "e" * 40, "python": "3.12.6"},
        })
    summaries = []
    for r in range(repeats):
        df = pd.DataFrame(_rows(qids, r))
        summary = summarize_repeat(cfg, df)
        summary["repeat_idx"] = r
        summaries.append(summary)
        if "csv" in formats:
            df.to_csv(run_dir / f"per_query.repeat{r}.csv", index=False)
        if "parquet" in formats:
            df.to_parquet(run_dir / f"per_query.repeat{r}.parquet", index=False)
    if complete:
        df_sum = pd.DataFrame(summaries)
        (run_dir / "summary.json").write_text(df_sum.to_json(orient="records", indent=2), encoding="utf-8")
        (run_dir / "summary.csv").write_text(df_sum.to_csv(index=False), encoding="utf-8")
        agg = aggregate_over_repeats(df_sum, cfg.name, repeats)
        (run_dir / "summary.aggregate.json").write_text(pd.Series(agg).to_json(indent=2), encoding="utf-8")
        pd.DataFrame([agg]).to_csv(run_dir / "summary.aggregate.csv", index=False)
    return run_dir


def make_feb_run_dir(root: Path, run_id: str = "baseline_qdrant_20260221T135048Z") -> Path:
    """Февральское поколение: только CSV, recall@k вместо doc_*, без outcome, final_doc_ids и manifest."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    snapshot = {
        "version": 1, "name": "baseline_qdrant", "tags": ["baseline"],
        "dataset": {"id": "data/queries/baseline_qdrant.jsonl", "limit": 2000, "query_field": "question"},
        "storage": {"mode": "single", "backend": {"type": "qdrant", "endpoint": "http://localhost:6333",
                                                  "collection": "chunks_v1"}},
        "run": {"repeats": 1, "artifacts_dir": "results/runs"},
    }
    (run_dir / "scenario.snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    pd.DataFrame([
        {"qid": "q1", "query": "a", "truth_ids": ["d1"], "retrieved_ids": ["d1", "d2"], "mrr": 1.0,
         "recall@1": 1.0, "recall@3": 1.0, "lat_embed_ms": 5.0, "lat_retrieve_ms": 7.0, "lat_total_ms": 12.0},
    ]).to_csv(run_dir / "per_query.repeat0.csv", index=False)
    summary = [{"scenario": "baseline_qdrant", "n": 1, "recall@1_mean": 1.0, "recall@3_mean": 1.0,
                "mrr_mean": 1.0, "lat_total_p50": 12.0, "lat_total_p95": 12.0, "repeat_idx": 0}]
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    agg = {"scenario": "baseline_qdrant", "repeats": 1, "recall@1_mean_mean_over_repeats": 1.0,
           "lat_total_p50_mean_over_repeats": 12.0, "lat_total_p95_mean_over_repeats": 12.0}
    (run_dir / "summary.aggregate.json").write_text(json.dumps(agg), encoding="utf-8")
    (run_dir / "summary.aggregate.csv").write_text("scenario,repeats\nbaseline_qdrant,1\n", encoding="utf-8")
    return run_dir
