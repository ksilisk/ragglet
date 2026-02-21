from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import time

import pandas as pd

from ragglet.config.scenario import ScenarioConfig
from ragglet.bench.dataset import load_jsonl_dataset
from ragglet.bench.metrics import recall_at_k, mrr_at_k
from ragglet.modules.embedders.st_embedder import SentenceTransformerEmbedder
from ragglet.stores.qdrant_store import QdrantStore

def _render_path(template: str, name: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(template.format(name=name, timestamp=ts))

def run_scenario(cfg: ScenarioConfig) -> dict:
    # Load dataset
    ds_path = Path(cfg.dataset.id)
    items = load_jsonl_dataset(ds_path, cfg.dataset.query_field, cfg.dataset.truth_field, cfg.dataset.limit)

    # Build components (minimal: embedder + qdrant)
    embedder = SentenceTransformerEmbedder(cfg.embedding.model, normalize=cfg.embedding.normalize)

    if cfg.storage.backend.type != "qdrant":
        raise NotImplementedError("Only qdrant backend is implemented in the first iteration")

    store = QdrantStore(cfg.storage.backend.endpoint, cfg.storage.backend.collection)

    rows = []
    lat_total = []
    lat_embed = []
    lat_retrieve = []

    for it in items:
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        vec = embedder.embed_one(it.query)
        t2 = time.perf_counter()

        candidates = store.search(vec, top_k=cfg.retrieval.top_k)
        t3 = time.perf_counter()

        retrieved_ids = []
        for c in candidates:
            ext = None
            if c.payload:
                ext = c.payload.get("external_id")
            retrieved_ids.append(str(ext) if ext is not None else str(c.id))

        truth = set(it.truth_ids)

        rec = {f"recall@{k}": recall_at_k(retrieved_ids, truth, k) for k in cfg.metrics.quality.recall_at_k}
        rr = mrr_at_k(retrieved_ids, truth, cfg.metrics.quality.mrr_at_k)

        row = {
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
        rows.append(row)

        lat_embed.append(row["lat_embed_ms"])
        lat_retrieve.append(row["lat_retrieve_ms"])
        lat_total.append(row["lat_total_ms"])

    df = pd.DataFrame(rows)

    # Summary
    summary = {
        "scenario": cfg.name,
        "n": len(df),
        "mrr_mean": float(df["mrr"].mean()) if len(df) else 0.0,
        "lat_total_p50": float(df["lat_total_ms"].quantile(0.50)) if len(df) else 0.0,
        "lat_total_p95": float(df["lat_total_ms"].quantile(0.95)) if len(df) else 0.0,
    }
    for k in cfg.metrics.quality.recall_at_k:
        col = f"recall@{k}"
        summary[f"{col}_mean"] = float(df[col].mean()) if len(df) else 0.0

    # Save artifacts
    out_path = _render_path(cfg.metrics.results_path, cfg.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = cfg.output.format
    if fmt in ("parquet", "both"):
        df.to_parquet(out_path.with_suffix(".parquet"), index=False)
    if fmt in ("csv", "both"):
        df.to_csv(out_path.with_suffix(".csv"), index=False)

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(pd.Series(summary).to_json(), encoding="utf-8")

    return {"summary": summary, "out_base": str(out_path)}