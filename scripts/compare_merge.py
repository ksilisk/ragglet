#!/usr/bin/env python3
"""Controlled merge comparison: exact dense retrieval, BM25 and two logical shards.

Uses public BEIR data through the existing dataset adapter. No Qdrant server,
new artifact protocol or production capture/replay mode is required.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean

import numpy as np

from ragglet.bench.dataset import eval_items, eval_sha256
from ragglet.bench.evaluation import evaluate_quality
from ragglet.config.scenario import MetricsQualityConfig
from ragglet.datasets.beir import document_text, read_corpus
from ragglet.datasets.checks import validate_dataset
from ragglet.datasets.sources import resolve_dataset
from ragglet.keyword.bm25 import BM25Index, tokenize
from ragglet.retrieval.merge import RetrievedItem, interleave_merge, rrf_merge, weighted_sum_merge


def variants(ranked, top_k):
    """A fixed comparison grid, not parameter selection on test qrels."""
    sources = list(ranked)
    for a, b in [(1, 1), (2, 1), (1, 2)]:
        weights = dict(zip(sources, [a, b], strict=True))
        label = f"{a}:{b}"
        for k in [10, 60, 100]:
            yield f"rrf/k={k}/w={label}", rrf_merge(ranked, weights, k, top_k, True)
        for normalization in ["minmax", "none"]:
            yield f"sum/{normalization}/w={label}", weighted_sum_merge(ranked, weights, top_k, normalize=normalization)
        yield f"interleave/weighted/w={label}", interleave_merge(ranked, weights, top_k, weighted=True)
    yield "interleave/round_robin", interleave_merge(ranked, {}, top_k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BeIR/nfcorpus")
    parser.add_argument("--source", choices=["local", "hf"], default="hf")
    parser.add_argument("--revision", help="Dataset revision, including qrels revision when required")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="Query limit; 0 means the full split")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model-revision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 10 or args.limit < 0 or args.threads < 1:
        parser.error("top-k >= 10, limit >= 0 and threads >= 1 are required")

    import torch
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    resolved = resolve_dataset(args.source, args.dataset, args.revision)
    dataset, warnings = validate_dataset(resolved.path, [args.split], label=args.dataset)
    queries = eval_items(dataset, args.split, args.limit)
    docs = sorted(read_corpus(resolved.path), key=lambda d: d.doc_id)
    if not queries or not docs:
        raise ValueError("the experiment needs a nonempty corpus and queries with qrels")
    revision = HfApi().model_info(args.model, revision=args.model_revision).sha
    model = SentenceTransformer(args.model, revision=revision, device="cpu")
    texts = [document_text(doc) for doc in docs]
    print(f"Encoding {len(docs)} documents and {len(queries)} queries", flush=True)
    corpus_vectors = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    query_vectors = model.encode([q.query for q in queries], normalize_embeddings=True, batch_size=64)
    similarities = query_vectors @ corpus_vectors.T
    bm25 = BM25Index([tokenize(text) for text in texts])
    permutation = np.random.default_rng(args.seed).permutation(len(docs))
    shards = [np.sort(part) for part in np.array_split(permutation, 2)]
    quality = MetricsQualityConfig(k=[10], doc=["ndcg", "recall", "mrr"])
    per_query = []
    metrics = defaultdict(list)
    exact_matches = 0
    score_matches = 0

    def candidate(index, score, source):
        doc_id = docs[int(index)].doc_id
        return RetrievedItem(doc_id, f"{doc_id}#0", doc_id, float(score), source, None)

    def dense(scores, indices, source):
        order = np.argsort(-scores[indices], kind="stable")[:args.top_k]
        return [candidate(i, scores[i], source) for i in indices[order]]

    for q, scores in zip(queries, similarities, strict=True):
        full = dense(scores, np.arange(len(docs)), "vector")
        keyword = [candidate(i, score, "keyword") for i, score in bm25.top_k(q.query, args.top_k)]
        sharded = {f"shard{i}": dense(scores, indices, f"shard{i}") for i, indices in enumerate(shards)}
        # Controlled overlaps/local duplicate injection check invariance on real candidates.
        overlap = {"a": full, "b": [replace(i, source="b") for i in full]}
        duplicated = {**overlap, "a": [full[0], *full]}
        assert rrf_merge(overlap, {}, 60, args.top_k, True) == rrf_merge(duplicated, {}, 60, args.top_k, True)
        assert weighted_sum_merge(overlap, {}, args.top_k) == weighted_sum_merge(duplicated, {}, args.top_k)
        for method, result in [
            ("single", full), ("keyword", keyword),
            ("one_source/rrf", rrf_merge({"vector": full}, {}, 60, args.top_k, True)),
            ("one_source/sum", weighted_sum_merge({"vector": full}, {}, args.top_k)),
            ("one_source/interleave", interleave_merge({"vector": full}, {}, args.top_k)),
            *[(f"hybrid/{name}", result) for name, result in variants({"vector": full, "keyword": keyword}, args.top_k)],
            *[(f"shards/{name}", result) for name, result in variants(sharded, args.top_k)],
        ]:
            values = evaluate_quality(quality, result, q.qrels)
            metrics[method].append(values)
            per_query.append({"qid": q.qid, "variant": method, **values})
            if method == "shards/sum/none/w=1:1":
                exact_matches += [i.chunk_id for i in result] == [i.chunk_id for i in full]
                score_matches += [i.ranking_score for i in result] == [i.score_raw for i in full]
    summary = [{"variant": name, **{key: mean(row[key] for row in rows) for key in rows[0]}}
               for name, rows in metrics.items()]
    report = {
        "dataset": {"id": args.dataset if args.source == "hf" else Path(args.dataset).name,
                    "source": args.source, "revision": resolved.revision, "split": args.split,
                    "corpus_sha256": dataset.corpus_sha256, "eval_sha256": eval_sha256(queries)},
        "model": {"id": args.model, "revision": revision, "max_seq_length": model.max_seq_length,
                  "normalize_embeddings": True},
        "conditions": {"seed": args.seed, "top_k": args.top_k, "queries": len(queries), "documents": len(docs),
                       "chunking": "none", "search": "exact cosine, CPU float32", "shard_sizes": [len(s) for s in shards],
                       "bm25": {"k1": 1.2, "b": .75}, "rrf_k": [10, 60, 100],
                       "weights": [[1, 1], [2, 1], [1, 2]], "normalization": ["minmax", "none"]},
        "warnings": [w.code for w in warnings],
        "equal_weight_raw_sum_matches_single": exact_matches,
        "equal_weight_raw_sum_matches_single_scores": score_matches,
        "summary": summary, "per_query": per_query,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(summary)} variants; raw shard merge matches single: "
          f"IDs {exact_matches}/{len(queries)}, scores {score_matches}/{len(queries)}")


if __name__ == "__main__":
    main()
