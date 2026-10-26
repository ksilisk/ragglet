from __future__ import annotations

import pandas as pd
import pytest

from ragglet.bench.evaluation import evaluate_quality, project_to_docs, quality_columns
from ragglet.bench.metrics import Ranking, ndcg_at_k, precision_at_k, r_cap_at_k, recall_at_k
from ragglet.bench.summarize import summarize_repeat
from ragglet.config.scenario import MetricsQualityConfig, ScenarioConfig
from ragglet.retrieval.merge import RetrievedItem

ALL_LEVELS = MetricsQualityConfig(
    k=[1, 2, 3],
    doc=["recall", "hit", "mrr", "precision"],
    chunk=["hit", "mrr", "precision"],
)


def _items(*chunk_ids: str) -> list[RetrievedItem]:
    return [
        RetrievedItem(
            point_id=chunk_id,
            chunk_id=chunk_id,
            doc_id=chunk_id.rsplit("#", 1)[0],
            score_raw=0.0,
            source="test",
            payload=None,
        )
        for chunk_id in chunk_ids
    ]


def _scenario(metrics: dict | None = None, **overrides) -> ScenarioConfig:
    data = {
        "name": "test",
        "dataset": {"id": "datasets/demo"},
        "storage": {
            "backend": {"type": "qdrant", "endpoint": "http://localhost:6333", "collection": "c"},
        },
        "metrics": {"quality": metrics or {}},
        **overrides,
    }
    return ScenarioConfig.model_validate(data)


def test_recall_is_share_of_relevant_docs_not_hit():
    qrels = {"d1": 1, "d2": 1, "d3": 1, "d4": 1}
    row = evaluate_quality(ALL_LEVELS, _items("d1#0", "x#0", "d2#0"), qrels)

    assert row["doc_recall@3"] == 0.5
    assert row["doc_hit@3"] == 1.0
    assert row["doc_mrr@3"] == 1.0
    assert row["doc_precision@3"] == pytest.approx(2 / 3)


def test_recall_is_bounded_by_k_when_many_docs_are_relevant():
    qrels = {f"d{i}": 1 for i in range(1, 6)}
    row = evaluate_quality(ALL_LEVELS, _items("d1#0", "d2#0", "d3#0"), qrels)

    assert row["doc_recall@3"] == pytest.approx(3 / 5)
    assert row["doc_hit@3"] == 1.0


def test_single_relevant_doc_recall_equals_hit():
    row = evaluate_quality(ALL_LEVELS, _items("x#0", "d2#0", "y#0"), {"d2": 1})

    for k in ALL_LEVELS.k:
        assert row[f"doc_recall@{k}"] == row[f"doc_hit@{k}"]
    assert row["doc_hit@1"] == 0.0
    assert row["doc_hit@2"] == 1.0


def test_chunks_of_one_doc_do_not_shift_doc_ranks():
    items = _items("d2#0", "d2#1", "d1#0")
    row = evaluate_quality(ALL_LEVELS, items, {"d1": 1})

    assert project_to_docs(items) == ["d2", "d1"]
    assert row["doc_mrr@3"] == 0.5
    assert row["chunk_mrr@3"] == pytest.approx(1 / 3)
    assert (row["n_chunks"], row["n_docs"], row["dup_chunks"]) == (3, 2, 0)


def test_exact_chunk_duplicates_collapse_and_are_counted():
    row = evaluate_quality(ALL_LEVELS, _items("d1#0", "d1#0", "d2#0"), {"d1": 1})

    assert (row["n_chunks"], row["dup_chunks"]) == (2, 1)
    assert row["chunk_precision@2"] == 0.5
    assert row["chunk_mrr@2"] == 1.0


def test_every_chunk_of_relevant_doc_counts_at_chunk_level():
    row = evaluate_quality(ALL_LEVELS, _items("d1#0", "d1#1", "x#0"), {"d1": 1})

    assert row["chunk_precision@3"] == pytest.approx(2 / 3)
    assert row["doc_recall@2"] == 1.0
    assert row["doc_precision@2"] == 0.5


def test_passage_granular_dataset_levels_coincide():
    # один документ = один чанк (passage-датасет или chunking.method=none)
    row = evaluate_quality(ALL_LEVELS, _items("a#0", "b#0", "c#0"), {"b": 1, "c": 2})

    for name in ALL_LEVELS.chunk:
        for k in ALL_LEVELS.k:
            assert row[f"doc_{name}@{k}"] == row[f"chunk_{name}@{k}"]


def test_empty_final_list_is_a_miss():
    row = evaluate_quality(ALL_LEVELS, [], {"d1": 1})

    assert all(row[col] == 0.0 for col in quality_columns(ALL_LEVELS))
    assert (row["n_chunks"], row["n_docs"], row["dup_chunks"]) == (0, 0, 0)


def test_graded_qrels_count_as_relevant_for_binary_metrics():
    ranking = Ranking(grades=[2, 0, 1], ideal=[2, 1])
    assert recall_at_k(ranking, 3) == 1.0
    assert precision_at_k(Ranking(grades=[1], ideal=[1]), 3) == pytest.approx(1 / 3)


# пример из docs/metrics.md: нужны A, D (оценка 2) и B, C (оценка 1)
GRADED = {"A": 2, "D": 2, "B": 1, "C": 1}
ALL_DOC_METRICS = ["ndcg", "recall", "hit", "mrr", "precision", "r_cap"]


def test_graded_example_matches_pytrec_eval():
    quality = MetricsQualityConfig(k=[3, 5], doc=ALL_DOC_METRICS)
    row = evaluate_quality(quality, _items("X#0", "B#0", "A#0", "Y#0", "C#0"), GRADED)

    # ndcg_cut, recall, P, recip_rank из pytrec_eval 0.5
    assert row["doc_ndcg@3"] == pytest.approx(0.433544, abs=1e-6)
    assert row["doc_ndcg@5"] == pytest.approx(0.481280, abs=1e-6)
    assert row["doc_recall@5"] == 0.75
    assert row["doc_precision@5"] == 0.6
    assert row["doc_mrr@5"] == 0.5
    assert row["doc_hit@5"] == 1.0
    assert row["doc_r_cap@3"] == pytest.approx(2 / 3)
    assert row["doc_r_cap@5"] == 0.75


def test_only_ndcg_sees_order_of_grades():
    quality = MetricsQualityConfig(k=[5], doc=ALL_DOC_METRICS)
    before = evaluate_quality(quality, _items("X#0", "B#0", "A#0", "Y#0", "C#0"), GRADED)
    after = evaluate_quality(quality, _items("X#0", "A#0", "B#0", "Y#0", "C#0"), GRADED)

    assert after["doc_ndcg@5"] > before["doc_ndcg@5"]
    for name in ("recall", "hit", "mrr", "precision", "r_cap"):
        assert after[f"doc_{name}@5"] == before[f"doc_{name}@5"]


def test_ndcg_is_one_for_ideal_order():
    assert ndcg_at_k(Ranking(grades=[2, 2, 1, 1], ideal=[2, 2, 1, 1]), 4) == pytest.approx(1.0)
    # нужных больше k: идеал тоже обрезается до k
    assert ndcg_at_k(Ranking(grades=[2, 2], ideal=[2, 2, 1, 1]), 2) == pytest.approx(1.0)


@pytest.mark.parametrize(("rank", "expected"), [(1, 1.0), (2, 0.630930), (3, 0.5)])
def test_ndcg_discounts_single_relevant_doc_by_rank(rank, expected):
    grades = [0] * (rank - 1) + [1]
    assert ndcg_at_k(Ranking(grades=grades, ideal=[1]), 3) == pytest.approx(expected, abs=1e-6)


def test_ndcg_and_r_cap_are_zero_without_relevant_docs_or_results():
    nothing_relevant = Ranking(grades=[0, 0], ideal=[])
    nothing_found = Ranking(grades=[], ideal=[2, 1])

    assert ndcg_at_k(nothing_relevant, 2) == r_cap_at_k(nothing_relevant, 2) == 0.0
    assert ndcg_at_k(nothing_found, 5) == r_cap_at_k(nothing_found, 5) == 0.0


def test_chunks_of_one_doc_count_once_in_ndcg():
    quality = MetricsQualityConfig(k=[3], doc=["ndcg"])
    row = evaluate_quality(quality, _items("d1#0", "d1#1", "d2#0"), {"d1": 2, "d2": 1})

    assert row["doc_ndcg@3"] == pytest.approx(1.0)


def test_r_cap_is_recall_or_precision_depending_on_relevant_count():
    few = Ranking(grades=[1, 0, 1], ideal=[1, 1])
    many = Ranking(grades=[1, 0, 1], ideal=[1] * 38)

    assert r_cap_at_k(few, 3) == recall_at_k(few, 3) == 1.0
    assert r_cap_at_k(many, 3) == precision_at_k(many, 3) == pytest.approx(2 / 3)
    assert recall_at_k(many, 3) == pytest.approx(2 / 38)


def test_summary_reports_means_and_short_doc_depth():
    cfg = _scenario({"k": [1, 3], "chunk": ["precision"]})
    rows = [
        evaluate_quality(cfg.metrics.quality, _items("d1#0", "d1#1", "d1#2"), {"d1": 1}),
        evaluate_quality(cfg.metrics.quality, _items("x#0", "d2#0", "y#0"), {"d2": 1, "d3": 1}),
    ]
    df = pd.DataFrame(
        [
            {**row, "lat_embed_ms": 1.0, "lat_retrieve_ms": 1.0, "lat_total_ms": 2.0,
             "t_start_ms": 0.0, "outcome": "ok",
             "timeout_stage": None}
            for row in rows
        ]
    )

    summary = summarize_repeat(cfg, df)

    assert summary["doc_recall@3_mean"] == pytest.approx((1.0 + 0.5) / 2)
    assert summary["doc_hit@3_mean"] == 1.0
    assert summary["chunk_precision@3_mean"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert summary["doc_short@3_rate"] == 0.5
    assert summary["doc_short@1_rate"] == 0.0
    assert summary["n_docs_mean"] == 2.0


def test_quality_defaults_doc_level_only():
    quality = MetricsQualityConfig()
    assert quality.k == [1, 3, 5, 10]
    assert quality_columns(quality) == [
        f"doc_{name}@{k}" for name in ("ndcg", "recall", "hit", "mrr") for k in (1, 3, 5, 10)
    ]


@pytest.mark.parametrize(
    ("quality", "message"),
    [
        ({"chunk": ["recall"]}, "unknown for chunks"),
        ({"chunk": ["ndcg"]}, "unknown for chunks"),
        ({"chunk": ["r_cap"]}, "unknown for chunks"),
        ({"doc": ["map"]}, "unknown metrics"),
        ({"doc": ["hit", "hit"]}, "must be unique"),
        ({"k": [5, 5]}, "must be unique"),
        ({"k": [0]}, ">= 1"),
        ({"k": []}, "at least one"),
        ({"recall_at_k": [1, 5]}, "Extra inputs"),
    ],
)
def test_quality_config_rejects_invalid(quality, message):
    with pytest.raises(ValueError, match=message):
        _scenario(quality)


def test_k_must_fit_final_list_depth():
    with pytest.raises(ValueError, match=r"k=30 exceeds final list depth 10 \(rerank.output_top_n\)"):
        _scenario(
            {"k": [1, 30]},
            retrieval={"top_k": 50},
            rerank={"enabled": True, "provider": "noop", "input_top_k": 50, "output_top_n": 10},
        )
    with pytest.raises(ValueError, match=r"depth 5 \(retrieval.top_k\)"):
        _scenario({"k": [10]}, retrieval={"top_k": 5})
    assert _scenario({"k": [1, 30]}, retrieval={"top_k": 30}).metrics.quality.k == [1, 30]


class _FakeStore:
    def __init__(self, chunk_ids: list[str]):
        self.chunk_ids = chunk_ids
        self.calls = 0

    async def search(self, vector, top_k, params=None):
        from ragglet.stores.qdrant_store_async import Candidate

        self.calls += 1
        return [
            Candidate(
                id=chunk_id,
                score_raw=1.0 - i / 100,
                payload={"chunk_id": chunk_id, "doc_id": chunk_id.rsplit("#", 1)[0], "text": "t"},
            )
            for i, chunk_id in enumerate(self.chunk_ids[:top_k])
        ]


class _FakeEmbedder:
    def embed_batch(self, texts):
        return [[0.0] for _ in texts]

    async def embed_query(self, text):
        return self.embed_batch([text])[0]


async def test_query_loop_rows_and_retrieval_cache_hit():
    from ragglet.bench.dataset import QueryItem
    from ragglet.bench.query_loop import run_queries_once
    from ragglet.cache.manager import CacheManager
    from ragglet.generation.noop import NoopGenerator
    from ragglet.rerank.noop import NoopReranker
    from ragglet.retrieval.backends import BackendRegistry

    cfg = _scenario(
        {"k": [1, 3], "chunk": ["precision"]},
        retrieval={"top_k": 3},
        cache={"enabled": True, "retrieval": {"type": "memory_lru", "size": 10}},
    )
    store = _FakeStore(["x#0", "d1#0", "d1#1"])
    query = QueryItem(qid="q1", query="same", qrels={"d1": 1, "d2": 1})

    df = await run_queries_once(
        cfg,
        [query, query],
        BackendRegistry(vector=store),
        _FakeEmbedder(),
        CacheManager(cfg),
        NoopReranker(),
        NoopGenerator(),
    )

    assert store.calls == 1  # второй запрос — retrieval cache hit
    for _, row in df.iterrows():
        assert row["final_doc_ids"] == ["x", "d1"]
        assert row["doc_recall@3"] == 0.5
        assert row["doc_hit@1"] == 0.0
        assert row["doc_mrr@3"] == 0.5
        assert row["chunk_precision@3"] == pytest.approx(2 / 3)
        assert row["truth_ids"] == ["d1", "d2"]
    assert not any(col.startswith("recall@") or col == "mrr" for col in df.columns)
