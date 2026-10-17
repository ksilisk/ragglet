from __future__ import annotations

import pytest

from ragglet.context.token_count import estimate_tokens
from ragglet.retrieval.merge import RetrievedItem


def _config(policy: str = "top_n", max_tokens: int = 10, threshold: float = 0.0):
    pytest.importorskip("yaml")
    pytest.importorskip("pydantic")
    from ragglet.config.scenario import ScenarioConfig

    return ScenarioConfig.model_validate(
        {
            "version": 1,
            "name": "context-test",
            "dataset": {"id": "datasets/demo"},
            "storage": {
                "mode": "single",
                "backend": {
                    "type": "qdrant",
                    "endpoint": "http://localhost:6333",
                    "collection": "chunks_v1",
                },
            },
            "context": {
                "enabled": True,
                "policy": policy,
                "max_tokens": max_tokens,
                "threshold": threshold,
                "include_metadata": True,
            },
        }
    )


def _item(chunk_id: str, score: float, text: str) -> RetrievedItem:
    return RetrievedItem(
        point_id=chunk_id,
        chunk_id=chunk_id,
        doc_id=chunk_id,
        score_raw=score,
        source="test",
        payload={"chunk_id": chunk_id, "doc_id": chunk_id, "text": text},
    )


def test_estimate_tokens_uses_words():
    assert estimate_tokens("one two three") == 3
    assert estimate_tokens("   ") == 0


def test_top_n_context_respects_budget():
    pytest.importorskip("yaml")
    pytest.importorskip("pydantic")
    from ragglet.context.engine import assemble_context

    cfg = _config(policy="top_n", max_tokens=3)
    result = assemble_context(
        cfg,
        [
            _item("a", 1.0, "one two"),
            _item("b", 0.9, "three four"),
            _item("c", 0.8, "five"),
        ],
    )

    assert result.ids == ["a", "c"]
    assert result.token_count == 3


def test_threshold_context_filters_by_raw_score():
    pytest.importorskip("yaml")
    pytest.importorskip("pydantic")
    from ragglet.context.engine import assemble_context

    cfg = _config(policy="threshold", max_tokens=10, threshold=0.5)
    result = assemble_context(
        cfg,
        [
            _item("a", 0.4, "low"),
            _item("b", 0.6, "high"),
        ],
    )

    assert result.ids == ["b"]


def test_adaptive_context_falls_back_to_first_candidate():
    pytest.importorskip("yaml")
    pytest.importorskip("pydantic")
    from ragglet.context.engine import assemble_context

    cfg = _config(policy="adaptive", max_tokens=10, threshold=1.1)
    result = assemble_context(
        cfg,
        [
            _item("a", 0.4, "first"),
            _item("b", 0.6, "second"),
        ],
    )

    assert result.ids == ["a"]


def test_summary_includes_context_latency_and_counts():
    pd = pytest.importorskip("pandas")
    from ragglet.bench.summarize import summarize_repeat

    cfg = _config(policy="adaptive", max_tokens=10, threshold=0.0)
    cfg.metrics.latency_steps.append("context")
    df = pd.DataFrame(
        [
            {
                "mrr": 1.0,
                "recall@1": 1.0,
                "recall@3": 1.0,
                "recall@5": 1.0,
                "recall@10": 1.0,
                "recall@30": 1.0,
                "lat_embed_ms": 1.0,
                "lat_retrieve_ms": 2.0,
                "lat_context_ms": 4.0,
                "lat_total_ms": 7.0,
                "context_tokens": 12,
                "context_chunks": 3,
                "emb_hit_rate": 0.0,
                "ret_hit_rate": 0.0,
                "timed_out": False,
                "timeout_stage": None,
            }
        ]
    )

    summary = summarize_repeat(cfg, df)

    assert summary["lat_context_p50"] == 4.0
    assert summary["context_tokens_mean"] == 12.0
    assert summary["context_chunks_mean"] == 3.0
