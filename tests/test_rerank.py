from __future__ import annotations

import anyio
import pytest

from ragglet.rerank.noop import NoopReranker
from ragglet.retrieval.merge import RetrievedItem


def _base_config(**rerank):
    pytest.importorskip("yaml")
    pytest.importorskip("pydantic")
    from ragglet.config.scenario import ScenarioConfig

    data = {
        "version": 1,
        "name": "test",
        "dataset": {"id": "datasets/demo"},
        "storage": {
            "mode": "single",
            "backend": {
                "type": "qdrant",
                "endpoint": "http://localhost:6333",
                "collection": "chunks_v1",
            },
        },
        "rerank": {"enabled": False, **rerank},
        "metrics": {
            "latency_steps": ["embed", "retrieve", "rerank", "total"],
            "quality": {"recall_at_k": [5, 10], "mrr_at_k": 30},
        },
    }
    return ScenarioConfig.model_validate(data)


def _item(chunk_id: str, score: float, text: str = "text") -> RetrievedItem:
    return RetrievedItem(
        point_id=chunk_id,
        chunk_id=chunk_id,
        score_raw=score,
        source="test",
        payload={"chunk_id": chunk_id, "doc_id": chunk_id, "text": text},
    )


def test_cross_encoder_requires_model_when_enabled():
    with pytest.raises(ValueError, match="rerank.model is required"):
        _base_config(enabled=True, provider="cross_encoder", model=None)


def test_noop_provider_does_not_require_model():
    pytest.importorskip("yaml")
    pytest.importorskip("pydantic")
    from ragglet.rerank.engine import build_reranker

    cfg = _base_config(enabled=True, provider="noop", model=None)
    reranker = build_reranker(cfg)
    assert isinstance(reranker, NoopReranker)


def test_noop_rerank_keeps_retrieval_order_and_cutoff():
    items = [_item("a", 0.1), _item("b", 0.9), _item("c", 0.3)]

    result = anyio.run(NoopReranker().rerank, "query", items, 3, 2)

    assert [row.item.chunk_id for row in result] == ["a", "b"]
    assert [row.score for row in result] == [0.1, 0.9]


def test_retrieval_cache_entry_can_store_items():
    pytest.importorskip("cachetools")
    from ragglet.cache.retrieval_lru import RetrievalCacheEntry

    items = [_item("a", 1.0), _item("b", 0.5)]
    entry = RetrievalCacheEntry(retrieved_ids=["a", "b"], retrieved_items=items)

    assert entry.retrieved_ids == ["a", "b"]
    assert entry.retrieved_items == items


def test_summary_includes_rerank_latency_when_requested():
    pd = pytest.importorskip("pandas")
    from ragglet.bench.summarize import summarize_repeat

    cfg = _base_config()
    df = pd.DataFrame(
        [
            {
                "mrr": 1.0,
                "recall@5": 1.0,
                "recall@10": 1.0,
                "lat_embed_ms": 1.0,
                "lat_retrieve_ms": 2.0,
                "lat_rerank_ms": 3.0,
                "lat_total_ms": 6.0,
                "emb_hit_rate": 0.0,
                "ret_hit_rate": 0.0,
                "timed_out": False,
                "timeout_stage": None,
            }
        ]
    )

    summary = summarize_repeat(cfg, df)

    assert summary["lat_rerank_p50"] == 3.0
    assert summary["lat_rerank_p95"] == 3.0
