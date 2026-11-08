from __future__ import annotations

import json
import math
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pandas as pd
import pytest
from test_load import _Embedder, _items, _scenario, runtime  # noqa: F401 -- fixture
from test_retrieval_timing import candidates, registry, scenario

from ragglet.bench import run_scenario as run_module
from ragglet.bench.query_loop import run_queries_once
from ragglet.bench.summarize import aggregate_over_repeats, summarize_repeat
from ragglet.cache.manager import CacheManager
from ragglet.core.errors import SourceNetworkTimeout, StageTimeout
from ragglet.generation.noop import NoopGenerator
from ragglet.rerank.noop import NoopReranker
from ragglet.stores.bm25_store import KeywordBm25Store


async def rows(cfg, backends=None, items=None, caches=None, generator=None, embedder=None):
    return await run_queries_once(
        cfg, items if items is not None else _items(2),
        backends or registry(SimpleNamespace(search=AsyncMock(return_value=candidates()))),
        embedder or _Embedder(), caches or CacheManager(cfg), NoopReranker(), generator or NoopGenerator(),
    )


@pytest.mark.parametrize("parallel", [False, True])
async def test_network_timeout_records_a_row_and_continues(parallel):
    cfg = scenario(**{"async": parallel})

    async def search(vector, **kwargs):
        if vector[0] == 1:
            raise SourceNetworkTimeout("timeout")
        return candidates()

    df = await rows(cfg, registry(SimpleNamespace(search=search)))
    assert list(df.outcome) == ["timeout", "ok"]
    assert df.iloc[0].timeout_stage == "retrieval"
    assert df.iloc[0]["doc_hit@1"] == 0
    assert df.iloc[0].lat_source_0_network_timeout_ms >= 0
    assert df.iloc[0][["lat_source_0_ms", "lat_sources_ms", "lat_merge_ms", "lat_retrieve_ms"]].isna().all()
    assert df.iloc[1][["lat_source_0_ms", "lat_source_1_ms", "lat_sources_ms", "lat_merge_ms"]].notna().all()
    assert df.iloc[1][["lat_source_0_network_timeout_ms", "lat_source_1_network_timeout_ms"]].isna().all()
    s = summarize_repeat(cfg, df)  # counters are independent of selected latency steps
    assert s["source_0_network_timeout_count"] == 1
    assert "lat_source_0_network_timeout_p50" not in s
    assert (s["ok_count"], s["timeout_count"]) == (1, 1)


async def test_retrieval_hit_has_no_reused_timers_and_cold_miss_is_measured():
    cfg = scenario()
    cfg.cache.enabled = True
    from ragglet.config.scenario import CacheRetrievalConfig
    cfg.cache.retrieval = CacheRetrievalConfig(size=10)
    items = [_items(1)[0]] * 2
    df = await rows(cfg, items=items)
    new = [c for c in df.columns if c.startswith(("lat_source", "lat_merge"))]
    assert list(df.ret_cache) == ["miss", "hit"]
    assert df.iloc[0][["lat_source_0_ms", "lat_source_1_ms", "lat_merge_ms"]].notna().all()
    assert df.iloc[1][new].isna().all()
    assert df.iloc[1].lat_retrieve_ms >= 0


async def test_all_hits_keep_numeric_columns_and_zero_counts():
    cfg = scenario()
    cfg.metrics.latency_steps = ["sources", "merge"]
    cache = CacheManager(cfg)
    cache.get_retrieval = lambda query: ([], "hit")
    df = await rows(cfg, caches=cache)
    new = [c for c in df.columns if c.startswith(("lat_source", "lat_merge"))]
    assert len(new) == 6
    assert df[new].isna().all().all()
    assert all(pd.api.types.is_float_dtype(df[c]) for c in new)
    s = summarize_repeat(cfg, df)
    assert s["sources_completed_count"] == s["merge_completed_count"] == s["source_0_completed_count"] == 0
    assert math.isnan(s["lat_source_0_p95"]) and math.isnan(s["lat_merge_p95"])


async def test_embedding_hit_still_measures_retrieval():
    cfg = scenario()
    cache = CacheManager(cfg)
    cache.get_embedding = AsyncMock(return_value=([1.0], "hit"))
    embedder = SimpleNamespace(embed_query=AsyncMock(side_effect=AssertionError("cache hit")))
    df = await rows(cfg, caches=cache, embedder=embedder)
    assert (df.emb_cache == "hit").all()
    assert df[["lat_source_0_ms", "lat_source_1_ms", "lat_sources_ms", "lat_merge_ms"]].notna().all().all()


async def test_timeout_before_retrieval_and_after_retrieval():
    cfg = scenario()
    cfg.timeouts.total_ms = 20

    async def slow_embed(query):
        await anyio.sleep_forever()

    df = await rows(cfg, embedder=SimpleNamespace(embed_query=slow_embed))
    assert (df.timeout_stage == "total").all()
    assert df[[c for c in df if c.startswith(("lat_source", "lat_merge"))]].isna().all().all()
    cfg.timeouts.total_ms = 5000
    cfg.context.enabled = True
    cfg.generation.enabled = True
    cfg.generation.model = "fake"
    generator = SimpleNamespace(generate=AsyncMock(side_effect=StageTimeout("generation")))
    df = await rows(cfg, generator=generator)
    assert (df.outcome == "timeout").all()
    assert df[["lat_source_0_ms", "lat_source_1_ms", "lat_sources_ms", "lat_merge_ms"]].notna().all().all()


async def test_bm25_timeout_waits_for_lane_and_is_not_a_network_timeout(monkeypatch):
    cfg = scenario(sources=[{"name": "bm25", "kind": "keyword"}])
    cfg.timeouts.retrieval_ms = 20
    store = KeywordBm25Store([{"doc_id": "d1", "chunk_id": "d1#0", "text": "x"}])
    finished = []

    def slow_search(*args):
        time.sleep(0.080)
        finished.append(True)
        return candidates()

    monkeypatch.setattr(store, "_search", slow_search)
    backends = registry(None)
    backends.keyword = store
    df = await rows(cfg, backends, items=_items(1))
    assert finished == [True]
    assert df.iloc[0].timeout_stage == "retrieval"
    assert df.iloc[0][["lat_source_0_ms", "lat_source_0_network_timeout_ms", "lat_merge_ms"]].isna().all()


async def test_percentiles_count_completed_suboperations_even_on_failed_queries():
    cfg = scenario()
    cfg.metrics.latency_steps = ["sources", "merge"]
    df = await rows(cfg, items=_items(4))
    df["lat_source_0_ms"] = [12, math.nan, 20, math.inf]
    df["lat_source_0_network_timeout_ms"] = [math.nan, 25, math.nan, 40]
    df["lat_sources_ms"] = [30, math.nan, 40, math.nan]
    df["lat_merge_ms"] = [3, math.nan, 5, math.nan]
    df.loc[2, ["outcome", "timeout_stage"]] = ["timeout", "generation"]
    s = summarize_repeat(cfg, df)
    assert s["source_0_completed_count"] == 2
    assert s["lat_source_0_p50"] == 16
    assert s["lat_source_0_p95"] == pytest.approx(19.6)
    assert s["source_0_network_timeout_count"] == 2
    assert s["lat_source_0_network_timeout_p50"] == 32.5
    assert (s["sources_completed_count"], s["merge_completed_count"]) == (2, 2)
    empty = df.copy()
    empty[[c for c in empty if c.startswith(("lat_source", "lat_merge"))]] = math.nan
    agg = aggregate_over_repeats(pd.DataFrame([s, summarize_repeat(cfg, empty)]), cfg.name, 2)
    assert agg["lat_source_0_p50_mean_over_repeats"] == 16  # empty repeat adds no zero
    assert agg["source_0_completed_count_mean_over_repeats"] == 1
    old = df.drop(columns=[c for c in df if c.startswith(("lat_source", "lat_merge"))])
    old_summary = summarize_repeat(cfg, old)
    assert not any(k.startswith(("source_", "sources_", "merge_", "lat_source", "lat_merge")) for k in old_summary)


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.usefixtures("runtime")
async def test_real_runner_writes_consistent_artifacts_and_counts(tmp_path, save):
    cfg = _scenario(tmp_path=tmp_path,
                    retrieval={"sources": [{"name": "used"}, {"name": "unused"}]},
                    metrics={"latency_steps": ["sources", "merge"], "quality": {"k": [1]}},
                    output={"format": "both"})
    cfg.run.save_per_query = save
    result = await run_module.run_scenario(cfg)
    summary = json.loads((result.run_dir / "summary.json").read_text())[0]
    assert summary["sources_count"] == 1
    assert summary["source_0_completed_count"] == 20
    assert summary["merge_completed_count"] == 0
    assert summary["lat_merge_p50"] is None
    assert "source_1_completed_count" not in summary
    if save:
        pq = pd.read_parquet(result.run_dir / "per_query.repeat0.parquet")
        csv = pd.read_csv(result.run_dir / "per_query.repeat0.csv")
        cols = [c for c in pq if c.startswith(("lat_source", "lat_merge"))]
        pd.testing.assert_frame_equal(pq[cols], csv[cols])
        recomputed = summarize_repeat(cfg, pq)
        assert recomputed["lat_source_0_p95"] == pytest.approx(summary["lat_source_0_p95"], abs=1e-8)
    else:
        assert not list(result.run_dir.glob("per_query.*"))
