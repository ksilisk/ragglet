from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest
from pydantic import ValidationError
from test_merge import ids, item
from test_preflight import _meta, _scenario, _Store

from ragglet.api.routers.meta import scenario_meta
from ragglet.bench.preflight import merge_warnings, preflight
from ragglet.cache.manager import CacheManager
from ragglet.config.scenario import RetrievalConfig, RetrievalMergeConfig, RetrievalSourceConfig
from ragglet.context.engine import assemble_context
from ragglet.core.issues import ValidationError as PreflightError
from ragglet.rerank.base import RerankedItem
from ragglet.rerank.engine import rerank_items
from ragglet.rerank.noop import NoopReranker
from ragglet.retrieval.backends import BackendRegistry
from ragglet.retrieval.engine import retrieve_items


def config(method="rrf", **merge):
    return _scenario(retrieval={
        "strategy": "fanout_merge", "top_k": 3,
        "sources": [{"name": s, "params": {"backend_ref": s}} for s in ["a", "b"]],
        "merge": {"method": method, **merge},
    }, metrics={"quality": {"k": [1, 3]}})


def candidates(source, scores):
    return [SimpleNamespace(id=f"{source}{i}", score_raw=score,
                            payload={"chunk_id": f"{source}{i}", "doc_id": f"{source}{i}", "text": "text"})
            for i, score in enumerate(scores)]


@pytest.mark.parametrize(("method", "options", "expected"), [
    ("rrf", {}, ["a0", "b0", "a1"]),
    ("weighted_sum", {"normalize": "none"}, ["a0", "a1", "a2"]),
    ("weighted_sum", {"normalize": "minmax"}, ["a0", "b0", "a1"]),
    ("interleave", {"interleave_weighted": False}, ["a0", "b0", "a1"]),
    ("interleave", {"interleave_weighted": True}, ["a0", "a1", "b0"]),
])
async def test_runtime_passes_settings_and_uses_configured_order(method, options, expected):
    cfg = config(method, **options)
    # 4:1 for interleave distinguishes weighting from ordinary round-robin.
    if method == "interleave":
        cfg.retrieval.sources[0].weight = 4
    async def slow(*args, **kwargs):
        await anyio.sleep(.005)
        return candidates("a", [.9, .8, .7])
    a = SimpleNamespace(search=slow)
    b = SimpleNamespace(search=AsyncMock(return_value=candidates("b", [.2, .1, 0])))
    backend = BackendRegistry(vector=a, vectors={"a": a, "b": b})
    cfg.retrieval.async_enabled = False
    sequential = await retrieve_items(cfg, backend, "q", [1.0])
    cfg.retrieval.async_enabled = True
    parallel = await retrieve_items(cfg, backend, "q", [1.0])
    assert parallel == sequential
    assert ids(parallel) == expected


@pytest.mark.parametrize("method", ["rrf", "weighted_sum", "interleave"])
async def test_single_and_one_source_fanout_have_same_order(method):
    cfg = config(method)
    cfg.retrieval.sources = cfg.retrieval.sources[:1]
    backend = SimpleNamespace(search=AsyncMock(return_value=candidates("a", [.9, .8, .7])))
    backends = BackendRegistry(vector=backend, vectors={"a": backend})
    merged = await retrieve_items(cfg, backends, "q", [1.0])
    cfg.retrieval.strategy = "single"
    single = await retrieve_items(cfg, backends, "q", [1.0])
    assert ids(merged) == ids(single) == ["a0", "a1", "a2"]
    assert [i.score_raw for i in single] == [.9, .8, .7]
    assert all(i.score_fused is None for i in single)


async def test_hybrid_backend_weights_use_fused_scores():
    cfg = config("weighted_sum")
    cfg.retrieval.sources[1].kind = "keyword"
    cfg.retrieval.sources[1].weight = 2
    vector = SimpleNamespace(search=AsyncMock(return_value=candidates("a", [.9, .8, .7])))
    keyword = SimpleNamespace(search=AsyncMock(return_value=candidates("b", [200, 100, 0])))
    result = await retrieve_items(cfg, BackendRegistry(vector=vector, vectors={"a": vector}, keyword=keyword), "query", [.2])
    assert ids(result) == ["b0", "a0", "b1"]
    assert result[0].score_raw == 200
    assert result[0].score_fused == 2
    keyword.search.assert_awaited_once_with("query", top_k=3)


@pytest.mark.parametrize("weight", [-1, float("inf"), float("nan")])
def test_schema_rejects_invalid_weights(weight):
    with pytest.raises(ValidationError):
        RetrievalSourceConfig(name="a", weight=weight)


def test_schema_defaults_and_empty_or_zero_weight_fanout():
    assert RetrievalMergeConfig().normalize == "minmax"
    assert not RetrievalMergeConfig().interleave_weighted
    for data in [{"strategy": "fanout_merge"}, {"strategy": "fanout_merge", "sources": [{"name": "a", "weight": 0}]}]:
        with pytest.raises(ValidationError):
            RetrievalConfig.model_validate(data)
    for data in [{"rrf_k": 0}, {"rrf_k": -1}, {"normalize": "zscore"}]:
        with pytest.raises(ValidationError):
            RetrievalMergeConfig.model_validate(data)


def test_meta_includes_settings_and_limits():
    meta = scenario_meta()
    assert meta.defaults["retrieval"]["merge"]["normalize"] == "minmax"
    assert meta.defaults["retrieval"]["merge"]["interleave_weighted"] is False
    props = meta.json_schema["$defs"]["RetrievalMergeConfig"]["properties"]
    assert props["normalize"]["enum"] == ["minmax", "none"]
    assert props["rrf_k"]["minimum"] == 1


@pytest.mark.parametrize("field,value", [("normalize", "none"), ("interleave_weighted", True), ("rrf_k", 10), ("deduplicate", False)])
def test_cache_signature_covers_merge_config_and_preserves_scores(field, value):
    cfg = config()
    cfg.cache.enabled = True
    from ragglet.config.scenario import CacheRetrievalConfig
    cfg.cache.retrieval = CacheRetrievalConfig()
    caches = CacheManager(cfg)
    result = [replace(item("a", 100), score_fused=0)]
    caches.set_retrieval("q", result)
    assert caches.get_retrieval("q") == (result, "hit")
    assert caches.get_retrieval("q")[0][0].ranking_score == 0
    setattr(cfg.retrieval.merge, field, value)
    assert caches.get_retrieval("q") == (None, "miss")


async def test_context_and_noop_use_fused_zero_without_changing_real_rerank():
    cfg = config()
    cfg.context.enabled = True
    cfg.context.policy = "threshold"
    cfg.context.threshold = .5
    items = [replace(item("raw-high", 100), score_fused=0), replace(item("fused-high", .1), score_fused=1)]
    assert assemble_context(cfg, items).ids == ["fused-high"]
    noop = NoopReranker()
    assert [x.score for x in await rerank_items(cfg, noop, "q", items)] == [0, 1]
    cfg.rerank.enabled = True
    cfg.rerank.provider = "noop"
    assert [x.score for x in await rerank_items(cfg, noop, "q", items)] == [0, 1]
    cfg.rerank.provider = "cross_encoder"
    real = SimpleNamespace(rerank=AsyncMock(return_value=[RerankedItem(items[0], 7)]))
    assert (await rerank_items(cfg, real, "q", items))[0].score == 7


@pytest.mark.parametrize("distance", ["Cosine", "Dot", "Euclid", None])
@pytest.mark.parametrize("params", [{}, {"backend_ref": None}])
async def test_preflight_checks_direction_only_for_score_fusion(distance, params):
    cfg = _scenario(retrieval={"strategy": "fanout_merge", "sources": [{"name": "a", "params": params}],
                              "merge": {"method": "weighted_sum"}})
    store = _Store(_meta())
    store.info = replace(store.info, params={**store.info.params, "distance": distance})
    if distance in ("Cosine", "Dot"):
        await preflight(cfg, {"default": store})
    else:
        with pytest.raises(PreflightError) as exc:
            await preflight(cfg, {"default": store})
        assert any(i.code == "retrieval.score_direction" and i.path == "retrieval.sources[0]" for i in exc.value.issues)
        cfg.retrieval.merge.method = "rrf"
        await preflight(cfg, {"default": store})


async def test_legacy_dedup_flag_is_accepted_and_warned_by_preflight():
    cfg = _scenario(retrieval={"strategy": "fanout_merge", "sources": [{"name": "a"}],
                              "merge": {"deduplicate": False}})
    checked = await preflight(cfg, {"default": _Store(_meta())})
    assert any(i.code == "retrieval.deduplicate_ignored" for i in checked.warnings)
    cfg.retrieval.merge.method = "interleave"
    assert list(merge_warnings(cfg)) == []
    cfg.context.enabled = True
    cfg.context.policy = "adaptive"
    assert [i.code for i in merge_warnings(cfg)] == ["retrieval.interleave_raw_scores"]
    cfg.rerank.enabled = True
    cfg.rerank.provider = "noop"
    assert len(list(merge_warnings(cfg))) == 1
    cfg.rerank.provider = "cross_encoder"
    assert list(merge_warnings(cfg)) == []
