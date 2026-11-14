from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import httpx
import pandas as pd
import pytest
from pydantic import ValidationError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from test_load import _items
from test_retrieval_metrics import rows
from test_retrieval_timing import candidates, registry, scenario

from ragglet.api.routers.meta import scenario_meta
from ragglet.bench.summarize import aggregate_over_repeats, summarize_repeat
from ragglet.config.scenario import CacheRetrievalConfig
from ragglet.core.errors import SourceNetworkTimeout, SourceUnavailable, StageTimeout
from ragglet.registry.per_query import query_page, query_row
from ragglet.registry.protocol import QueryFilter
from ragglet.retrieval import engine
from ragglet.retrieval.diagnostics import RetrievalDiagnostics
from ragglet.stores.qdrant_store_async import AsyncQdrantStore


def store(*, error=None, found=None):
    return SimpleNamespace(search=AsyncMock(side_effect=error, return_value=candidates() if found is None else found))


def test_policy_defaults_validation_and_api_metadata():
    assert scenario().retrieval.failure_policy == "strict"
    with pytest.raises(ValidationError):
        scenario(failure_policy="ignore_errors")
    with pytest.raises(ValidationError):
        scenario(sources=[{"name": "a", "test_fault": {"mode": "timeout"}}])
    meta = scenario_meta()
    assert meta.defaults["retrieval"]["failure_policy"] == "strict"
    props = meta.json_schema["$defs"]["RetrievalConfig"]["properties"]
    assert props["failure_policy"]["enum"] == ["strict", "allow_partial"]


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("policy", ["strict", "allow_partial"])
@pytest.mark.parametrize("failure, outcome, status", [
    (SourceUnavailable("disconnected"), "error", "unavailable"),
    (SourceNetworkTimeout("read timed out"), "timeout", "timeout"),
])
async def test_source_failure_records_rows_and_partial_quality(parallel, policy, failure, outcome, status):
    cfg = scenario(**{"async": parallel, "failure_policy": policy})
    df = await rows(cfg, registry(store(), store(error=failure)), items=_items(1) * 2)
    partial = policy == "allow_partial"
    assert list(df.outcome) == ["ok" if partial else outcome] * 2
    assert list(df.retrieval_status) == ["partial" if partial else "failed"] * 2
    assert list(df.source_0_status) == ["ok"] * 2
    assert list(df.source_1_status) == [status] * 2
    assert list(df["doc_hit@1"]) == [1.0 if partial else 0.0] * 2
    assert df.lat_source_0_ms.notna().all()
    assert df.lat_source_1_ms.isna().all()
    summary = summarize_repeat(cfg, df)
    assert summary["partial_count"] == (2 if partial else 0)
    assert summary["partial_rate"] == (1.0 if partial else 0.0)
    assert summary["ok_count"] + summary["timeout_count"] + summary["error_count"] == 2


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("policy", ["strict", "allow_partial"])
async def test_deadline_preserves_ready_results_and_waits_for_cancellation(parallel, policy):
    cfg = scenario(**{"async": parallel, "failure_policy": policy})
    cfg.timeouts.retrieval_ms = 30
    stopped = []

    async def pending(*args, **kwargs):
        try:
            await anyio.sleep_forever()
        finally:
            stopped.append(True)

    df = await rows(cfg, registry(store(), SimpleNamespace(search=pending)), items=_items(1))
    row = df.iloc[0]
    assert stopped == [True]
    assert row.source_0_status == "ok" and row.source_1_status == "timeout"
    assert pd.isna(row.lat_source_1_network_timeout_ms)
    if policy == "allow_partial":
        assert row.outcome == "ok" and row.retrieval_status == "partial"
        assert row.final_doc_ids == ["d1", "d2"]
        assert row.lat_sources_ms >= 20 and row.lat_merge_ms >= 0
    else:
        assert row.outcome == "timeout" and row.timeout_stage == "retrieval"
        assert row.final_doc_ids == [] and pd.isna(row.lat_merge_ms)


async def test_strict_cancels_an_already_running_sibling():
    started = anyio.Event()
    stopped = []

    async def pending(*args, **kwargs):
        started.set()
        try:
            await anyio.sleep_forever()
        finally:
            stopped.append(True)

    async def fail(*args, **kwargs):
        await started.wait()
        raise SourceUnavailable("offline")

    df = await rows(scenario(), registry(SimpleNamespace(search=pending), SimpleNamespace(search=fail)))
    assert stopped == [True, True]
    assert list(df.source_0_status) == ["cancelled", "cancelled"]
    assert list(df.source_1_status) == ["unavailable", "unavailable"]


@pytest.mark.parametrize("policy", ["strict", "allow_partial"])
async def test_sequential_failure_only_stops_later_sources_in_strict(policy):
    cfg = scenario(**{"async": False, "failure_policy": policy})
    b = store()
    df = await rows(cfg, registry(store(error=SourceUnavailable()), b), items=_items(1))
    assert df.iloc[0].source_1_status == ("not_started" if policy == "strict" else "ok")
    assert b.search.await_count == (0 if policy == "strict" else 1)


@pytest.mark.parametrize("failures, expected", [
    ([SourceNetworkTimeout(), SourceNetworkTimeout()], "timeout"),
    ([SourceUnavailable(), SourceUnavailable()], "error"),
    ([SourceNetworkTimeout(), SourceUnavailable()], "error"),
])
async def test_no_successful_source_is_never_an_ok_answer(failures, expected):
    cfg = scenario(failure_policy="allow_partial")
    df = await rows(cfg, registry(*(store(error=e) for e in failures)), items=_items(1))
    assert df.iloc[0].outcome == expected
    assert df.iloc[0].retrieval_status == "failed"
    assert df.iloc[0].final_doc_ids == []


async def test_empty_success_is_partial_but_zero_weight_cannot_rescue_failure():
    cfg = scenario(failure_policy="allow_partial")
    backends = registry(store(found=[]), store(error=SourceUnavailable()))
    df = await rows(cfg, backends, items=_items(1))
    assert df.iloc[0].outcome == "ok" and df.iloc[0].retrieval_status == "partial"
    assert df.iloc[0]["doc_hit@1"] == 0
    cfg.retrieval.sources[0].weight = 0
    df = await rows(cfg, backends, items=_items(1))
    assert df.iloc[0].outcome == "error" and df.iloc[0].retrieval_status == "failed"


@pytest.mark.parametrize("policy", ["strict", "allow_partial"])
async def test_single_records_failure_and_continues(policy):
    cfg = scenario(strategy="single", failure_policy=policy)
    a = store()
    a.search.side_effect = [SourceUnavailable(), candidates()]
    df = await rows(cfg, registry(a))
    assert list(df.outcome) == ["error", "ok"]
    assert list(df.retrieval_status) == ["failed", "complete"]
    assert "source_1_status" not in df


@pytest.mark.parametrize("policy", ["strict", "allow_partial"])
async def test_total_timeout_and_external_cancellation_win(policy):
    cfg = scenario(failure_policy=policy)
    cfg.timeouts.total_ms = 25

    async def pending(*args, **kwargs):
        await anyio.sleep_forever()

    backends = registry(store(), SimpleNamespace(search=pending))
    df = await rows(cfg, backends, items=_items(1))
    assert df.iloc[0].outcome == "timeout" and df.iloc[0].timeout_stage == "total"
    assert df.iloc[0].retrieval_status == "failed"
    assert df.iloc[0].source_1_status == "cancelled"
    with anyio.move_on_after(.025) as scope:
        await engine.retrieve_items(cfg, backends, "q", [1.0])
        pytest.fail("external cancellation swallowed")
    assert scope.cancelled_caught


async def test_total_timeout_during_synchronous_merge_is_observed(monkeypatch):
    cfg = scenario(failure_policy="allow_partial")
    cfg.timeouts.total_ms = 30
    original = engine.rrf_merge

    def slow_merge(*args, **kwargs):
        time.sleep(.060)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "rrf_merge", slow_merge)
    df = await rows(cfg, registry(store(), store(error=SourceUnavailable())), items=_items(1))
    assert df.iloc[0].outcome == "timeout" and df.iloc[0].timeout_stage == "total"
    assert df.iloc[0].final_doc_ids == []


async def test_partial_not_cached_and_recovery_becomes_complete_then_cache_hit():
    cfg = scenario(failure_policy="allow_partial")
    cfg.cache.enabled = True
    cfg.cache.retrieval = CacheRetrievalConfig()
    b = store()
    b.search.side_effect = [SourceUnavailable(), candidates()]
    df = await rows(cfg, registry(store(), b), items=_items(1) * 3)
    assert list(df.retrieval_status) == ["partial", "complete", "cache_hit"]
    assert list(df.ret_cache) == ["miss", "miss", "hit"]
    assert df.iloc[2].source_0_status == df.iloc[2].source_1_status == "not_started"
    summary = summarize_repeat(cfg, df)
    assert summary["partial_count"] == 1 and summary["partial_rate"] == pytest.approx(1 / 3)
    aggregate = aggregate_over_repeats(pd.DataFrame([summary, summary]), cfg.name, 2)
    assert aggregate["partial_count_mean_over_repeats"] == 1


async def test_later_failure_keeps_retrieval_diagnostics_but_not_delivered_partial_count():
    cfg = scenario(failure_policy="allow_partial")
    cfg.context.enabled = cfg.generation.enabled = True
    cfg.generation.model = "test"
    generator = SimpleNamespace(generate=AsyncMock(side_effect=StageTimeout("generation")))
    df = await rows(cfg, registry(store(), store(error=SourceUnavailable())), generator=generator)
    assert set(df.retrieval_status) == {"partial"}
    assert set(df.outcome) == {"timeout"}
    assert set(df["doc_hit@1"]) == {0.0}
    assert summarize_repeat(cfg, df)["partial_count"] == 0


async def test_programming_and_payload_errors_are_not_swallowed():
    cfg = scenario(failure_policy="allow_partial")
    with pytest.raises(ExceptionGroup) as exc:
        await rows(cfg, registry(store(error=ValueError("broken code")), store(error=SourceUnavailable())))
    assert exc.value.subgroup(ValueError)
    bad = candidates()
    bad[0].payload["doc_id"] = ""
    with pytest.raises(ValueError, match="nonempty"):
        await rows(cfg, registry(store(found=bad), store(error=SourceUnavailable())))


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("status", [400, 401, 404, 429, 500, 502, 503, 504])
async def test_http_failure_allowlist(monkeypatch, wrapped, status):
    original = UnexpectedResponse(status, "test", b"test response", httpx.Headers())
    error = ResponseHandlingException(original) if wrapped else original
    qdrant = AsyncQdrantStore("http://localhost:6333", "test")
    monkeypatch.setattr(qdrant.client, "query_points", AsyncMock(side_effect=error))
    try:
        expected = SourceUnavailable if status in (502, 503, 504) else type(error)
        with pytest.raises(expected):
            await qdrant.search([1.0], 1)
    finally:
        await qdrant.close()


@pytest.mark.parametrize("ext", ["csv", "parquet"])
async def test_diagnostics_survive_artifacts_and_registry_projection(tmp_path, ext):
    df = await rows(scenario(failure_policy="allow_partial"), registry(store(), store(error=SourceUnavailable())))
    path = tmp_path / f"per_query.repeat0.{ext}"
    getattr(df, f"to_{ext}")(path, index=False)
    page = query_page(tmp_path, QueryFilter(fields=["qid", "retrieval_status", "source_1_status"]))
    assert page.items[0] == {"qid": "q0", "retrieval_status": "partial", "source_1_status": "unavailable"}
    detail = query_row(tmp_path, 0, "q0")
    assert detail["source_0_status"] == "ok" and detail["final_doc_ids"] == ["d1", "d2"]


async def test_diagnostics_belong_to_each_concurrent_query():
    cfg = scenario(failure_policy="allow_partial")
    cfg.run.load.concurrency = 4

    async def alternating(vector, **kwargs):
        await anyio.sleep(0)
        if vector[0] % 2:
            raise SourceUnavailable()
        return candidates()

    df = await rows(cfg, registry(store(), SimpleNamespace(search=alternating)), items=_items(8))
    assert list(df.retrieval_status) == ["partial", "complete"] * 4
    assert list(df.source_1_status) == ["unavailable", "ok"] * 4


async def test_direct_call_diagnostics_and_complete_empty_results():
    diagnostic = RetrievalDiagnostics()
    assert await engine.retrieve_items(scenario(), registry(store(found=[])), "q", [1.0], diagnostics=diagnostic) == []
    assert diagnostic.status == "complete" and diagnostic.sources == {0: "ok", 1: "ok"}
