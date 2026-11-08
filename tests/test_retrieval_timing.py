from __future__ import annotations

import math
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import httpx
import pytest
from pydantic import ValidationError
from qdrant_client.http.exceptions import ResponseHandlingException

from ragglet.config.scenario import ScenarioConfig
from ragglet.core.errors import SourceNetworkTimeout, StageTimeout
from ragglet.retrieval import engine, timing
from ragglet.retrieval.backends import BackendRegistry
from ragglet.retrieval.sources import active_sources
from ragglet.retrieval.timing import RetrievalTimings
from ragglet.stores.qdrant_store_async import AsyncQdrantStore, Candidate


def scenario(**retrieval):
    return ScenarioConfig.model_validate({
        "name": "timing", "dataset": {"id": "datasets/demo"},
        "storage": {"backend": {"type": "qdrant", "endpoint": "http://q:6333", "collection": "c"}},
        "retrieval": {"strategy": "fanout_merge", "top_k": 2, "sources": [
            {"name": "a", "params": {"backend_ref": "a"}},
            {"name": "b", "params": {"backend_ref": "b"}},
        ], **retrieval},
        "metrics": {"quality": {"k": [1]}},
        "timeouts": {"retrieval_ms": 3000, "total_ms": 5000},
    })


def candidates():
    return [Candidate(id=d, score_raw=s, payload={"chunk_id": f"{d}#0", "doc_id": d})
            for d, s in [("d1", 0.8), ("d2", 0.4)]]


def registry(a, b=None):
    return BackendRegistry(vector=a, vectors={"a": a, "b": b or a})


@pytest.mark.parametrize("method", ["rrf", "weighted_sum", "interleave"])
async def test_boundaries_include_conversion_and_merge_without_changing_ranking(monkeypatch, method):
    clock = [0.0]
    monkeypatch.setattr(timing, "perf_counter", lambda: clock[0])

    async def search(*args, **kwargs):
        clock[0] += 0.012
        return candidates()

    store = SimpleNamespace(search=search)
    cfg = scenario(**{"async": False, "merge": {"method": method}})
    expected = await engine.retrieve_items(cfg, registry(store), "q", [1.0])
    original_convert = engine._to_retrieved
    original_merge = getattr(engine, f"{method}_merge")

    def convert(*args):
        clock[0] += 0.002
        return original_convert(*args)

    def merge(*args, **kwargs):
        clock[0] += 0.003
        return original_merge(*args, **kwargs)

    monkeypatch.setattr(engine, "_to_retrieved", convert)
    monkeypatch.setattr(engine, f"{method}_merge", merge)
    measured = RetrievalTimings()
    assert await engine.retrieve_items(cfg, registry(store), "q", [1.0], timings=measured) == expected
    assert measured.completed == pytest.approx({"source_0": 12, "source_1": 12, "sources": 24, "merge": 11})
    assert measured.network_timeouts == {}


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("concurrency", [1, 4])
async def test_synthetic_delays_and_query_local_timers(parallel, concurrency):
    intervals = {}

    class Store:
        def __init__(self, name, delay):
            self.name, self.delay = name, delay

        async def search(self, vector, **kwargs):
            q = int(vector[0])
            start = time.perf_counter()
            await anyio.sleep(self.delay + q / 1000)
            intervals[q, self.name] = (start, time.perf_counter())
            return candidates()

    cfg = scenario(**{"async": parallel})
    backends = registry(Store("a", 0.020), Store("b", 0.080))
    measurements = [RetrievalTimings() for _ in range(concurrency)]

    async def run(q):
        await engine.retrieve_items(cfg, backends, "same query", [float(q)], timings=measurements[q])

    async with anyio.create_task_group() as tg:
        for q in range(concurrency):
            tg.start_soon(run, q)
    for q, measured in enumerate(measurements):
        a, b = intervals[q, "a"], intervals[q, "b"]
        assert (b[0] < a[1]) is parallel
        for i, (start, end) in enumerate((a, b)):
            assert measured.completed[f"source_{i}"] == pytest.approx((end - start) * 1000, abs=5)
        assert measured.completed["sources"] == pytest.approx((max(a[1], b[1]) - min(a[0], b[0])) * 1000, abs=5)
        assert measured.completed["merge"] >= 0


@pytest.mark.parametrize("sources", [[], [{"name": "custom"}], [{"name": "a"}, {"name": "unused"}]])
async def test_single_active_source_and_empty_results(sources):
    cfg = scenario(strategy="single", sources=sources)
    store = SimpleNamespace(search=AsyncMock(return_value=[]))
    measured = RetrievalTimings()
    assert await engine.retrieve_items(cfg, registry(store), "q", [1.0], timings=measured) == []
    assert len(active_sources(cfg.retrieval)) == 1
    assert active_sources(cfg.retrieval)[0].name == (sources[0]["name"] if sources else "vec")
    assert set(measured.completed) == {"source_0", "sources"}
    assert math.isnan(measured.columns(1)["lat_merge_ms"])
    store.search.assert_awaited_once()


def test_duplicate_names_identify_the_second_field():
    with pytest.raises(ValidationError) as exc:
        scenario(sources=[{"name": "a"}, {"name": "a"}])
    assert exc.value.errors()[0]["ctx"]["error"].path == "sources[1].name"


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("network", [False, True])
async def test_completed_branch_survives_timeout(parallel, network):
    cfg = scenario(**{"async": parallel})
    cfg.timeouts.retrieval_ms = 50
    measured = RetrievalTimings()
    cancelled = []

    async def slow(*args, **kwargs):
        try:
            await anyio.sleep(0.010)
            if network:
                raise SourceNetworkTimeout("read timed out")
            await anyio.sleep_forever()
        finally:
            cancelled.append(True)

    a = SimpleNamespace(search=AsyncMock(return_value=candidates()))
    b = SimpleNamespace(search=slow)
    with pytest.raises(StageTimeout, match="retrieval"):
        await engine.retrieve_items(cfg, registry(a, b), "q", [1.0], timings=measured)
    assert set(measured.completed) == {"source_0"}
    assert set(measured.network_timeouts) == ({1} if network else set())
    assert cancelled == [True]


async def test_sequential_timeout_does_not_start_later_sources():
    cfg = scenario(**{"async": False})
    cfg.timeouts.retrieval_ms = 20
    a = SimpleNamespace(search=AsyncMock(side_effect=SourceNetworkTimeout("timeout")))
    b = SimpleNamespace(search=AsyncMock(return_value=[]))
    measured = RetrievalTimings()
    with pytest.raises(StageTimeout):
        await engine.retrieve_items(cfg, registry(a, b), "q", [1.0], timings=measured)
    b.search.assert_not_awaited()
    assert measured.completed == {}
    assert set(measured.network_timeouts) == {0}


async def test_external_cancellation_leaves_no_successful_measurement():
    async def wait(*args, **kwargs):
        await anyio.sleep_forever()

    measured = RetrievalTimings()
    with anyio.move_on_after(0.020) as scope:
        await engine.retrieve_items(scenario(), registry(SimpleNamespace(search=wait)), "q", [1.0], timings=measured)
        pytest.fail("external cancellation was swallowed")
    assert scope.cancelled_caught
    assert measured.completed == measured.network_timeouts == {}


async def test_mixed_error_group_is_fatal():
    mixed = ExceptionGroup("both", [SourceNetworkTimeout("timeout"), ValueError("bad payload")])
    store = SimpleNamespace(search=AsyncMock(side_effect=mixed))
    with pytest.raises(ExceptionGroup) as exc:
        await engine.retrieve_items(scenario(), registry(store), "q", [1.0])
    assert exc.value.subgroup(ValueError) is not None


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout])
async def test_qdrant_transport_timeout_normalization(monkeypatch, wrapped, error_type):
    cause = error_type("timed out")
    original = ResponseHandlingException(cause) if wrapped else cause
    store = AsyncQdrantStore("http://localhost:6333", "c")
    monkeypatch.setattr(store.client, "query_points", AsyncMock(side_effect=original))
    try:
        with pytest.raises(SourceNetworkTimeout) as exc:
            await store.search([1.0], 1)
        assert exc.value.__cause__ is original
    finally:
        await store.close()


@pytest.mark.parametrize("wrapped", [False, True])
async def test_qdrant_other_transport_errors_remain_fatal(monkeypatch, wrapped):
    cause = httpx.ConnectError("connection refused")
    original = ResponseHandlingException(cause) if wrapped else cause
    store = AsyncQdrantStore("http://localhost:6333", "c")
    monkeypatch.setattr(store.client, "query_points", AsyncMock(side_effect=original))
    try:
        with pytest.raises(type(original)) as exc:
            await store.search([1.0], 1)
        assert exc.value is original
    finally:
        await store.close()
