from __future__ import annotations

import threading
import time

import anyio
import pytest
import sentence_transformers

from ragglet.config.scenario import ScenarioConfig
from ragglet.core.errors import StageTimeout
from ragglet.core.lanes import Lane
from ragglet.rerank import cross_encoder
from ragglet.rerank.engine import build_reranker, rerank_items
from ragglet.retrieval.merge import RetrievedItem
from ragglet.stores.bm25_store import KeywordBm25Store


class _Probe:
    """Синхронная работа, которая считает, сколько её копий идёт одновременно."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.active = self.peak = self.finished = 0
        self._lock = threading.Lock()

    def __call__(self, *args):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.seconds)
        with self._lock:
            self.active -= 1
            self.finished += 1
        return args


async def test_lane_runs_one_call_at_a_time_without_blocking_event_loop():
    lane, probe = Lane(), _Probe(0.05)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await anyio.sleep(0.01)
            ticks += 1

    async with anyio.create_task_group() as tg:
        tg.start_soon(ticker)
        async with anyio.create_task_group() as calls:
            for i in range(3):
                calls.start_soon(lane.run, probe, i)
        tg.cancel_scope.cancel()

    assert probe.peak == 1
    assert probe.finished == 3
    assert ticks >= 8  # ~150 мс работы в потоке, event loop всё это время свободен


async def test_lane_timeout_fires_after_the_call_and_leaves_nothing_running():
    lane, probe = Lane(), _Probe(0.2)
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        with anyio.fail_after(0.05):
            await lane.run(probe)
    assert time.perf_counter() - started >= 0.2
    assert probe.finished == 1 and probe.active == 0


def _items(n: int) -> list[RetrievedItem]:
    return [
        RetrievedItem(point_id=str(i), chunk_id=f"d{i}#0", doc_id=f"d{i}", score_raw=0.0, source="vec",
                      payload={"text": f"t{i}"})
        for i in range(n)
    ]


async def test_rerank_timeout_waits_for_the_model_instead_of_abandoning_it(monkeypatch):
    probe = _Probe(0.2)

    class _CrossEncoder:
        def __init__(self, model_name, revision=None):
            pass

        def predict(self, pairs):
            probe()
            return [0.0] * len(pairs)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _CrossEncoder)
    monkeypatch.setattr(cross_encoder, "cached_model_revision", lambda repo_id, revision: "r")
    cfg = ScenarioConfig.model_validate({
        "name": "s",
        "dataset": {"id": "datasets/demo"},
        "storage": {"backend": {"type": "qdrant", "endpoint": "http://q:6333", "collection": "c"}},
        "timeouts": {"rerank_ms": 50},
        "rerank": {"enabled": True, "model": "ce", "input_top_k": 30, "output_top_n": 10},
    })

    started = time.perf_counter()
    with pytest.raises(StageTimeout) as exc:
        await rerank_items(cfg, build_reranker(cfg), "q", _items(3))
    assert exc.value.stage == "rerank"
    assert time.perf_counter() - started >= 0.2
    assert probe.finished == 1 and probe.active == 0


async def test_bm25_search_is_async_and_one_at_a_time():
    store = KeywordBm25Store([{"chunk_id": "d1#0", "doc_id": "d1", "text": "vitamin d"},
                              {"chunk_id": "d2#0", "doc_id": "d2", "text": "breast cancer"}])
    probe = _Probe(0.03)
    top_k = store.index.top_k
    store.index.top_k = lambda query, k: (probe(), top_k(query, k))[1]

    results: list = []

    async def search(q):
        results.append(await store.search(q, top_k=5))

    async with anyio.create_task_group() as tg:
        for q in ("vitamin", "cancer", "vitamin d"):
            tg.start_soon(search, q)

    assert probe.peak == 1
    assert sorted(c[0].id for c in results) == ["d1#0", "d1#0", "d2#0"]
