from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from ragglet.bench import run_scenario as run_module
from ragglet.bench.dataset import QueryItem
from ragglet.bench.load import LOAD_MODELS
from ragglet.bench.query_loop import run_queries_once
from ragglet.cache.manager import CacheManager
from ragglet.config.scenario import ScenarioConfig
from ragglet.core.lanes import Lane
from ragglet.generation.base import GenerationResult
from ragglet.generation.noop import NoopGenerator
from ragglet.rerank.noop import NoopReranker
from ragglet.retrieval.backends import BackendRegistry
from ragglet.stores.qdrant_store_async import Candidate

TIMING = [
    "t_start_ms", "lat_embed_ms", "lat_retrieve_ms", "lat_rerank_ms", "lat_context_ms",
    "lat_generation_ms", "lat_total_ms",
]


def _scenario(load: dict | None = None, tmp_path: Path | None = None, **overrides) -> ScenarioConfig:
    data = {
        "name": "load",
        "dataset": {"id": "datasets/demo"},
        "storage": {"backend": {"type": "qdrant", "endpoint": "http://q:6333", "collection": "c"}},
        "run": {"load": load or {}, "shuffle": False, "artifacts_dir": str(tmp_path or "results/runs")},
        "metrics": {"quality": {"k": [1]}},
        **overrides,
    }
    return ScenarioConfig.model_validate(data)


def _items(n: int) -> list[QueryItem]:
    # длина текста запроса = номер документа, который вернёт фейковый store
    return [QueryItem(qid=f"q{i}", query="x" * (i + 1), qrels={f"d{i + 1}": 1}) for i in range(n)]


class _Embedder:
    def __init__(self, fail_on: str | None = None, delay: float = 0.0):
        self.fail_on = fail_on
        self.delay = delay
        self.active = self.peak = 0
        self._count = threading.Lock()
        self._lane = Lane()

    def embed_batch(self, texts):
        with self._count:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            if self.fail_on in texts:
                raise ValueError("embedder exploded")
            return [[float(len(t))] for t in texts]
        finally:
            with self._count:
                self.active -= 1

    async def embed_query(self, text):
        return (await self._lane.run(self.embed_batch, [text]))[0]


class _Store:
    def __init__(self, delay: float = 0.01):
        self.delay = delay
        self.in_flight = self.peak = self.calls = 0
        self.closed = False

    async def search(self, vector, top_k, params=None):
        self.calls += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await anyio.sleep(self.delay)
            doc = f"d{int(vector[0])}"
            return [Candidate(id=doc, score_raw=1.0, payload={"chunk_id": f"{doc}#0", "doc_id": doc, "text": "t"})]
        finally:
            self.in_flight -= 1


async def _run(cfg, items, store=None, embedder=None, generator=None):
    store = store or _Store()
    return await run_queries_once(
        cfg, items, BackendRegistry(vector=store), embedder or _Embedder(), CacheManager(cfg),
        NoopReranker(), generator or NoopGenerator(),
    )


class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _TestLoad:
    Params = _NoParams

    def __init__(self, params, seed):
        pass


class _Reverse(_TestLoad):
    async def drive(self, n, launch):
        for i in reversed(range(n)):
            await launch(i)


class _Twice(_TestLoad):
    async def drive(self, n, launch):
        await launch(0)
        await launch(0)


class _Skips(_TestLoad):
    async def drive(self, n, launch):
        for i in range(0, n, 2):
            await launch(i)


class _Late(_TestLoad):
    async def drive(self, n, launch):
        for i in range(n):
            await launch(i, anyio.current_time() - 0.2)


@pytest.fixture
def test_load(monkeypatch):
    for name, model in {"reverse": _Reverse, "twice": _Twice, "skips": _Skips, "late": _Late}.items():
        monkeypatch.setitem(LOAD_MODELS, name, model)


async def test_concurrency_changes_only_latency_not_rows():
    items = _items(8)
    one = await _run(_scenario({"concurrency": 1}), items)
    four = await _run(_scenario({"concurrency": 4}), items)

    assert list(one["qid"]) == [it.qid for it in items]
    assert one.drop(columns=TIMING).equals(four.drop(columns=TIMING))
    assert (one["doc_hit@1"] == 1.0).all()


async def test_in_flight_reaches_but_never_exceeds_concurrency():
    store = _Store()
    await _run(_scenario({"concurrency": 3}), _items(10), store=store)
    assert store.peak == 3

    store = _Store()
    await _run(_scenario({"concurrency": 8}), _items(2), store=store)
    assert store.peak == 2  # воркеров не больше, чем запросов


async def test_any_load_model_gets_rows_in_list_order(test_load):
    store = _Store()
    df = await _run(_scenario({"type": "reverse"}), _items(5), store=store)
    assert list(df["qid"]) == ["q0", "q1", "q2", "q3", "q4"]
    assert list(df["final_doc_ids"]) == [["d1"], ["d2"], ["d3"], ["d4"], ["d5"]]
    assert store.peak == 1


async def test_broken_load_models_fail_loudly(test_load):
    with pytest.raises(RuntimeError, match=r"launched query #0 twice"):
        await _run(_scenario({"type": "twice"}), _items(3))
    with pytest.raises(RuntimeError, match=r"without launching 2 queries, e.g. \['q1', 'q3'\]"):
        await _run(_scenario({"type": "skips"}), _items(5))


async def test_query_clock_starts_at_planned_moment(test_load):
    df = await _run(_scenario({"type": "late"}), _items(2))
    assert (df["lat_total_ms"] >= 200).all()
    assert (df["outcome"] == "ok").all()

    df = await _run(_scenario({"type": "late"}, timeouts={"total_ms": 100}), _items(2))
    assert (df["outcome"] == "timeout").all()
    assert set(df["timeout_stage"]) == {"total"}


def test_load_config_is_checked_by_the_model():
    assert _scenario().run.load.model_dump() == {"type": "closed", "concurrency": 1}
    with pytest.raises(ValidationError, match=r"unknown load model 'open'; available: \['closed'\]"):
        _scenario({"type": "open"})
    with pytest.raises(ValidationError, match=r"run\.load\.concurency"):
        _scenario({"concurency": 4})
    with pytest.raises(ValidationError, match=r"run\.load\.concurrency"):
        _scenario({"concurrency": 0})


async def test_generation_runs_concurrently_too():
    class _Llm:
        in_flight = peak = 0

        async def generate(self, request):
            _Llm.in_flight += 1
            _Llm.peak = max(_Llm.peak, _Llm.in_flight)
            await anyio.sleep(0.02)
            _Llm.in_flight -= 1
            return GenerationResult("ok", 1, 1, 2, None, "fake", request.model)

    cfg = _scenario(
        {"concurrency": 4},
        context={"enabled": True},
        generation={"enabled": True, "provider": "openai_compatible", "model": "m"},
    )
    df = await _run(cfg, _items(8), generator=_Llm())
    assert _Llm.peak == 4
    assert list(df["answer"]) == ["ok"] * 8


async def test_embedding_does_not_block_other_queries():
    embedder = _Embedder(delay=0.05)
    df = await _run(_scenario({"concurrency": 2}), _items(4), store=_Store(delay=0.02), embedder=embedder)

    assert embedder.peak == 1  # одна модель — один вызов за раз
    # пока считается эмбеддинг соседа, ответ store принимается сразу; в event loop было бы ~50 мс
    assert df["lat_retrieve_ms"].max() < 40
    assert df["lat_embed_ms"].max() > 90  # ожидание своей очереди к модели видно в latency


async def test_unexpected_error_stops_run_with_original_exception():
    store = _Store(delay=0.05)
    with pytest.raises(ValueError, match="embedder exploded") as exc:
        await _run(_scenario({"concurrency": 4}), _items(8), store=store, embedder=_Embedder("xxxx"))
    assert exc.value.__notes__ == ["qid=q3"]
    assert store.in_flight == 0
    assert store.calls < 8


class _RunStore(_Store):
    created: list = []

    def __init__(self, endpoint, collection):
        super().__init__(delay=0.02)
        self.created.append(self)

    async def describe(self):
        return {"type": "qdrant", "version": "test"}

    async def close(self):
        self.closed = True


class _RunEmbedder(_Embedder):
    fail_on: str | None = None

    def __init__(self, model_name, normalize=True, revision=None):
        super().__init__(self.fail_on)
        self.revision = revision

    def compute_info(self):
        return {"device": "cpu", "accelerator": None, "torch_threads": 1}


@pytest.fixture
def runtime(monkeypatch):
    items = _items(20)

    async def preflight(cfg, stores):
        return SimpleNamespace(
            dataset=SimpleNamespace(items=items),
            index={"embedding_revision": "rev"},
            warnings=[],
            manifest=lambda: {"dataset": {}, "index": {}},
        )

    _RunStore.created = []
    _RunEmbedder.fail_on = None
    monkeypatch.setattr(run_module, "AsyncQdrantStore", _RunStore)
    monkeypatch.setattr(run_module, "SentenceTransformerEmbedder", _RunEmbedder)
    monkeypatch.setattr(run_module, "preflight", preflight)
    return SimpleNamespace(stores=_RunStore.created, embedder=_RunEmbedder)


def _summaries(tmp_path: Path) -> list[Path]:
    return list(tmp_path.rglob("summary*"))


async def test_failed_run_closes_stores_and_writes_no_summary(tmp_path, runtime):
    runtime.embedder.fail_on = "x" * 6
    with pytest.raises(ValueError) as exc:
        await run_module.run_scenario(_scenario({"concurrency": 4}, tmp_path))
    assert exc.value.__notes__ == ["qid=q5"]
    assert [s.closed for s in runtime.stores] == [True]
    assert list(tmp_path.rglob("manifest.json"))
    assert not _summaries(tmp_path)


async def test_cancelled_run_closes_stores_and_writes_no_summary(tmp_path, runtime):
    with anyio.move_on_after(0.05) as scope:
        await run_module.run_scenario(_scenario({"concurrency": 2}, tmp_path))
    assert scope.cancelled_caught
    assert [s.closed for s in runtime.stores] == [True]
    assert runtime.stores[0].in_flight == 0
    assert not _summaries(tmp_path)


async def test_timeout_keeps_finished_stages_and_is_a_miss():
    df = await _run(_scenario(timeouts={"total_ms": 50}), _items(2), store=_Store(delay=1.0))

    assert list(df["outcome"]) == ["timeout", "timeout"]
    assert df["lat_embed_ms"].notna().all()  # эмбеддинг успел, это видно по строке
    assert df["lat_retrieve_ms"].isna().all() and df["lat_rerank_ms"].isna().all()
    assert (df["doc_hit@1"] == 0).all() and df["final_ids"].map(len).eq(0).all()


async def test_query_without_answer_is_a_miss_at_any_stage():
    class _Llm:
        def __init__(self, behaviour):
            self.behaviour = behaviour

        async def generate(self, request):
            if self.behaviour == "slow":
                await anyio.sleep(1)
            if self.behaviour == "broken":
                raise RuntimeError("llm 500")
            return GenerationResult("ok", 1, 1, 2, None, "fake", request.model)

    cfg = _scenario(
        context={"enabled": True},
        generation={"enabled": True, "provider": "openai_compatible", "model": "m"},
        timeouts={"llm_ms": 20},
    )
    ok = await _run(cfg, _items(2), generator=_Llm("ok"))
    slow = await _run(cfg, _items(2), generator=_Llm("slow"))
    broken = await _run(cfg, _items(2), generator=_Llm("broken"))

    assert (ok["outcome"] == "ok").all() and (ok["doc_hit@1"] == 1).all()
    assert ok["lat_generation_ms"].notna().all()
    assert list(slow["timeout_stage"]) == ["generation", "generation"]
    assert list(broken["generation_error"]) == ["llm 500", "llm 500"]
    for df, outcome in ((slow, "timeout"), (broken, "error")):
        assert list(df["outcome"]) == [outcome, outcome]
        assert list(df["retrieved_doc_ids"]) == list(ok["final_doc_ids"])  # поиск нашёл нужное
        assert (df["doc_hit@1"] == 0).all() and df["final_ids"].map(len).eq(0).all()
        assert df["lat_retrieve_ms"].notna().all() and df["lat_generation_ms"].isna().all()


async def test_disabled_stage_has_no_latency():
    df = await _run(_scenario(), _items(2))

    assert df[["lat_rerank_ms", "lat_context_ms", "lat_generation_ms"]].isna().all().all()
