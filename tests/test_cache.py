from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError
from test_load import (  # noqa: F401  # runtime — фикстура
    _Embedder,
    _example_rows,
    _items,
    _RunStore,
    _scenario,
    _Store,
    runtime,
)

from ragglet.bench import run_scenario as run_module
from ragglet.bench.dataset import QueryItem
from ragglet.bench.query_loop import run_queries_once
from ragglet.bench.summarize import summarize_repeat
from ragglet.cache import embedding_redis
from ragglet.cache.embedding_redis import RedisEmbeddingCache
from ragglet.cache.manager import CacheManager
from ragglet.generation.noop import NoopGenerator
from ragglet.rerank.noop import NoopReranker
from ragglet.retrieval.backends import BackendRegistry

pytestmark = pytest.mark.usefixtures("runtime")

PREFIX = "emb:all-MiniLM-L6-v2@rev:1:"  # модель по умолчанию, ревизия из фейкового preflight
REDIS = {"type": "redis", "endpoint": "redis://r:6379/15"}


class _Redis:
    """Redis в памяти; data переживает прогоны, как настоящий."""

    def __init__(self, data: dict[str, str]):
        self.data = data
        self.peak = len(data)
        self.ttls: list[int | None] = []

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key] = value
        self.ttls.append(ex)
        self.peak = max(self.peak, len(self.data))

    async def unlink(self, *keys):
        for key in keys:
            self.data.pop(key, None)

    async def info(self, section):
        return {"redis_version": "7.4.1"}

    async def aclose(self):
        pass


@pytest.fixture
def redis(monkeypatch):
    server = SimpleNamespace(data={}, clients=[])

    def from_url(endpoint, **kwargs):
        server.clients.append(_Redis(server.data))
        return server.clients[-1]

    monkeypatch.setattr(embedding_redis.redis, "from_url", from_url)
    return server


def _cache_scenario(tmp_path, repeats: int, load: dict | None = None, **cache):
    return _scenario(
        run={"repeats": repeats, "shuffle": False, "artifacts_dir": str(tmp_path), "load": load or {}},
        cache={"enabled": True, **cache},
    )


class _CountingEmbedder(_Embedder):
    calls = 0

    async def embed_query(self, text):
        self.calls += 1
        return await super().embed_query(text)


async def _rows(cfg, items, embedder=None, store=None):
    return await run_queries_once(
        cfg, items, BackendRegistry(vector=store or _Store(delay=0)), embedder or _Embedder(),
        CacheManager(cfg, embedding_revision="rev"), NoopReranker(), NoopGenerator(),
    )


async def _fill_namespace(items, namespace: str) -> None:
    """Как заполняющий прогон, который остановили после этих запросов."""
    filler = RedisEmbeddingCache(REDIS["endpoint"], 60, "all-MiniLM-L6-v2", "rev", True, namespace=namespace)
    for it in items:
        await filler.set(it.query, [float(len(it.query))])


def _rates(res, col: str) -> list[float]:
    return [summary[col] for summary in res.repeats]


async def test_every_repeat_starts_with_empty_retrieval_cache(tmp_path):
    res = await run_module.run_scenario(_cache_scenario(tmp_path, 3, retrieval={"size": 100}))

    assert _rates(res, "ret_hit_rate") == [0.0, 0.0, 0.0]  # раньше 0, 1, 1
    assert _RunStore.created[0].calls == 3 * 20  # каждый повтор ходит в store
    assert len(set(_rates(res, "doc_hit@1_mean"))) == 1


async def test_run_deletes_only_its_own_redis_keys(tmp_path, redis):
    foreign = f"{PREFIX}someone:abc"
    redis.data[foreign] = "[1.0]"

    res = await run_module.run_scenario(_cache_scenario(tmp_path, 2, embedding=REDIS))

    assert _rates(res, "emb_hit_rate") == [0.0, 0.0]  # повтор 1 не видит записей повтора 0
    assert redis.clients[0].peak == 1 + 20
    assert redis.data == {foreign: "[1.0]"}


async def test_shared_namespace_lets_the_next_run_start_warm(tmp_path, redis):
    cfg = _cache_scenario(tmp_path, 1, embedding={**REDIS, "namespace": "trial1"})

    cold = await run_module.run_scenario(cfg)
    warm = await run_module.run_scenario(cfg)

    assert _rates(cold, "emb_hit_rate") == [0.0]
    assert _rates(warm, "emb_hit_rate") == [1.0]
    assert len(redis.data) == 20
    assert all(key.startswith(f"{PREFIX}trial1:") for key in redis.data)


def test_shared_namespace_needs_a_single_repeat(tmp_path):
    with pytest.raises(ValidationError, match="use run.repeats=1 and repeat the run instead"):
        _cache_scenario(tmp_path, 2, embedding={**REDIS, "namespace": "trial1"})
    _cache_scenario(tmp_path, 1, embedding={**REDIS, "namespace": "trial1"})
    _scenario(
        run={"repeats": 2},
        cache={"enabled": False, "embedding": {**REDIS, "namespace": "trial1"}},
    )


async def test_hit_rate_is_counted_not_averaged(tmp_path, redis):
    items = _items(6)
    await _fill_namespace(items[:3], "trial1")
    cfg = _cache_scenario(tmp_path, 1, embedding={**REDIS, "namespace": "trial1"})

    df = await _rows(cfg, items)
    s = summarize_repeat(cfg, df)

    assert list(df["emb_cache"]) == ["hit"] * 3 + ["miss"] * 3
    assert (s["emb_hits"], s["emb_misses"], s["emb_hit_rate"]) == (3, 3, 0.5)  # среднее накопленных долей: 0,81


async def test_hit_rate_does_not_depend_on_order_or_concurrency(tmp_path, redis):
    items = _items(6)
    warm = {it.qid for it in items[:3]}
    for namespace, load, order in (
        ("reversed", None, items[3:] + items[:3]),  # среднее накопленных долей: 0,19
        ("concurrent", {"concurrency": 2}, items),
    ):
        await _fill_namespace(items[:3], namespace)
        cfg = _cache_scenario(tmp_path, 1, load, embedding={**REDIS, "namespace": namespace})

        df = await _rows(cfg, order)
        s = summarize_repeat(cfg, df)

        assert list(df["emb_cache"]) == ["hit" if qid in warm else "miss" for qid in df["qid"]]
        assert (s["emb_hits"], s["emb_misses"], s["emb_hit_rate"]) == (3, 3, 0.5)


async def test_retrieval_cache_hit_skips_embedding(tmp_path, redis):
    first = _items(1)[0]
    again = QueryItem(qid="again", query=first.query, qrels=first.qrels)
    cfg = _cache_scenario(tmp_path, 1, retrieval={"size": 10}, embedding=REDIS)
    embedder, store = _CountingEmbedder(), _Store(delay=0)

    df = await _rows(cfg, [first, again], embedder, store)

    assert list(df["ret_cache"]) == ["miss", "hit"]
    assert df["emb_cache"][0] == "miss"
    assert pd.isna(df["emb_cache"][1])  # вектор не понадобился, Redis не спрашивали
    assert embedder.calls == store.calls == 1
    assert df["lat_embed_ms"].isna().tolist() == [False, True]
    assert df["lat_retrieve_ms"].notna().all()
    assert list(df["final_doc_ids"]) == [["d1"], ["d1"]]


def test_total_latency_is_split_by_hit_and_miss(tmp_path):
    cfg = _cache_scenario(tmp_path, 1, embedding=REDIS)
    df = _example_rows(cfg)  # lat_total: 100, 120, 150, 300 (таймаут), 120, 90
    df["emb_cache"] = ["hit", "hit", "miss", "miss", "miss", None]

    s = summarize_repeat(cfg, df)

    assert (s["emb_hits"], s["emb_misses"], s["emb_hit_rate"]) == (2, 3, 0.4)
    assert s["lat_total_emb_hit_p50"] == 110
    assert s["lat_total_emb_miss_p50"] == 150  # таймаут в своём срезе; без него было бы 135


def test_summary_has_cache_fields_only_for_enabled_layers(tmp_path):
    plain = _scenario()
    both = _cache_scenario(tmp_path, 1, retrieval={"size": 10}, embedding=REDIS)
    df = _example_rows(plain)
    df["ret_cache"] = "hit"
    df["emb_cache"] = None  # все ответы из LRU, к Redis не обращались

    off = summarize_repeat(plain, df)
    on = summarize_repeat(both, df)

    assert not [k for k in off if k.startswith(("emb_", "ret_", "lat_total_emb_", "lat_total_ret_"))]
    assert (on["ret_hits"], on["ret_misses"], on["ret_hit_rate"]) == (6, 0, 1.0)
    assert (on["emb_hits"], on["emb_misses"]) == (0, 0)
    assert math.isnan(on["emb_hit_rate"]) and math.isnan(on["lat_total_emb_hit_p50"])
    assert math.isnan(on["lat_total_ret_miss_p50"])


async def test_redis_writes_with_ttl_under_a_key_of_model_revision_and_scope(tmp_path, redis):
    cfg = _cache_scenario(tmp_path, 1, embedding={**REDIS, "ttl_seconds": 60})
    await _rows(cfg, _items(3))
    assert redis.clients[0].ttls == [60, 60, 60]

    def key(**changes) -> str:
        args = {"model_name": "m", "revision": "r1", "normalize": True, "namespace": "ns", **changes}
        return RedisEmbeddingCache("redis://r", 60, **args)._key("query")

    variants = [key(model_name="m2"), key(revision="r2"), key(normalize=False), key(namespace="ns2")]
    assert key() == key()
    assert len({key(), *variants}) == 5


def test_ttl_must_be_positive(tmp_path):
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        _cache_scenario(tmp_path, 1, embedding={**REDIS, "ttl_seconds": 0})
