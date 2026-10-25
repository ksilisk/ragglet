from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_load import _RunStore, _scenario, runtime  # noqa: F401  # runtime — фикстура

from ragglet.bench import run_scenario as run_module
from ragglet.cache import embedding_redis

pytestmark = pytest.mark.usefixtures("runtime")

PREFIX = "emb:all-MiniLM-L6-v2@rev:1:"  # модель по умолчанию, ревизия из фейкового preflight
REDIS = {"type": "redis", "endpoint": "redis://r:6379/15"}


class _Redis:
    """Redis в памяти; data переживает прогоны, как настоящий."""

    def __init__(self, data: dict[str, str]):
        self.data = data
        self.peak = len(data)

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key] = value
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


def _cache_scenario(tmp_path, repeats: int, **cache):
    return _scenario(
        run={"repeats": repeats, "shuffle": False, "artifacts_dir": str(tmp_path)},
        cache={"enabled": True, **cache},
    )


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
