from __future__ import annotations

import hashlib
import json
from typing import Any

import anyio
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from ragglet.core.errors import ServiceUnavailable

_UNLINK_BATCH = 1000


async def redis_version(endpoint: str, timeout: float) -> str:
    """Версия сервера по INFO server; ServiceUnavailable — не ответил за timeout."""
    try:
        client = redis.from_url(endpoint, socket_connect_timeout=timeout, socket_timeout=timeout)
        try:
            with anyio.fail_after(timeout):
                server = await client.info("server")
        finally:
            await client.aclose()
    except (RedisConnectionError, RedisTimeoutError, TimeoutError, OSError, ValueError) as e:
        raise ServiceUnavailable(endpoint, f"{type(e).__name__}: {e}") from e
    return str(server["redis_version"])


class RedisEmbeddingCache:
    def __init__(
        self,
        endpoint: str,
        ttl_seconds: int,
        model_name: str,
        revision: str | None,
        normalize: bool,
        *,
        namespace: str,
    ):
        self.redis = redis.from_url(endpoint, encoding="utf-8", decode_responses=True)
        self.ttl = int(ttl_seconds)
        self.ns = f"emb:{model_name}@{revision}:{int(normalize)}:{namespace}"
        self._written: set[str] = set()

    def _key(self, text: str) -> str:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"{self.ns}:{h}"

    async def get(self, text: str) -> list[float] | None:
        val = await self.redis.get(self._key(text))
        if not val:
            return None
        return json.loads(val)

    async def set(self, text: str, vector: list[float]) -> None:
        key = self._key(text)
        self._written.add(key)  # до записи: прерванный set тоже убираем
        await self.redis.set(key, json.dumps(vector), ex=self.ttl)

    async def delete_written(self) -> None:
        """Удаляет только ключи, записанные этим экземпляром."""
        keys, self._written = list(self._written), set()
        for i in range(0, len(keys), _UNLINK_BATCH):
            await self.redis.unlink(*keys[i:i + _UNLINK_BATCH])

    async def describe(self) -> dict[str, Any]:
        server = await self.redis.info("server")
        return {"type": "redis", "version": server["redis_version"]}

    async def close(self) -> None:
        await self.redis.aclose()
