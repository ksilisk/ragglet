from __future__ import annotations

import json
import hashlib
from typing import Any

import redis.asyncio as redis


class RedisEmbeddingCache:
    def __init__(self, endpoint: str, ttl_seconds: int, model_name: str, normalize: bool):
        self.redis = redis.from_url(endpoint, encoding="utf-8", decode_responses=True)
        self.ttl = int(ttl_seconds)
        self.ns = f"emb:{model_name}:{int(normalize)}"

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
        await self.redis.set(key, json.dumps(vector), ex=self.ttl)

    async def close(self) -> None:
        await self.redis.aclose()