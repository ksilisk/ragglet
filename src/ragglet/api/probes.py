from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import anyio

from ragglet.api.schemas import ServiceStatus
from ragglet.cache.embedding_redis import redis_version
from ragglet.core.errors import ServiceUnavailable
from ragglet.stores.qdrant_store_async import (
    AsyncQdrantStore,
    CollectionInfo,
    ServerCollections,
    list_collections,
)


def normalize_endpoint(url: str) -> str:
    """Схема и хост в нижнем регистре, без / в конце; DNS не разрешаем: localhost ≠ 127.0.0.1."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


async def gather[T](calls: list[Callable[[], Awaitable[T]]]) -> list[T]:
    results: list[Any] = [None] * len(calls)

    async def run(i: int, call: Callable[[], Awaitable[T]]) -> None:
        results[i] = await call()

    async with anyio.create_task_group() as tg:
        for i, call in enumerate(calls):
            tg.start_soon(run, i, call)
    return results


def _reason(e: Exception, timeout: float) -> str:
    if isinstance(e, ServiceUnavailable):
        return e.reason
    return f"TimeoutError: no answer in {timeout:g} s"


async def qdrant(endpoint: str, timeout: float) -> tuple[ServiceStatus, ServerCollections | None]:
    try:
        with anyio.fail_after(timeout):
            server = await list_collections(endpoint)
    except (ServiceUnavailable, TimeoutError) as e:
        return ServiceStatus(url=endpoint, available=False, error=_reason(e, timeout)), None
    status = ServiceStatus(url=endpoint, available=True, version=server.version, collections=len(server.collections))
    return status, server


async def qdrant_all(endpoints: list[str], timeout: float) -> list[tuple[ServiceStatus, ServerCollections | None]]:
    return await gather([partial(qdrant, endpoint, timeout) for endpoint in endpoints])


async def redis(url: str, timeout: float) -> ServiceStatus:
    try:
        version = await redis_version(url, timeout)
    except ServiceUnavailable as e:
        return ServiceStatus(url=url, available=False, error=e.reason)
    return ServiceStatus(url=url, available=True, version=version)


@dataclass(frozen=True)
class TargetState:
    reachable: bool
    error: str | None
    info: CollectionInfo | None  # None — коллекции нет или endpoint недоступен


async def target(endpoint: str, collection: str, timeout: float) -> TargetState:
    store = AsyncQdrantStore(endpoint, collection)
    try:
        with anyio.fail_after(timeout):
            info = await store.collection_info()
    except (ServiceUnavailable, TimeoutError) as e:
        return TargetState(reachable=False, error=_reason(e, timeout), info=None)
    finally:
        await store.close()
    return TargetState(reachable=True, error=None, info=info)
