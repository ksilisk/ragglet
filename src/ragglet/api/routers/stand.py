from __future__ import annotations

import os
from typing import Any

import anyio
from fastapi import APIRouter

from ragglet.api import probes
from ragglet.api.deps import Services, Svc
from ragglet.api.schemas import ServiceStatus, StandStatus, TaskItem
from ragglet.core.environment import ENV_FACTS, predicted_compute
from ragglet.jobs.protocol import TaskState

router = APIRouter(tags=["stand"])

DEFAULT_KEY_ENV = "OPENAI_API_KEY"


async def _compute(svc: Services) -> dict[str, Any]:
    # импорт torch — секунды: один раз на процесс сервера, в потоке
    if svc.compute is None:
        try:
            svc.compute = await anyio.to_thread.run_sync(predicted_compute)
        except Exception as e:  # стенд отвечает и без torch
            return {"error": f"{type(e).__name__}: {e}"}
    return svc.compute


def _key_env_names(svc: Services) -> list[str]:
    names = {DEFAULT_KEY_ENV}
    names |= {entry.config.generation.api_key_env for entry in svc.scenarios.list() if entry.config is not None}
    return sorted(names)


@router.get("/stand")
async def stand(svc: Svc) -> StandStatus:
    """Сервисы, вычисления, код и очередь — живьём; недоступный сервис не роняет ответ."""
    settings = svc.settings
    timeout = settings.probe_timeout_s
    qdrant: list[ServiceStatus] = []
    redis: ServiceStatus | None = None
    compute: dict[str, Any] = {}

    async def probe_qdrant() -> None:
        qdrant.extend(status for status, _ in await probes.qdrant_all(settings.qdrant_endpoints, timeout))

    async def probe_redis() -> None:
        nonlocal redis
        if settings.redis_url:
            redis = await probes.redis(settings.redis_url, timeout)

    async def probe_compute() -> None:
        compute.update(await _compute(svc))

    async with anyio.create_task_group() as tg:  # параллельно: каждый со своим таймаутом
        for probe in (probe_qdrant, probe_redis, probe_compute):
            tg.start_soon(probe)
    names = await anyio.to_thread.run_sync(_key_env_names, svc)
    queue = await svc.jobs.queue()
    running = [status for status in queue if status.state is TaskState.RUNNING]
    return StandStatus(
        qdrant=qdrant,
        redis=redis,
        compute=compute,
        system=ENV_FACTS["system"](),
        commit=ENV_FACTS["commit"](),
        python=ENV_FACTS["python"](),
        env={name: bool(os.environ.get(name)) for name in names},
        current=TaskItem.build(running[0]) if running else None,
        queued=sum(1 for status in queue if status.state is TaskState.QUEUED),
    )

