from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import anyio
import anyio.lowlevel

T = TypeVar("T")


class Lane:
    """
    Очередь к локальному компоненту, который считает на CPU: один вызов за раз, в отдельном
    потоке, чтобы event loop продолжал принимать ответы сети. Начатый вызов не бросается:
    таймаут срабатывает, когда он закончился, и фоновых вычислений не остаётся.
    """

    def __init__(self) -> None:
        self._limiter = anyio.CapacityLimiter(1)

    async def run(self, fn: Callable[..., T], *args: object) -> T:
        result = await anyio.to_thread.run_sync(fn, *args, limiter=self._limiter)
        # экранированный вызов сам не доставляет просроченный дедлайн, checkpoint доставляет
        await anyio.lowlevel.checkpoint()
        return result
