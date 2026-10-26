from __future__ import annotations

import threading
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import anyio

if TYPE_CHECKING:
    from ragglet.bench.run_scenario import RunResult
    from ragglet.config.index_profile import IndexProfileConfig
    from ragglet.config.scenario import ScenarioConfig
    from ragglet.indexing.build_index import BuildResult

# проверки по одной: две первые проверки одного hf-датасета не качают его одновременно
_checks = threading.Lock()


def _serial[T](fn: Callable[[], T]) -> T:
    with _checks:
        return fn()


async def check_run(cfg: ScenarioConfig) -> RunResult:
    """
    run_scenario(dry_run=True) в потоке со своим event loop: preflight синхронно читает и
    хэширует датасет, а для hf ещё и скачивает его — в loop сервера это остановило бы все запросы.
    """
    from ragglet.bench.run_scenario import run_scenario  # тянет torch: только при первой проверке

    return await anyio.to_thread.run_sync(_serial, partial(anyio.run, partial(run_scenario, cfg, dry_run=True)))


async def check_build(cfg: IndexProfileConfig) -> BuildResult:
    """build_index(dry_run=True): синхронный и блокирующий, поэтому тоже в потоке."""
    from ragglet.indexing.build_index import build_index

    return await anyio.to_thread.run_sync(_serial, partial(build_index, cfg, dry_run=True))
