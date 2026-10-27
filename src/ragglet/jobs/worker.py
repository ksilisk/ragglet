"""
Процесс одной задачи: `python -m ragglet.jobs.worker <каталог задачи>`. Читает task.json, вызывает
ядро, пишет progress.json и в конце result.json; stdout и stderr процесса — это task.log. Его
запускает менеджер задач (ragglet-29), а для отладки можно запустить и руками.
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
import time
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Any

import anyio

from ragglet.core.issues import ValidationError
from ragglet.jobs import files
from ragglet.jobs.protocol import ProgressInfo, TaskFailure, TaskKind, TaskSpec, TaskState

# SIGINT — потому что воркер в одной группе процессов с сервером: Ctrl+C в его терминале
# приходит обоим, а сервер сам остановит задачу
CANCEL_SIGNALS = (signal.SIGTERM, signal.SIGINT)
CANCELLED_CODE = 143  # 128 + SIGTERM
MESSAGE_LIMIT = 2000
PARENT_CHECK_S = 2.0
ORPHAN_GRACE_S = 10.0


class Cancelled(Exception):
    """Задачу отменили сигналом: менеджер, Ctrl+C или смерть сервера."""


class ThrottledProgress:
    """
    progress ядра. Ядро зовёт его на каждый запрос, а progress.json пишется не чаще раза в
    `interval`, плюс первое событие стадии и done == total. Не бросает: исключение в callback
    ядро считает сбоем запроса и останавливает прогон.
    """

    def __init__(self, path: Path, interval: float = 1.0, clock: Callable[[], float] = time.time):
        self._path = path
        self._interval = interval
        self._clock = clock
        self._stage: str | None = None
        self._done = self._total = 0
        self._first_at = self._last_at = self._written_at = 0.0

    def __call__(self, stage: str, done: int, total: int) -> None:
        now = self._clock()
        new_stage = stage != self._stage
        if new_stage:
            self._stage, self._first_at = stage, now
        self._done, self._total, self._last_at = done, total, now
        if new_stage or done >= total or now - self._written_at >= self._interval:
            self._written_at = now
            try:
                files.write_json(self._path, files.progress_to_json(self.last))
            except OSError:
                pass  # пропавшая полоса лучше сорванного замера

    @property
    def last(self) -> ProgressInfo | None:
        """Последний вызов, а не последняя запись: из него берётся final_progress."""
        if self._stage is None:
            return None
        return ProgressInfo(self._stage, self._done, self._total,
                            datetime.fromtimestamp(self._first_at, UTC), datetime.fromtimestamp(self._last_at, UTC))


Operation = Callable[[TaskSpec, ThrottledProgress], dict[str, Any] | None]


def _raise_cancelled(signum: int, frame: FrameType | None) -> None:
    raise Cancelled(signal.Signals(signum).name)


async def cancellable[T](fn: Callable[[], Awaitable[T]]) -> T:
    """
    Сигнал отмены отменяет scope: ядро останавливается на ближайшем await тем же путём, что при
    сбое запроса, и finally закрывает клиентов. Синхронный код и начатые вызовы в потоках
    дорабатывают. Без task group: исключение ядра выходит как есть, а не ExceptionGroup.
    """
    loop = asyncio.get_running_loop()
    with anyio.CancelScope() as scope:
        for sig in CANCEL_SIGNALS:
            loop.add_signal_handler(sig, scope.cancel)
        try:
            return await fn()
        finally:
            for sig in CANCEL_SIGNALS:
                loop.remove_signal_handler(sig)
                signal.signal(sig, _raise_cancelled)  # asyncio вернул бы действие по умолчанию
    raise Cancelled("signal")


def run(spec: TaskSpec, progress: ThrottledProgress) -> None:
    from ragglet.bench.run_scenario import run_scenario
    from ragglet.config.scenario import ScenarioConfig

    assert spec.run_dir is not None
    cfg = ScenarioConfig.model_validate(spec.config)
    # ядро пишет артефакты туда же, где лежат файлы задачи
    cfg = cfg.model_copy(update={"run": cfg.run.model_copy(update={"artifacts_dir": str(spec.run_dir.parent)})})
    anyio.run(cancellable, partial(run_scenario, cfg, progress=progress, run_id=spec.run_dir.name))


def build(spec: TaskSpec, progress: ThrottledProgress) -> dict[str, Any]:
    from ragglet.config.index_profile import IndexProfileConfig
    from ragglet.indexing.build_index import build_index

    res = build_index(IndexProfileConfig.model_validate(spec.config), progress=progress)
    return {"documents": res.documents, "index": res.index}


OPERATIONS: Mapping[TaskKind, Operation] = {TaskKind.RUN: run, TaskKind.BUILD: build}


def watch_parent(interval: float = PARENT_CHECK_S, grace: float = ORPHAN_GRACE_S) -> None:
    """
    Сервер умер — воркер переходит к init/launchd, и pid родителя меняется. Тогда задача
    отменяется, как по SIGTERM, а через grace процесс выходит в любом случае: сирота не
    держит стенд. Смена ppid не страдает от переиспользования pid.
    """
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(interval)
        print(f"parent process {parent} is gone: cancelling the task", file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(grace)
        os._exit(CANCELLED_CODE)

    threading.Thread(target=watch, name="ragglet-parent-watch", daemon=True).start()


def failure_of(e: Exception) -> TaskFailure:
    if isinstance(e, ValidationError):
        return TaskFailure(type="ValidationError", message=e.title, title=e.title, hint=e.hint, issues=tuple(e.issues))
    qid = next((note.removeprefix("qid=") for note in getattr(e, "__notes__", ()) if note.startswith("qid=")), None)
    message = str(e)
    if len(message) > MESSAGE_LIMIT:
        message = message[:MESSAGE_LIMIT] + " …"
    return TaskFailure(type=type(e).__name__, message=message, qid=qid)


def _duration(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds < 60 else f"{int(seconds // 60)} min {int(seconds % 60)} s"


def _finish(spec: TaskSpec, progress: ThrottledProgress, started: float, outcome: files.Outcome) -> None:
    signal.pthread_sigmask(signal.SIG_BLOCK, CANCEL_SIGNALS)  # итог пишется целиком
    assert spec.run_dir is not None
    outcome = files.Outcome(outcome.state, outcome.failure, progress.last, outcome.result)
    files.write_outcome(spec.run_dir, outcome)
    summary = str(outcome.state)
    if outcome.failure is not None:
        summary += f": {outcome.failure.type}: {outcome.failure.message}"
    elif outcome.state is TaskState.CANCELLED and (last := progress.last) is not None:
        summary += f" at {last.stage} {last.done} / {last.total}"
    print(f"== {summary} after {_duration(time.monotonic() - started)} ==", flush=True)


def main(argv: Sequence[str] | None = None, operations: Mapping[TaskKind, Operation] = OPERATIONS) -> int:
    for sig in CANCEL_SIGNALS:  # до любой работы: сигнал во время импортов — тоже отмена
        signal.signal(sig, _raise_cancelled)
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m ragglet.jobs.worker <task dir>", file=sys.stderr)
        return 2
    spec = files.read_task(Path(args[0]))
    watch_parent()
    progress = ThrottledProgress(Path(args[0]) / files.PROGRESS)
    started = time.monotonic()
    print(f"== {spec.ref.kind} {spec.ref.id} ({spec.name}) started "
          f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}, pid {os.getpid()} ==", flush=True)
    try:
        result = operations[spec.ref.kind](spec, progress)
    except Cancelled:
        _finish(spec, progress, started, files.Outcome(TaskState.CANCELLED))
        return CANCELLED_CODE
    except ValidationError as e:  # preflight №2 при старте
        print(e, file=sys.stderr, flush=True)
        _finish(spec, progress, started, files.Outcome(TaskState.FAILED, failure_of(e)))
        return 2
    except Exception as e:
        traceback.print_exception(e)
        _finish(spec, progress, started, files.Outcome(TaskState.FAILED, failure_of(e)))
        return 1
    _finish(spec, progress, started, files.Outcome(TaskState.COMPLETED, result=result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
