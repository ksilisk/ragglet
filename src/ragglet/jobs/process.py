from __future__ import annotations

import codecs
import logging
import os
import signal
import subprocess
import sys
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import Process, TaskGroup

from ragglet.jobs import files
from ragglet.jobs.protocol import (
    TERMINAL_STATES,
    LogChunk,
    TaskFailure,
    TaskFinished,
    TaskKind,
    TaskRef,
    TaskSpec,
    TaskState,
    TaskStatus,
)
from ragglet.registry.protocol import Registry, TaskChange

log = logging.getLogger("ragglet.jobs")

WORKER = (sys.executable, "-m", "ragglet.jobs.worker")
MESSAGE_LIMIT = 2000
LOG_TAIL_LINES = 5

SHUTDOWN = TaskFailure(type="ServerShutdown", message="the server stopped before the task finished")


@dataclass
class _Task:
    spec: TaskSpec  # run_dir задан всегда: каталог задачи с её файлами
    status: TaskStatus
    cancel: anyio.Event = field(default_factory=anyio.Event)
    progress_key: tuple[int, int] | None = None  # (inode, mtime) прочитанного progress.json

    @property
    def dir(self) -> Path:
        assert self.spec.run_dir is not None
        return self.spec.run_dir


def decide(outcome: files.Outcome | None, *, cancelled: bool, stopping: bool, code: int | None,
           task_dir: Path) -> files.Outcome:
    """
    Итог задачи, когда процесс вышел: result.json воркера плюс то, что знает менеджер — просили
    ли отмену и останавливается ли сервер. Без result.json процесс упал или был убит.
    """
    if outcome is not None and outcome.state is TaskState.COMPLETED:
        return outcome  # успела закончиться
    final = outcome.final_progress if outcome is not None else files.read_progress(task_dir)
    if stopping:
        return files.Outcome(TaskState.FAILED, SHUTDOWN, final)
    if outcome is not None and not (cancelled and outcome.state is TaskState.FAILED):
        return outcome
    if cancelled:  # сбой из-за отмены, SIGKILL после grace или сигнал во время импортов
        return files.Outcome(TaskState.CANCELLED, final_progress=final)
    return files.Outcome(TaskState.FAILED, crash_failure(code, task_dir), final)


def crash_failure(code: int | None, task_dir: Path) -> TaskFailure:
    how = f"code {code}"
    if code is not None and code < 0:
        with suppress(ValueError):
            how += f" ({signal.Signals(-code).name})"
    message = f"worker exited with {how} without {files.RESULT}"
    tail = _log_tail(task_dir / files.LOG)
    if tail:  # причина обычно в последних строках: OOM, segfault, трейсбек до записи итога
        message += "; last log lines:\n"
        message += tail[-(MESSAGE_LIMIT - len(message)):]
    return TaskFailure(type="WorkerCrashed", message=message)


def _log_tail(path: Path, max_bytes: int = 4096) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(max(0, path.stat().st_size - max_bytes))
            data = f.read()
    except FileNotFoundError:
        return ""
    lines = [line for line in data.decode("utf-8", "replace").splitlines() if line.strip()]
    return "\n".join(lines[-LOG_TAIL_LINES:])


def read_log(path: Path, offset: int, limit: int, finished: bool) -> LogChunk:
    """Кусок лога с байтового смещения; режется по границе символа UTF-8, битые байты — U+FFFD."""
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            f.seek(offset)
            data = f.read(limit)
    except FileNotFoundError:
        return LogChunk(text="", next_offset=offset, complete=finished)
    at_end = offset + len(data) >= size
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    text = decoder.decode(data, final=finished and at_end)
    used = len(data) - len(decoder.getstate()[0])  # неполный последний символ — следующему запросу
    if used == 0 and data:
        text, used = data.decode("utf-8", "replace"), len(data)
    next_offset = offset + used
    return LogChunk(text=text, next_offset=next_offset, complete=finished and next_offset >= size)


class ProcessJobManager:
    """
    Задачи стенда по одной, FIFO, каждая в свежем процессе `python -m ragglet.jobs.worker
    <каталог>` (ragglet-29). Общение — файлы в каталоге задачи; при каждом переходе
    состояния — Registry.update, прогресс и лог остаются здесь. Работает внутри `async with`:
    там task group процессов, а выход — остановка сервера, задачи становятся ServerShutdown.
    """

    def __init__(
            self,
            registry: Registry,
            *,
            cancel_grace_s: float = 10.0,
            command: Sequence[str] = WORKER,
            builds_dir: Path = Path("results/builds"),
            now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self._registry = registry
        self._grace = cancel_grace_s
        self._command = list(command)
        self._builds_dir = builds_dir  # до ragglet-30 реестр не создаёт каталог сборки
        self._now = now
        self._tasks: dict[TaskRef, _Task] = {}  # все задачи этого запуска сервера, и завершённые
        self._waiting: deque[TaskRef] = deque()
        self._current: _Task | None = None
        self._stack = AsyncExitStack()
        self._tg: TaskGroup | None = None
        self._stopping = False

    async def __aenter__(self) -> ProcessJobManager:
        self._tg = await self._stack.enter_async_context(anyio.create_task_group())
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self._stopping = True
        while self._waiting:
            await self._finish(self._tasks[self._waiting.popleft()], files.Outcome(TaskState.FAILED, SHUTDOWN))
        if self._current is not None:
            self._current.cancel.set()
        await self._stack.aclose()  # ждёт, пока процесс текущей задачи выйдет
        self._tg = None

    async def submit(self, spec: TaskSpec) -> TaskStatus:
        if self._tg is None or self._stopping:
            raise RuntimeError("ProcessJobManager is not running: enter it with `async with`")
        if spec.run_dir is None:
            spec = replace(spec, run_dir=self._builds_dir / spec.ref.id)
        assert spec.run_dir is not None
        spec.run_dir.mkdir(parents=True, exist_ok=True)  # обычно его уже создал реестр
        files.write_task(spec)
        self._tasks[spec.ref] = _Task(spec=spec, status=TaskStatus(
            ref=spec.ref, name=spec.name, state=TaskState.QUEUED, position=None, enqueued_at=self._now(),
            started_at=None, finished_at=None, progress=None, failure=None,
        ))
        self._waiting.append(spec.ref)
        self._dispatch()
        return self._view(self._tasks[spec.ref])

    async def status(self, ref: TaskRef) -> TaskStatus | None:
        task = self._tasks.get(ref)
        return None if task is None else self._view(task)

    async def cancel(self, ref: TaskRef) -> TaskStatus:
        task = self._tasks[ref]
        if task.status.state in TERMINAL_STATES:
            raise TaskFinished(f"{ref.kind} {ref.id} is already {task.status.state}")
        if task.status.state is TaskState.QUEUED:
            self._waiting.remove(ref)
            await self._finish(task, files.Outcome(TaskState.CANCELLED))
        else:
            task.cancel.set()  # процесс прерывает _execute; ответ — ещё running
        return self._view(task)

    async def log(self, ref: TaskRef, offset: int = 0, limit: int = 65536) -> LogChunk:
        task = self._tasks.get(ref)
        if task is None:  # завершилась до рестарта: её лог отдаёт реестр
            return LogChunk(text="", next_offset=offset, complete=True)
        return read_log(task.dir / files.LOG, offset, limit, task.status.state in TERMINAL_STATES)

    async def queue(self) -> list[TaskStatus]:
        current = [self._current] if self._current is not None else []
        return [self._view(task) for task in (*current, *(self._tasks[ref] for ref in self._waiting))]

    # --- выполнение

    def _dispatch(self) -> None:
        """Следующая задача — только когда процесс предыдущей вышел: замеры не перекрываются."""
        if self._current is not None or self._stopping or not self._waiting or self._tg is None:
            return
        task = self._tasks[self._waiting.popleft()]
        task.status = replace(task.status, state=TaskState.RUNNING, started_at=self._now())
        self._current = task
        self._tg.start_soon(self._execute, task)

    async def _execute(self, task: _Task) -> None:
        ref = task.spec.ref
        await self._update(ref, TaskChange(state=TaskState.RUNNING, started_at=task.status.started_at))
        try:
            process = await self._spawn(task)
        except OSError as e:
            await self._finish(task, files.Outcome(TaskState.FAILED, TaskFailure(type(e).__name__, str(e))))
            return
        async with process, anyio.create_task_group() as tg:
            tg.start_soon(self._stop_on_cancel, task, process)
            code = await process.wait()
            tg.cancel_scope.cancel()
        outcome = decide(files.read_outcome(task.dir), cancelled=task.cancel.is_set(), stopping=self._stopping,
                         code=code, task_dir=task.dir)
        await self._finish(task, outcome)

    async def _spawn(self, task: _Task) -> Process:
        for stale in (files.PROGRESS, files.RESULT):  # от ручного запуска воркера
            (task.dir / stale).unlink(missing_ok=True)
        with open(task.dir / files.LOG, "ab") as log_file:  # у процесса своя копия дескриптора
            return await anyio.open_process(
                [*self._command, str(task.dir)],
                stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file,
                env=os.environ | {"PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1"},
            )

    async def _stop_on_cancel(self, task: _Task, process: Process) -> None:
        await task.cancel.wait()
        with suppress(ProcessLookupError):
            process.terminate()
        await anyio.sleep(self._grace)
        with suppress(ProcessLookupError):
            process.kill()

    async def _finish(self, task: _Task, outcome: files.Outcome) -> None:
        ref = task.spec.ref
        task.status = replace(task.status, state=outcome.state, position=None, finished_at=self._now(),
                              progress=outcome.final_progress, failure=outcome.failure)
        await self._update(ref, TaskChange(
            state=outcome.state, finished_at=task.status.finished_at, failure=outcome.failure,
            final_progress=outcome.final_progress, result=outcome.result if ref.kind is TaskKind.BUILD else None,
        ))
        if self._current is task:
            self._current = None
            self._dispatch()

    async def _update(self, ref: TaskRef, change: TaskChange) -> None:
        """Сбой реестра не останавливает стенд: состояние в памяти верное до рестарта."""
        try:
            await self._registry.update(ref, change)
        except Exception:
            log.exception("registry update failed for %s %s", ref.kind, ref.id)

    def _view(self, task: _Task) -> TaskStatus:
        if task.status.state is TaskState.QUEUED:
            return replace(task.status, position=self._waiting.index(task.spec.ref) + 1)
        if task.status.state is TaskState.RUNNING:
            self._refresh_progress(task)
        return task.status

    def _refresh_progress(self, task: _Task) -> None:
        """progress.json перечитывается, только если воркер его заменил: не чаще его записи."""
        try:
            st = (task.dir / files.PROGRESS).stat()
        except FileNotFoundError:
            return
        key = (st.st_ino, st.st_mtime_ns)  # os.replace даёт новый inode, даже если mtime грубый
        if key != task.progress_key:
            task.progress_key = key
            if (progress := files.read_progress(task.dir)) is not None:
                task.status = replace(task.status, progress=progress)
