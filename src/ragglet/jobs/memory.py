from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from ragglet.jobs.protocol import (
    TERMINAL_STATES,
    LogChunk,
    ProgressInfo,
    TaskFailure,
    TaskFinished,
    TaskRef,
    TaskSpec,
    TaskState,
    TaskStatus,
)
from ragglet.registry.protocol import Registry, TaskChange


@dataclass
class _Task:
    spec: TaskSpec
    status: TaskStatus
    log: str = ""
    cancel_requested: bool = False


class InMemoryJobManager:
    """
    Заглушка менеджера задач до ragglet-29: ничего не выполняет. submit ставит задачу в
    очередь, а переходы делают start / report / finish — их зовут тесты, и они обновляют
    реестр так же, как будет обновлять настоящий менеджер.
    """

    def __init__(self, registry: Registry, now: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self._registry = registry
        self._now = now
        self._tasks: dict[TaskRef, _Task] = {}

    async def submit(self, spec: TaskSpec) -> TaskStatus:
        status = TaskStatus(
            ref=spec.ref, name=spec.name, state=TaskState.QUEUED, position=None,
            enqueued_at=self._now(), started_at=None, finished_at=None, progress=None, failure=None,
        )
        self._tasks[spec.ref] = _Task(spec=spec, status=status)
        return self._view(spec.ref)

    async def status(self, ref: TaskRef) -> TaskStatus | None:
        return self._view(ref) if ref in self._tasks else None

    async def cancel(self, ref: TaskRef) -> TaskStatus:
        task = self._tasks[ref]
        if task.status.state in TERMINAL_STATES:
            raise TaskFinished(f"{ref.kind} {ref.id} is already {task.status.state}")
        if task.status.state is TaskState.QUEUED:
            await self.finish(ref, TaskState.CANCELLED)
        else:
            task.cancel_requested = True  # настоящий менеджер прерывает процесс
        return self._view(ref)

    async def log(self, ref: TaskRef, offset: int = 0, limit: int = 65536) -> LogChunk:
        task = self._tasks.get(ref)
        text = task.log if task else ""
        chunk = text[offset:offset + limit]
        end = offset + len(chunk)
        finished = task is None or task.status.state in TERMINAL_STATES
        return LogChunk(text=chunk, next_offset=end, complete=finished and end >= len(text))

    async def queue(self) -> list[TaskStatus]:
        running = [t for t in self._tasks.values() if t.status.state is TaskState.RUNNING]
        queued = [t for t in self._tasks.values() if t.status.state is TaskState.QUEUED]
        return [self._view(t.spec.ref) for t in (*running, *queued)]

    # --- переходы, которые в 29 делает воркер

    async def start(self, ref: TaskRef) -> None:
        task = self._tasks[ref]
        task.status = replace(task.status, state=TaskState.RUNNING, started_at=self._now())
        await self._registry.update(ref, TaskChange(state=TaskState.RUNNING, started_at=task.status.started_at))

    def report(self, ref: TaskRef, stage: str, done: int, total: int) -> None:
        task = self._tasks[ref]
        now = self._now()
        first_at = task.status.progress.first_at if task.status.progress else now
        task.status = replace(task.status, progress=ProgressInfo(stage, done, total, first_at, now))

    def append_log(self, ref: TaskRef, text: str) -> None:
        self._tasks[ref].log += text

    async def finish(self, ref: TaskRef, state: TaskState, failure: TaskFailure | None = None,
                     result: dict | None = None) -> None:
        task = self._tasks[ref]
        task.status = replace(task.status, state=state, finished_at=self._now(), failure=failure)
        await self._registry.update(ref, TaskChange(
            state=state, finished_at=task.status.finished_at, failure=failure,
            final_progress=task.status.progress, result=result,
        ))

    def _view(self, ref: TaskRef) -> TaskStatus:
        status = self._tasks[ref].status
        if status.state is not TaskState.QUEUED:
            return status
        waiting = [t.spec.ref for t in self._tasks.values() if t.status.state is TaskState.QUEUED]
        return replace(status, position=waiting.index(ref) + 1)
