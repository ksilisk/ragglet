from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from ragglet.core.issues import Issue


class TaskKind(StrEnum):
    RUN = "run"
    BUILD = "build"


class TaskState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED})


@dataclass(frozen=True)
class TaskRef:
    kind: TaskKind
    id: str  # id прогона или сборки из реестра: своих id у задач нет


@dataclass(frozen=True)
class TaskSpec:
    ref: TaskRef
    name: str  # сценарий или профиль: для очереди и шапки
    config: dict[str, Any]  # проверенный конфиг; воркер валидирует его заново
    run_dir: Path | None = None  # каталог задачи: прогона или сборки; его создаёт реестр при постановке


@dataclass(frozen=True)
class ProgressInfo:
    stage: str  # queries | corpus
    done: int
    total: int
    first_at: datetime  # первое событие: от него UI считает «осталось ≈»
    updated_at: datetime


@dataclass(frozen=True)
class TaskFailure:
    type: str  # класс исключения
    message: str
    qid: str | None = None  # из __notes__ ядра
    title: str | None = None  # у ValidationError второго preflight при старте
    hint: str | None = None
    issues: tuple[Issue, ...] = ()


@dataclass(frozen=True)
class TaskStatus:
    ref: TaskRef
    name: str
    state: TaskState
    position: int | None  # в очереди: 1 — следующая; иначе None
    enqueued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    progress: ProgressInfo | None
    failure: TaskFailure | None


@dataclass(frozen=True)
class LogChunk:
    text: str
    next_offset: int
    complete: bool  # задача завершена и лог дочитан


class TaskFinished(Exception):
    """Отмена задачи, которая уже завершилась."""


class JobManager(Protocol):
    """
    Выполняет прогоны и сборки по одному на стенд (ragglet-29). При каждом переходе
    состояния сам зовёт Registry.update; прогресс и лог держит у себя.
    """

    async def submit(self, spec: TaskSpec) -> TaskStatus: ...

    async def status(self, ref: TaskRef) -> TaskStatus | None:
        """None — менеджер задачу не знает (например, она завершилась до рестарта сервера)."""
        ...

    async def cancel(self, ref: TaskRef) -> TaskStatus:
        """queued — убрать; running — прервать процесс; завершённая — TaskFinished."""
        ...

    async def log(self, ref: TaskRef, offset: int = 0, limit: int = 65536) -> LogChunk: ...

    async def queue(self) -> list[TaskStatus]:
        """Текущая задача, затем ожидающие по порядку."""
        ...
