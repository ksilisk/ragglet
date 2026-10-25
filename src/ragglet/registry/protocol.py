from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ragglet.bench.plan import RunPlan
from ragglet.core.issues import Issue
from ragglet.jobs.protocol import ProgressInfo, TaskFailure, TaskRef, TaskState
from ragglet.storage.layout import StorageTarget


@dataclass(frozen=True)
class NewRun:
    name: str
    scenario_id: str | None  # ссылка на сценарий библиотеки из запроса
    scenario: dict[str, Any]  # cfg.model_dump(): ровно то, что ляжет в scenario.snapshot
    etag: str  # sha256 текста сценария: для «изменился после прогона»
    tags: list[str]
    dataset: str  # label
    collection: str  # primary target
    plan: RunPlan
    warnings: list[Issue]  # preflight при постановке


@dataclass(frozen=True)
class RunRecord:
    id: str  # "<name>_<UTC постановки>[-N]", оно же имя каталога прогона
    run_dir: Path
    name: str
    scenario_id: str | None
    scenario: dict[str, Any]
    etag: str
    tags: list[str]
    dataset: str
    collection: str
    plan: RunPlan
    warnings: list[Issue]
    state: TaskState
    enqueued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failure: TaskFailure | None = None
    final_progress: ProgressInfo | None = None  # «отменён, выполнено 420 / 969»


@dataclass(frozen=True)
class NewBuild:
    name: str  # id профиля
    profile_id: str | None  # ссылка на профиль библиотеки из запроса
    profile: dict[str, Any]
    etag: str
    dataset: str
    targets: list[StorageTarget]  # коллекции, которые будут пересозданы
    warnings: list[Issue]


@dataclass(frozen=True)
class BuildRecord:
    id: str  # "<id профиля>_<UTC постановки>[-N]"
    name: str
    profile_id: str | None
    profile: dict[str, Any]
    etag: str
    dataset: str
    targets: list[StorageTarget]
    warnings: list[Issue]
    state: TaskState
    enqueued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failure: TaskFailure | None = None
    final_progress: ProgressInfo | None = None
    result: dict[str, Any] | None = None  # documents и записанные метаданные


@dataclass(frozen=True)
class TaskChange:
    """Что меняет менеджер задач при переходах состояния; None — поле не меняется."""

    state: TaskState | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failure: TaskFailure | None = None
    final_progress: ProgressInfo | None = None
    result: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunQuery:
    """Фильтры списка прогонов; 30 добавляет свои поля, не меняя смысла этих."""

    scenario_id: str | None = None
    name: str | None = None
    state: TaskState | None = None
    tag: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    offset: int = 0
    limit: int = 50  # от новых к старым по enqueued_at


@dataclass(frozen=True)
class BuildQuery:
    profile_id: str | None = None
    state: TaskState | None = None
    offset: int = 0
    limit: int = 50


@dataclass(frozen=True)
class QueryFilter:
    repeat: int = 0
    outcome: str | None = None
    offset: int = 0
    limit: int = 100


@dataclass(frozen=True)
class Page[T]:
    items: list[T]
    total: int


@dataclass(frozen=True)
class RunSummary:
    repeats: list[dict[str, Any]] = field(default_factory=list)  # summary.json
    aggregate: dict[str, Any] = field(default_factory=dict)  # summary.aggregate.json


@dataclass(frozen=True)
class Artifact:
    name: str
    size: int


class Registry(Protocol):
    """Долговечные записи всех прогонов и сборок (ragglet-30)."""

    async def create_run(self, new: NewRun) -> RunRecord:
        """Резервирует уникальный id: при совпадении — суффикс -2, -3."""
        ...

    async def create_build(self, new: NewBuild) -> BuildRecord: ...

    async def update(self, ref: TaskRef, change: TaskChange) -> None: ...

    async def get_run(self, run_id: str) -> RunRecord | None: ...

    async def get_build(self, build_id: str) -> BuildRecord | None: ...

    async def list_runs(self, query: RunQuery) -> Page[RunRecord]: ...

    async def list_builds(self, query: BuildQuery) -> Page[BuildRecord]: ...

    async def summary(self, run_id: str) -> RunSummary | None: ...

    async def manifest(self, run_id: str) -> dict[str, Any] | None: ...

    async def artifacts(self, run_id: str) -> list[Artifact]: ...

    async def queries(self, run_id: str, query: QueryFilter) -> Page[dict[str, Any]]: ...
