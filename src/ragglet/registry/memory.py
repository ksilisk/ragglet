from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragglet.jobs.protocol import TaskKind, TaskRef, TaskState
from ragglet.registry.protocol import (
    Artifact,
    BuildQuery,
    BuildRecord,
    NewBuild,
    NewRun,
    Page,
    QueryFilter,
    RunQuery,
    RunRecord,
    RunSummary,
    TaskChange,
)


def unique_id(base: str, taken: Callable[[str], bool]) -> str:
    """base, а если занят — base-2, base-3: время в id всегда настоящее."""
    candidate, n = base, 2
    while taken(candidate):
        candidate, n = f"{base}-{n}", n + 1
    return candidate


class InMemoryRegistry:
    """Заглушка реестра до ragglet-30: записи в памяти процесса, артефактов не читает."""

    def __init__(self, artifacts_dir: Path, now: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.artifacts_dir = artifacts_dir
        self._now = now
        self._runs: dict[str, RunRecord] = {}
        self._builds: dict[str, BuildRecord] = {}

    async def create_run(self, new: NewRun) -> RunRecord:
        now = self._now()
        run_id = unique_id(
            f"{new.name}_{now:%Y%m%dT%H%M%SZ}",
            # каталог мог создать и CLI, который до ragglet-37 зовёт ядро в обход API
            lambda candidate: candidate in self._runs or (self.artifacts_dir / candidate).exists(),
        )
        record = RunRecord(
            id=run_id, run_dir=self.artifacts_dir / run_id, state=TaskState.QUEUED, enqueued_at=now,
            **{f.name: getattr(new, f.name) for f in fields(NewRun)},
        )
        self._runs[run_id] = record
        return record

    async def create_build(self, new: NewBuild) -> BuildRecord:
        now = self._now()
        build_id = unique_id(f"{new.name}_{now:%Y%m%dT%H%M%SZ}", self._builds.__contains__)
        record = BuildRecord(
            id=build_id, state=TaskState.QUEUED, enqueued_at=now,
            **{f.name: getattr(new, f.name) for f in fields(NewBuild)},
        )
        self._builds[build_id] = record
        return record

    async def update(self, ref: TaskRef, change: TaskChange) -> None:
        records: dict[str, Any] = self._runs if ref.kind is TaskKind.RUN else self._builds
        changes = {f.name: getattr(change, f.name) for f in fields(TaskChange) if getattr(change, f.name) is not None}
        if ref.kind is TaskKind.RUN:
            changes.pop("result", None)
        records[ref.id] = replace(records[ref.id], **changes)

    async def get_run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    async def get_build(self, build_id: str) -> BuildRecord | None:
        return self._builds.get(build_id)

    async def list_runs(self, query: RunQuery) -> Page[RunRecord]:
        found = [
            r for r in self._runs.values()
            if (query.scenario_id is None or r.scenario_id == query.scenario_id)
            and (query.name is None or r.name == query.name)
            and (query.state is None or r.state is query.state)
            and (query.tag is None or query.tag in r.tags or query.tag in r.scenario.get("tags", []))
            and (query.since is None or r.enqueued_at >= query.since)
            and (query.until is None or r.enqueued_at < query.until)
        ]
        found.sort(key=lambda r: r.enqueued_at, reverse=True)
        return Page(items=found[query.offset:query.offset + query.limit], total=len(found))

    async def list_builds(self, query: BuildQuery) -> Page[BuildRecord]:
        found = [
            b for b in self._builds.values()
            if (query.profile_id is None or b.profile_id == query.profile_id)
            and (query.state is None or b.state is query.state)
        ]
        found.sort(key=lambda b: b.enqueued_at, reverse=True)
        return Page(items=found[query.offset:query.offset + query.limit], total=len(found))

    async def summary(self, run_id: str) -> RunSummary | None:
        return None

    async def manifest(self, run_id: str) -> dict[str, Any] | None:
        return None

    async def artifacts(self, run_id: str) -> list[Artifact]:
        return []

    async def queries(self, run_id: str, query: QueryFilter) -> Page[dict[str, Any]]:
        return Page(items=[], total=0)
