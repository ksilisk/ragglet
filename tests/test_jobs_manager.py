"""ProcessJobManager с фиктивным воркером: очередь, отмена, сбои, прогресс, лог, реестр, остановка."""
from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import anyio
import pytest

from ragglet.jobs import files
from ragglet.jobs.process import ProcessJobManager, decide, read_log
from ragglet.jobs.protocol import (
    LogChunk,
    ProgressInfo,
    TaskFailure,
    TaskFinished,
    TaskKind,
    TaskRef,
    TaskSpec,
    TaskState,
    TaskStatus,
)
from ragglet.registry.memory import InMemoryRegistry
from ragglet.registry.protocol import NewBuild, TaskChange

FAKE = Path(__file__).with_name("jobs_fake_worker.py")
GRACE = 0.5


class RecordingRegistry:
    """Реестр, который только запоминает вызовы update; fail — бросать на каждом."""

    def __init__(self, fail: bool = False):
        self.updates: list[tuple[TaskRef, TaskChange]] = []
        self.fail = fail

    async def update(self, ref: TaskRef, change: TaskChange) -> None:
        self.updates.append((ref, change))
        if self.fail:
            raise OSError("disk full")

    def states(self, ref: TaskRef) -> list[TaskState]:
        return [change.state for r, change in self.updates if r == ref]

    def last(self, ref: TaskRef) -> TaskChange:
        return [change for r, change in self.updates if r == ref][-1]


def manager(registry, tmp_path: Path, **kwargs) -> ProcessJobManager:
    return ProcessJobManager(registry, cancel_grace_s=GRACE, command=[sys.executable, str(FAKE)],
                             builds_dir=tmp_path / "builds", **kwargs)


def spec(tmp_path: Path, name: str, script=None, kind: TaskKind = TaskKind.RUN, **config) -> TaskSpec:
    run_dir = tmp_path / "runs" / name if kind is TaskKind.RUN else None
    return TaskSpec(TaskRef(kind, name), name, {"script": script, **config}, run_dir)


async def until(jobs: ProcessJobManager, ref: TaskRef, check, timeout: float = 10.0) -> TaskStatus:
    with anyio.fail_after(timeout):
        while True:
            status = await jobs.status(ref)
            if check(status):
                return status
            await anyio.sleep(0.02)


def finished(status: TaskStatus) -> bool:
    return status.state not in (TaskState.QUEUED, TaskState.RUNNING)


# --- очередь


async def test_fifo_one_at_a_time(tmp_path):
    registry = RecordingRegistry()
    marks = tmp_path / "marks"
    script = [["mark", str(marks), "start"], ["sleep", 0.2], ["mark", str(marks), "end"]]
    async with manager(registry, tmp_path) as jobs:
        a, b, c = [await jobs.submit(spec(tmp_path, name, script)) for name in ("a", "b", "c")]

        assert (a.state, a.position) == (TaskState.RUNNING, None)
        assert [(s.state, s.position) for s in (b, c)] == [(TaskState.QUEUED, 1), (TaskState.QUEUED, 2)]
        assert [s.ref.id for s in await jobs.queue()] == ["a", "b", "c"]
        assert all((tmp_path / "runs" / name / files.TASK).exists() for name in "abc")

        for ref in (a.ref, b.ref, c.ref):
            assert (await until(jobs, ref, finished)).state is TaskState.COMPLETED

    lines = [line.split() for line in marks.read_text().splitlines()]
    assert [label for label, _ in lines] == ["start", "end"] * 3  # процессы не перекрываются
    assert [float(t) for _, t in lines] == sorted(float(t) for _, t in lines)
    assert registry.states(b.ref) == [TaskState.RUNNING, TaskState.COMPLETED]
    assert await jobs.queue() == []


async def test_submit_needs_a_running_manager(tmp_path):
    with pytest.raises(RuntimeError):
        await manager(RecordingRegistry(), tmp_path).submit(spec(tmp_path, "a"))


# --- отмена


async def test_cancel_queued(tmp_path):
    registry = RecordingRegistry()
    async with manager(registry, tmp_path) as jobs:
        a = await jobs.submit(spec(tmp_path, "a", [["sleep", 30]]))
        b = await jobs.submit(spec(tmp_path, "b"))
        c = await jobs.submit(spec(tmp_path, "c"))

        cancelled = await jobs.cancel(b.ref)

        assert cancelled.state is TaskState.CANCELLED
        assert (await jobs.status(c.ref)).position == 1
        assert registry.states(b.ref) == [TaskState.CANCELLED]
        assert registry.last(b.ref).finished_at is not None
        assert not (tmp_path / "runs" / "b" / files.LOG).exists()  # процесс не запускался
        await jobs.cancel(a.ref)
        await until(jobs, c.ref, finished)


async def test_cancel_running_keeps_final_progress(tmp_path):
    registry = RecordingRegistry()
    async with manager(registry, tmp_path) as jobs:
        a = await jobs.submit(spec(tmp_path, "a", [["progress", "queries", 5, 10], ["sleep", 30]]))
        await until(jobs, a.ref, lambda s: s.progress is not None)

        assert (await jobs.cancel(a.ref)).state is TaskState.RUNNING  # пока процесс не вышел
        status = await until(jobs, a.ref, finished)

    assert status.state is TaskState.CANCELLED
    assert (status.progress.done, status.progress.total) == (5, 10)  # последний вызов, не запись
    assert registry.states(a.ref) == [TaskState.RUNNING, TaskState.CANCELLED]
    assert registry.last(a.ref).final_progress.done == 5
    with pytest.raises(TaskFinished):
        await jobs.cancel(a.ref)


async def test_cancel_kills_a_worker_that_ignores_sigterm(tmp_path):
    async with manager(RecordingRegistry(), tmp_path) as jobs:
        a = await jobs.submit(spec(tmp_path, "a", [["ignore_sigterm"], ["sleep", 30]]))
        await anyio.sleep(0.5)
        started = time.monotonic()
        await jobs.cancel(a.ref)
        status = await until(jobs, a.ref, finished)

    assert status.state is TaskState.CANCELLED
    assert GRACE <= time.monotonic() - started < GRACE + 3
    assert not (tmp_path / "runs" / "a" / files.RESULT).exists()


def _progress(done: int) -> ProgressInfo:
    now = datetime(2026, 11, 25, tzinfo=UTC)
    return ProgressInfo("queries", done, 10, now, now)


@pytest.mark.parametrize("outcome, cancelled, stopping, expected", [
    (files.Outcome(TaskState.COMPLETED), True, True, TaskState.COMPLETED),
    (files.Outcome(TaskState.CANCELLED), False, False, TaskState.CANCELLED),
    (files.Outcome(TaskState.FAILED, TaskFailure("RuntimeError", "boom")), False, False, TaskState.FAILED),
    (files.Outcome(TaskState.FAILED, TaskFailure("RuntimeError", "boom")), True, False, TaskState.CANCELLED),
    (None, False, False, TaskState.FAILED),
    (None, True, False, TaskState.CANCELLED),
    (files.Outcome(TaskState.CANCELLED), True, True, TaskState.FAILED),
])
def test_decide(tmp_path, outcome, cancelled, stopping, expected):
    files.write_json(tmp_path / files.PROGRESS, files.progress_to_json(_progress(4)))
    decided = decide(outcome, cancelled=cancelled, stopping=stopping, code=-9, task_dir=tmp_path)
    assert decided.state is expected
    if stopping and expected is TaskState.FAILED:
        assert decided.failure.type == "ServerShutdown"
    if outcome is None:
        assert decided.final_progress.done == 4  # из последнего progress.json
    if outcome is None and expected is TaskState.FAILED:
        assert decided.failure.message.startswith("worker exited with code -9 (SIGKILL) without result.json")
    if expected is TaskState.CANCELLED:
        assert decided.failure is None


# --- сбои


async def test_failures(tmp_path):
    registry = RecordingRegistry()
    async with manager(registry, tmp_path) as jobs:
        boom = await jobs.submit(spec(tmp_path, "boom", [["raise", "boom", "qid=q7"]]))
        invalid = await jobs.submit(spec(tmp_path, "invalid", [["invalid"]]))
        died = await jobs.submit(spec(tmp_path, "died", [["eprint", "about to die"], ["exit", 9]]))
        killed = await jobs.submit(spec(tmp_path, "killed", [["kill"]]))
        statuses = [await until(jobs, s.ref, finished) for s in (boom, invalid, died, killed)]

    assert all(s.state is TaskState.FAILED for s in statuses)
    boom_f, invalid_f, died_f, killed_f = (s.failure for s in statuses)
    assert (boom_f.type, boom_f.message, boom_f.qid) == ("RuntimeError", "boom", "q7")
    assert invalid_f.title == "Scenario config is invalid"
    assert [i.path for i in invalid_f.issues] == ["metrics.quality.k", None]
    assert died_f.type == "WorkerCrashed"
    assert died_f.message.startswith("worker exited with code 9 without result.json")
    assert died_f.message.endswith("about to die")  # последние строки лога
    assert "code -9 (SIGKILL)" in killed_f.message
    assert registry.last(boom.ref).failure == boom_f


async def test_spawn_failure_is_a_failed_task(tmp_path):
    registry = RecordingRegistry()
    jobs = ProcessJobManager(registry, command=[str(tmp_path / "no-such-python")])
    async with jobs:
        a = await jobs.submit(spec(tmp_path, "a"))
        status = await until(jobs, a.ref, finished)
    assert status.state is TaskState.FAILED
    assert status.failure.type in {"FileNotFoundError", "PermissionError"}
    assert registry.states(a.ref) == [TaskState.RUNNING, TaskState.FAILED]


# --- прогресс и лог


async def test_progress_comes_from_the_file(tmp_path):
    async with manager(RecordingRegistry(), tmp_path) as jobs:
        a = await jobs.submit(spec(tmp_path, "a", [["progress", "queries", 3, 3], ["sleep", 1]]))
        status = await until(jobs, a.ref, lambda s: s.progress is not None and s.progress.done == 3)

        assert status.state is TaskState.RUNNING  # done == total у живого процесса: фаза finishing
        assert status.progress.first_at <= status.progress.updated_at
        assert (await jobs.queue())[0].progress == status.progress
        await until(jobs, a.ref, finished)


async def test_log_by_offsets(tmp_path):
    text = "строка лога ✓\n" * 50
    async with manager(RecordingRegistry(), tmp_path) as jobs:
        a = await jobs.submit(spec(tmp_path, "a", [["print", text.rstrip("\n")], ["sleep", 0.5]]))
        await anyio.sleep(0.3)
        assert not (await jobs.log(a.ref)).complete  # задача ещё идёт
        await until(jobs, a.ref, finished)

        chunks, offset = [], 0
        while True:
            chunk = await jobs.log(a.ref, offset, limit=7)  # меньше двух кириллических символов с хвостом
            chunks.append(chunk.text)
            offset = chunk.next_offset
            if chunk.complete:
                break

    log = "".join(chunks)
    assert log == (tmp_path / "runs" / "a" / files.LOG).read_text(encoding="utf-8")
    assert text in log and log.startswith("== run a (a) started")
    assert await jobs.log(TaskRef(TaskKind.RUN, "unknown")) == LogChunk("", 0, True)  # лог отдаёт реестр


def test_read_log_replaces_broken_bytes_at_the_end(tmp_path):
    path = tmp_path / files.LOG
    path.write_bytes(b"ok " + "я".encode()[:1])  # оборванный символ в конце

    running = read_log(path, 0, 100, finished=False)
    assert (running.text, running.next_offset, running.complete) == ("ok ", 3, False)
    done = read_log(path, 3, 100, finished=True)
    assert (done.text, done.next_offset, done.complete) == ("�", 4, True)


# --- реестр


async def test_build_result_reaches_the_registry(tmp_path):
    registry = InMemoryRegistry(tmp_path / "runs")
    record = await registry.create_build(NewBuild(name="demo", profile_id="demo", profile={}, etag="e",
                                                  dataset="local:datasets/demo", targets=[], warnings=[]))
    ref = TaskRef(TaskKind.BUILD, record.id)
    async with manager(registry, tmp_path) as jobs:
        await jobs.submit(TaskSpec(ref, "demo", {"script": None}))  # без run_dir: до ragglet-30
        await until(jobs, ref, finished)

    build = await registry.get_build(record.id)
    assert build.state is TaskState.COMPLETED
    assert build.result == {"documents": 3}
    assert build.final_progress.done == 3
    assert build.started_at <= build.finished_at
    assert (tmp_path / "builds" / record.id / files.RESULT).exists()


async def test_broken_registry_does_not_stop_the_queue(tmp_path):
    async with manager(RecordingRegistry(fail=True), tmp_path) as jobs:
        a = await jobs.submit(spec(tmp_path, "a"))
        b = await jobs.submit(spec(tmp_path, "b"))
        assert (await until(jobs, b.ref, finished)).state is TaskState.COMPLETED
        assert (await jobs.status(a.ref)).state is TaskState.COMPLETED


# --- остановка сервера


async def test_shutdown_fails_current_and_queued(tmp_path):
    registry = RecordingRegistry()
    async with manager(registry, tmp_path) as jobs:
        a = await jobs.submit(spec(tmp_path, "a", [["sleep", 30]]))
        b = await jobs.submit(spec(tmp_path, "b"))
        await anyio.sleep(0.5)
        started = time.monotonic()

    assert time.monotonic() - started < GRACE + 3
    for ref in (a.ref, b.ref):
        assert registry.last(ref).state is TaskState.FAILED
        assert registry.last(ref).failure.type == "ServerShutdown"
    assert registry.states(b.ref) == [TaskState.FAILED]
    pid = int((tmp_path / "runs" / "a" / files.LOG).read_text().split("pid ")[1].split()[0])
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
