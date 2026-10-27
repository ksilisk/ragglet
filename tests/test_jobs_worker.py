"""Воркер задачи: файлы, троттлинг прогресса, итоги, отмена, сирота. Ядро заменено фиктивным."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import anyio
import pytest

from ragglet.core.issues import Severity
from ragglet.jobs import files, worker
from ragglet.jobs.protocol import TaskKind, TaskRef, TaskSpec, TaskState

FAKE = Path(__file__).with_name("jobs_fake_worker.py")


def make_task(tmp_path: Path, script=None, kind: TaskKind = TaskKind.RUN) -> Path:
    task_dir = tmp_path / "t_20261125T101540Z"
    task_dir.mkdir()
    files.write_task(TaskSpec(TaskRef(kind, task_dir.name), "t", {"script": script}, task_dir))
    return task_dir


def start(task_dir: Path, *prefix: str) -> subprocess.Popen:
    with open(task_dir / files.LOG, "ab") as log:
        return subprocess.Popen([*prefix, sys.executable, str(FAKE), str(task_dir)], stdout=log, stderr=log,
                                env=os.environ | {"PYTHONUNBUFFERED": "1"})


def wait_for(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} did not appear"
        time.sleep(0.02)


def log_of(task_dir: Path) -> str:
    return (task_dir / files.LOG).read_text(encoding="utf-8")


@pytest.fixture
def restore_signals():
    saved = {sig: signal.getsignal(sig) for sig in worker.CANCEL_SIGNALS}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


# --- троттлинг


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_progress_writes_first_last_and_once_a_second(tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(files, "write_json", lambda path, data: written.append((data["done"], data["total"])))
    clock = Clock()
    progress = worker.ThrottledProgress(tmp_path / files.PROGRESS, clock=clock)

    for i in range(1000):  # 1000 запросов за 0,5 с
        clock.t = i * 0.0005
        progress("queries", i + 1, 2000)
    assert written == [(1, 2000)]

    for done in (1001, 1002, 1003):  # раз в секунду — каждый
        clock.t += 1.0
        progress("queries", done, 2000)
    clock.t += 0.1
    progress("queries", 2000, 2000)  # done == total — всегда
    assert written == [(1, 2000), (1001, 2000), (1002, 2000), (1003, 2000), (2000, 2000)]

    clock.t += 0.1
    progress("corpus", 1, 5)  # новая стадия — сразу и с новым first_at
    assert written[-1] == (1, 5)
    assert progress.last.first_at == progress.last.updated_at


def test_progress_last_is_the_last_call_not_the_last_write(tmp_path):
    clock = Clock()
    progress = worker.ThrottledProgress(tmp_path / files.PROGRESS, clock=clock)
    for i in range(1, 6):
        clock.t = i * 0.01
        progress("queries", i, 10)

    assert files.read_progress(tmp_path).done == 1
    last = progress.last
    assert (last.stage, last.done, last.total) == ("queries", 5, 10)
    assert last.first_at.timestamp() == pytest.approx(0.01)
    assert last.updated_at.timestamp() == pytest.approx(0.05)


def test_progress_never_raises(tmp_path):
    progress = worker.ThrottledProgress(tmp_path / "missing" / files.PROGRESS)
    progress("queries", 1, 2)  # каталога нет: OSError проглатывается
    assert progress.last.done == 1


# --- файлы задачи


def test_task_and_outcome_round_trip(tmp_path):
    task_dir = make_task(tmp_path, [["return", None]], kind=TaskKind.BUILD)
    spec = files.read_task(task_dir)
    assert spec.ref == TaskRef(TaskKind.BUILD, "t_20261125T101540Z")
    assert spec.run_dir == task_dir
    assert files.read_outcome(task_dir) is None
    assert not list(task_dir.glob(".*"))  # временных файлов атомарной записи не осталось


# --- процесс воркера


def test_completed_task_writes_result_progress_and_log(tmp_path):
    task_dir = make_task(tmp_path, [["progress", "corpus", 3, 3], ["print", "to stdout"], ["eprint", "to stderr"],
                                    ["return", {"documents": 3}]], kind=TaskKind.BUILD)

    assert start(task_dir).wait(10) == 0

    outcome = files.read_outcome(task_dir)
    assert outcome.state is TaskState.COMPLETED
    assert outcome.result == {"documents": 3}
    assert (outcome.final_progress.done, outcome.final_progress.total) == (3, 3)
    assert files.read_progress(task_dir).done == 3
    log = log_of(task_dir)
    assert log.startswith("== build t_20261125T101540Z (t) started ")
    assert "to stdout\n" in log and "to stderr\n" in log
    assert log.splitlines()[-1].startswith("== completed after ")


def test_failure_keeps_type_message_qid_and_traceback(tmp_path):
    task_dir = make_task(tmp_path, [["raise", "boom", "qid=q7"]])

    assert start(task_dir).wait(10) == 1

    failure = files.read_outcome(task_dir).failure
    assert (failure.type, failure.message, failure.qid) == ("RuntimeError", "boom", "q7")
    log = log_of(task_dir)
    assert "Traceback (most recent call last)" in log and "qid=q7" in log
    assert log.splitlines()[-1].startswith("== failed: RuntimeError: boom after ")


def test_validation_error_keeps_issues(tmp_path):
    task_dir = make_task(tmp_path, [["invalid"]])

    assert start(task_dir).wait(10) == 2

    failure = files.read_outcome(task_dir).failure
    assert failure.type == "ValidationError"
    assert (failure.title, failure.hint) == ("Scenario config is invalid", "lower metrics.quality.k")
    assert [(i.code, i.severity, i.path) for i in failure.issues] == [
        ("config.value_error", Severity.ERROR, "metrics.quality.k"),
        ("index.not_green", Severity.WARNING, None),
    ]
    assert "config.value_error" in log_of(task_dir)  # таблица проблем


def test_long_message_is_clipped():
    failure = worker.failure_of(RuntimeError("x" * 5000))
    assert len(failure.message) < worker.MESSAGE_LIMIT + 10


def test_exit_without_result(tmp_path):
    task_dir = make_task(tmp_path, [["exit", 9]])

    assert start(task_dir).wait(10) == 9
    assert files.read_outcome(task_dir) is None


@pytest.mark.parametrize("sleep", ["sleep", "async_sleep"])
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_cancels_with_final_progress(tmp_path, sleep, sig):
    task_dir = make_task(tmp_path, [["progress", "queries", 5, 10], [sleep, 30]])
    proc = start(task_dir)
    wait_for(task_dir / files.PROGRESS)
    time.sleep(0.3)

    proc.send_signal(sig)

    assert proc.wait(10) == worker.CANCELLED_CODE
    outcome = files.read_outcome(task_dir)
    assert outcome.state is TaskState.CANCELLED
    assert (outcome.final_progress.done, outcome.final_progress.total) == (5, 10)
    assert log_of(task_dir).splitlines()[-1].startswith("== cancelled at queries 5 / 10 after ")


def test_orphan_cancels_itself(tmp_path):
    task_dir = make_task(tmp_path, [["sleep", 60]])
    # промежуточный процесс запускает воркер и через секунду выходит: воркер становится сиротой
    spawner = "import subprocess, sys, time; subprocess.Popen(sys.argv[1:]); time.sleep(1)"
    with open(task_dir / files.LOG, "ab") as log:
        subprocess.run([sys.executable, "-c", spawner, sys.executable, str(FAKE), str(task_dir)],
                       stdout=log, stderr=log, check=True)

    wait_for(task_dir / files.RESULT, timeout=worker.PARENT_CHECK_S + 5)
    assert files.read_outcome(task_dir).state is TaskState.CANCELLED
    assert "is gone: cancelling the task" in log_of(task_dir)


# --- отмена прогона в процессе


async def _sleep_with_finally(marks: list[str]) -> None:
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        await anyio.sleep(5)
    finally:
        marks.append("finally")


def test_cancellable_turns_signal_into_cancelled(restore_signals):
    marks: list[str] = []
    started = time.monotonic()
    with pytest.raises(worker.Cancelled):
        anyio.run(worker.cancellable, lambda: _sleep_with_finally(marks))
    assert marks == ["finally"]
    assert time.monotonic() - started < 2


def test_cancellable_passes_core_errors_through(restore_signals):
    async def boom():
        raise KeyError("q1")

    with pytest.raises(KeyError):  # исключение ядра как есть, а не ExceptionGroup
        anyio.run(worker.cancellable, boom)
    assert anyio.run(worker.cancellable, lambda: anyio.sleep(0)) is None
