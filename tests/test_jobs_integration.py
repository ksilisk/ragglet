"""
Настоящий воркер и ядро: сборка `demo` во временную коллекцию и прогон по ней. Нужны Qdrant на
localhost:6333 и модель all-MiniLM-L6-v2 (скачается с Hub при первом запуске); без Qdrant тест
пропускается.
"""
from __future__ import annotations

import json
import socket
import uuid
from pathlib import Path

import anyio
import pytest
import yaml

from ragglet.jobs import files
from ragglet.jobs.process import ProcessJobManager
from ragglet.jobs.protocol import TaskKind, TaskRef, TaskSpec, TaskState
from ragglet.registry.memory import InMemoryRegistry

ROOT = Path(__file__).resolve().parents[1]
QDRANT = "http://localhost:6333"


def _qdrant_up() -> bool:
    try:
        socket.create_connection(("localhost", 6333), timeout=1).close()
        return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _qdrant_up(), reason="Qdrant is not running at localhost:6333"),
]


@pytest.fixture
def collection():
    from qdrant_client import QdrantClient

    name = f"ragglet_test_{uuid.uuid4().hex[:8]}"
    yield name
    QdrantClient(QDRANT).delete_collection(name)


def _config(path: str, **overrides) -> dict:
    data = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    for key, value in overrides.items():
        data[key] = {**data.get(key, {}), **value}
    return data


async def _finish(jobs: ProcessJobManager, ref: TaskRef, timeout: float = 600):
    with anyio.fail_after(timeout):
        while (status := await jobs.status(ref)).state in (TaskState.QUEUED, TaskState.RUNNING):
            await anyio.sleep(0.5)
    return status


async def test_build_then_run_on_the_real_core(tmp_path, collection, monkeypatch):
    monkeypatch.chdir(ROOT)  # относительный datasets/demo — как у сервера, запущенного из корня
    registry = InMemoryRegistry(tmp_path / "runs")
    profile = _config("configs/indexes/demo.yaml", storage={"endpoint": QDRANT, "collection": collection})
    scenario = _config(
        "configs/scenarios/baseline_qdrant.yaml",
        run={"repeats": 1, "artifacts_dir": str(tmp_path / "runs")},
        storage={"backend": {"type": "qdrant", "endpoint": QDRANT, "collection": collection}},
    )
    long_scenario = {**scenario, "run": {**scenario["run"], "repeats": 30}}
    build = TaskSpec(TaskRef(TaskKind.BUILD, "demo_build"), "demo", profile, tmp_path / "builds" / "demo_build")
    run = TaskSpec(TaskRef(TaskKind.RUN, "demo_run_1"), "baseline_qdrant", scenario, tmp_path / "runs" / "demo_run_1")
    long = TaskSpec(TaskRef(TaskKind.RUN, "demo_run_2"), "baseline_qdrant", long_scenario,
                    tmp_path / "runs" / "demo_run_2")

    async with ProcessJobManager(registry, cancel_grace_s=5, builds_dir=tmp_path / "builds") as jobs:
        await jobs.submit(build)
        await jobs.submit(run)  # ждёт сборку в очереди
        built = await _finish(jobs, build.ref)
        ran = await _finish(jobs, run.ref)

        await jobs.submit(long)
        with anyio.fail_after(120):  # первый повтор записан, идёт второй
            while (progress := (await jobs.status(long.ref)).progress) is None or progress.done <= 150:
                await anyio.sleep(0.2)
        await jobs.cancel(long.ref)
        cancelled = await _finish(jobs, long.ref, timeout=30)

    assert built.state is TaskState.COMPLETED, (build.run_dir / files.LOG).read_text()
    assert files.read_outcome(build.run_dir).result["documents"] > 0
    assert ran.state is TaskState.COMPLETED, (run.run_dir / files.LOG).read_text()
    assert ran.progress.done == ran.progress.total > 0
    summary = json.loads((run.run_dir / "summary.json").read_text())
    assert len(summary) == 1
    assert (run.run_dir / "manifest.json").exists()

    # отмена настоящего прогона — штатная, без SIGKILL: воркер успел записать итог
    assert cancelled.state is TaskState.CANCELLED, (long.run_dir / files.LOG).read_text()
    assert files.read_outcome(long.run_dir).state is TaskState.CANCELLED
    assert 150 < cancelled.progress.done < cancelled.progress.total
    assert (long.run_dir / "per_query.repeat0.csv").exists()
    assert not (long.run_dir / "summary.json").exists()
