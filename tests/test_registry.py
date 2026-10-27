"""FileRegistry: run.json и build.json, скан, импорт каталогов без записи, восстановление."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from api_support import Clock
from registry_support import RUN_ID, make_feb_run_dir, make_run_dir

from ragglet.bench.plan import RunPlan
from ragglet.core.issues import warning
from ragglet.jobs.files import PROGRESS, Outcome, progress_to_json, write_json, write_outcome
from ragglet.jobs.memory import InMemoryJobManager
from ragglet.jobs.protocol import ProgressInfo, TaskFailure, TaskKind, TaskRef, TaskSpec, TaskState
from ragglet.registry.files import FileRegistry
from ragglet.registry.protocol import NewBuild, NewRun, RunQuery, TaskChange
from ragglet.storage.layout import StorageTarget

AT = datetime(2026, 11, 25, 10, 15, 40, tzinfo=UTC)


def new_run(name: str = "demo_run") -> NewRun:
    return NewRun(
        name=name, scenario_id=name, scenario={"name": name, "tags": ["demo"], "run": {"repeats": 1}},
        etag="0123456789abcdef", tags=["night"], dataset="local:datasets/demo", collection="chunks_v1",
        plan=RunPlan(repeats=1, queries_per_repeat=3, total=3, load={"type": "closed", "concurrency": 1}),
        warnings=[warning("index.not_green", "target default (chunks_v1)", "collection status is yellow")],
    )


def new_build() -> NewBuild:
    return NewBuild(name="demo", profile_id="demo", profile={"id": "demo"}, etag="fedcba9876543210",
                    dataset="local:datasets/demo",
                    targets=[StorageTarget("default", "qdrant", "http://q:6333", "chunks_v1", "storage")],
                    warnings=[])


@pytest.fixture
def registry(tmp_path):
    return FileRegistry(tmp_path / "runs", tmp_path / "builds", now=Clock(AT))


def reopen(registry: FileRegistry) -> FileRegistry:
    return FileRegistry(registry.artifacts_dir, registry.builds_dir)


async def test_run_record_lives_in_run_dir_and_survives_reopen(registry):
    record = await registry.create_run(new_run())

    assert record.id == "demo_run_20261125T101540Z"
    assert record.run_dir == registry.artifacts_dir / record.id
    assert json.loads((record.run_dir / "run.json").read_text())["state"] == "queued"

    ref = TaskRef(TaskKind.RUN, record.id)
    await registry.update(ref, TaskChange(state=TaskState.RUNNING, started_at=AT))
    failure = TaskFailure(type="RuntimeError", message="boom", qid="q003")
    progress = ProgressInfo(stage="queries", done=2, total=3, first_at=AT, updated_at=AT)
    await registry.update(ref, TaskChange(state=TaskState.FAILED, finished_at=AT, failure=failure,
                                          final_progress=progress, result={"ignored": True}))

    assert sorted(p.name for p in record.run_dir.iterdir()) == ["run.json"]  # временных файлов нет
    again = await reopen(registry).get_run(record.id)
    assert again == await registry.get_run(record.id)
    assert (again.state, again.failure, again.final_progress, again.origin) == (
        TaskState.FAILED, failure, progress, "api")
    assert again.warnings[0].code == "index.not_green" and again.plan.total == 3


async def test_same_second_ids_skip_taken_dirs(registry):
    (registry.artifacts_dir / "demo_run_20261125T101540Z-2").mkdir(parents=True)  # каталог от CLI

    ids = [(await registry.create_run(new_run())).id for _ in range(3)]

    assert ids == ["demo_run_20261125T101540Z", "demo_run_20261125T101540Z-3", "demo_run_20261125T101540Z-4"]


async def test_unreadable_run_json_is_imported_not_overwritten(registry):
    run_dir = make_run_dir(registry.artifacts_dir)
    (run_dir / "run.json").write_text("{not json", encoding="utf-8")

    record = await registry.get_run(RUN_ID)

    assert (record.origin, record.state) == ("import", TaskState.COMPLETED)
    assert (run_dir / "run.json").read_text(encoding="utf-8") == "{not json"


async def test_build_record_lives_in_build_dir(registry):
    record = await registry.create_build(new_build())

    assert record.build_dir == registry.builds_dir / "demo_20261125T101540Z"
    await registry.update(TaskRef(TaskKind.BUILD, record.id),
                          TaskChange(state=TaskState.COMPLETED, finished_at=AT, result={"documents": 50}))

    again = await reopen(registry).get_build(record.id)
    assert (again.state, again.result, again.targets) == (TaskState.COMPLETED, {"documents": 50}, record.targets)
    assert (again.build_dir / "build.json").is_file()


async def test_complete_dir_without_record_is_imported(registry):
    run_dir = make_run_dir(registry.artifacts_dir)
    before = sorted(p.name for p in run_dir.iterdir())

    record = await registry.get_run(RUN_ID)

    assert (record.origin, record.state, record.failure) == ("import", TaskState.COMPLETED, None)
    assert (record.name, record.scenario_id, record.etag, record.plan) == ("demo_run", None, None, None)
    assert record.dataset.startswith("local:") and record.dataset.endswith("datasets/demo")
    assert record.collection == "chunks_v1"
    assert record.enqueued_at == datetime(2026, 10, 24, 18, 9, 17, tzinfo=UTC)
    assert (record.started_at, record.finished_at) == (None, None)  # mtime не используем
    assert record.metrics.repeats == 2
    assert record.metrics.means["doc_ndcg@3_mean"] == pytest.approx((1 + 1 / 1.5849625007211562) / 3)
    assert sorted(p.name for p in run_dir.iterdir()) == before  # в импортированный каталог не пишем


async def test_incomplete_dir_becomes_completed_when_cli_finishes(registry):
    run_dir = make_run_dir(registry.artifacts_dir, complete=False)

    record = await registry.get_run(RUN_ID)
    assert (record.state, record.failure.type) == (TaskState.FAILED, "RunIncomplete")
    assert record.metrics is None

    make_run_dir(registry.artifacts_dir, "other_20261024T180918Z")  # чтобы взять summary готового прогона
    for name in ("summary.json", "summary.aggregate.json", "summary.aggregate.csv"):
        (run_dir / name).write_bytes((registry.artifacts_dir / "other_20261024T180918Z" / name).read_bytes())

    page = await registry.list_runs(RunQuery())
    assert {r.id: r.state for r in page.items}[RUN_ID] is TaskState.COMPLETED


async def test_what_is_not_a_run_is_not_listed(registry):
    make_feb_run_dir(registry.artifacts_dir)
    (registry.artifacts_dir / "empty_20261024T180917Z").mkdir()  # без snapshot
    make_run_dir(registry.artifacts_dir, ".hidden_20261024T180917Z")
    make_run_dir(registry.artifacts_dir, "renamed-by-hand")

    page = await registry.list_runs(RunQuery())

    assert [r.id for r in page.items] == ["baseline_qdrant_20260221T135048Z"]
    feb = page.items[0]
    assert (feb.name, feb.dataset, feb.collection) == (
        "baseline_qdrant", "local:data/queries/baseline_qdrant.jsonl", "chunks_v1")
    assert feb.metrics.means == {"recall@1_mean": 1.0, "lat_total_p50": 12.0, "lat_total_p95": 12.0}


async def test_removed_dir_leaves_the_list(registry):
    run_dir = make_run_dir(registry.artifacts_dir)
    assert (await registry.list_runs(RunQuery())).total == 1

    for path in run_dir.iterdir():
        path.unlink()
    run_dir.rmdir()

    assert (await registry.list_runs(RunQuery())).total == 0


async def test_dir_created_after_scan_is_found_by_id(registry):
    assert (await registry.list_runs(RunQuery())).total == 0
    make_run_dir(registry.artifacts_dir)

    assert (await registry.get_run(RUN_ID)).state is TaskState.COMPLETED
    assert await registry.get_run("..") is None


async def test_recover_finishes_orphans_only(registry):
    jobs = InMemoryJobManager(registry)
    queued = await registry.create_run(new_run("a"))
    running = await registry.create_run(new_run("b"))
    finished = await registry.create_run(new_run("c"))
    known = await registry.create_run(new_run("d"))
    build = await registry.create_build(new_build())
    for record in (running, finished):
        await registry.update(TaskRef(TaskKind.RUN, record.id), TaskChange(state=TaskState.RUNNING, started_at=AT))
    await jobs.submit(TaskSpec(ref=TaskRef(TaskKind.RUN, known.id), name="d", config={}))
    # воркер дописал summary, а переход до сервера не дошёл
    (finished.run_dir / "summary.aggregate.csv").write_text("scenario\nc\n", encoding="utf-8")

    await registry.recover(jobs)

    restarted = reopen(registry)
    states = {r.id: (r.state, r.failure.type if r.failure else None)
              for r in (await restarted.list_runs(RunQuery())).items}
    assert states == {
        queued.id: (TaskState.FAILED, "ServerRestart"),
        running.id: (TaskState.FAILED, "ServerRestart"),
        finished.id: (TaskState.COMPLETED, None),
        known.id: (TaskState.QUEUED, None),
    }
    assert "before the run started" in (await restarted.get_run(queued.id)).failure.message
    assert (await restarted.get_build(build.id)).failure.type == "ServerRestart"


async def test_recover_keeps_last_progress_of_the_worker(registry):
    running = await registry.create_run(new_run("a"))
    cancelled = await registry.create_run(new_run("b"))
    build = await registry.create_build(new_build())
    progress = ProgressInfo(stage="queries", done=635, total=3000, first_at=AT, updated_at=AT)
    last = ProgressInfo(stage="queries", done=641, total=3000, first_at=AT, updated_at=AT)
    for ref, task_dir in ((TaskRef(TaskKind.RUN, running.id), running.run_dir),
                          (TaskRef(TaskKind.RUN, cancelled.id), cancelled.run_dir),
                          (TaskRef(TaskKind.BUILD, build.id), build.build_dir)):
        await registry.update(ref, TaskChange(state=TaskState.RUNNING, started_at=AT))
        write_json(task_dir / PROGRESS, progress_to_json(progress))  # воркер успел до падения сервера
    # сторожевой поток воркера отменил прогон и записал итог: в нём последний вызов, а не раз в секунду
    write_outcome(cancelled.run_dir, Outcome(state=TaskState.CANCELLED, final_progress=last))

    await registry.recover(InMemoryJobManager(registry))

    restarted = reopen(registry)
    run = await restarted.get_run(running.id)
    assert (run.failure.type, run.final_progress) == ("ServerRestart", progress)
    assert (await restarted.get_run(cancelled.id)).final_progress == last
    assert (await restarted.get_build(build.id)).final_progress == progress


async def test_summary_and_manifest(registry):
    run_dir = make_run_dir(registry.artifacts_dir)
    aggregate = json.loads((run_dir / "summary.aggregate.json").read_text())
    (run_dir / "summary.aggregate.json").write_text(
        json.dumps({**aggregate, "emb_hit_rate_mean_over_repeats": float("nan")}), encoding="utf-8")

    summary = await registry.summary(RUN_ID)

    assert [r["repeat_idx"] for r in summary.repeats] == [0, 1]
    assert summary.aggregate["emb_hit_rate_mean_over_repeats"] is None  # NaN -> null
    assert summary.aggregate["repeats"] == 2
    assert (await registry.manifest(RUN_ID))["index"]["built_at"] == "2026-10-20T12:00:00Z"

    failed = make_run_dir(registry.artifacts_dir, "demo_run_20261024T180918Z", complete=False, manifest=False)
    assert await registry.summary(failed.name) is None
    assert await registry.manifest(failed.name) is None
