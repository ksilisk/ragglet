from __future__ import annotations

import re
import time

import anyio
import httpx
import pytest
from api_support import DEMO_SHA, QDRANT, collection, index_meta, profile_yaml, scenario_yaml

from ragglet.core.issues import Issue, Severity
from ragglet.jobs.protocol import TaskFailure, TaskKind, TaskRef, TaskState
from ragglet.registry.protocol import RunQuery


def _issues(response) -> list[tuple[str, str | None, str | None]]:
    return [(i["code"], i["path"], i["where"]) for i in response.json()["issues"]]


@pytest.fixture
def no_models(monkeypatch):
    from ragglet.bench import run_scenario as run_module

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not load models")

    monkeypatch.setattr(run_module, "SentenceTransformerEmbedder", forbidden)


def _broken_dataset(tmp_path):
    root = tmp_path / "broken"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text('{"_id":"d1","text":"a"}\n', encoding="utf-8")
    (root / "queries.jsonl").write_text('{"_id":"q1","text":"a?"}\n', encoding="utf-8")
    (root / "qrels" / "test.tsv").write_text("q1\td2\t1\n", encoding="utf-8")
    return root

# -----------------------
# Dry-run прогона
# -----------------------


def test_run_check_reports_preflight_and_plan(stand, no_models):
    stand.qdrant.collections[(QDRANT, "chunks_v1")] = collection(index_meta(), status="yellow")

    response = stand.client.post("/v1/runs", json={"scenario": scenario_yaml(), "dry_run": True})

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["dataset"]["corpus_sha256"] == DEMO_SHA
    queries = body["dataset"]["queries"]
    assert queries > 0
    assert body["plan"] == {"repeats": 2, "queries_per_repeat": queries, "total": 2 * queries,
                            "load": {"type": "closed", "concurrency": 1}}
    assert body["targets"] == [{"id": "default", "endpoint": QDRANT, "collection": "chunks_v1", "status": "yellow",
                                "points": 53, "indexed_vectors": 0}]
    assert body["index"]["index_profile_id"] == "demo"
    assert [(w["code"], w["path"], w["where"]) for w in body["warnings"]] == [
        ("index.not_green", None, "target default (chunks_v1)"),
    ]
    assert not stand.settings.artifacts_dir.exists()


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"retrieval": {"strategy": "cascade", "sources": [{"name": "v"}]}},
         [("retrieval.not_implemented", "retrieval.strategy", "retrieval.strategy")]),
        ({"cache": {"enabled": True, "embedding": {"endpoint": "redis://r:6379/0"}}},
         [("cache.unreachable", "cache.embedding.endpoint", "cache.embedding (redis://r:6379/0)")]),
    ],
    ids=["cascade", "redis_down"],
)
def test_run_check_core_errors(stand, overrides, expected):
    response = stand.client.post("/v1/runs", json={"scenario": scenario_yaml(**overrides), "dry_run": True})
    assert response.status_code == 422
    assert _issues(response) == expected


def test_run_check_dataset_errors_have_where_not_path(stand, tmp_path):
    response = stand.client.post("/v1/runs", json={
        "scenario": scenario_yaml(dataset={"id": str(_broken_dataset(tmp_path))}), "dry_run": True,
    })
    assert response.status_code == 422
    assert ("qrels.unknown_doc", None, "qrels/test.tsv:1") in _issues(response)


def test_run_check_missing_dataset_points_to_dataset_id(stand, tmp_path):
    response = stand.client.post("/v1/runs", json={
        "scenario": scenario_yaml(dataset={"id": str(tmp_path / "nope")}), "dry_run": True,
    })
    assert [(code, path) for code, path, _ in _issues(response)] == [("dataset.not_found", "dataset.id")]


def test_run_check_unreachable_qdrant(stand):
    stand.qdrant.down.add(QDRANT)
    response = stand.client.post("/v1/runs", json={"scenario": scenario_yaml(), "dry_run": True})
    assert response.status_code == 422
    assert _issues(response) == [("storage.unreachable", "storage.backend.endpoint", f"target default ({QDRANT})")]


def test_library_ref_must_match_name(stand):
    response = stand.client.post("/v1/runs", json={"scenario": scenario_yaml("a"), "scenario_id": "b"})
    assert response.status_code == 422
    assert _issues(response) == [("request.library_ref", "scenario_id", None)]


async def test_dry_run_does_not_block_the_event_loop(stand, monkeypatch):
    from ragglet.bench import preflight as preflight_module

    real = preflight_module.load_eval_dataset

    def slow(cfg):
        time.sleep(1.0)  # чтение и хэш большого датасета
        return real(cfg)

    monkeypatch.setattr(preflight_module, "load_eval_dataset", slow)
    transport = httpx.ASGITransport(app=stand.client.app)
    latency: list[float] = []
    async with httpx.AsyncClient(transport=transport, base_url="http://ragglet") as client:
        async def check() -> None:
            response = await client.post("/v1/runs", json={"scenario": scenario_yaml(), "dry_run": True})
            assert response.status_code == 200

        async def poll() -> None:
            await anyio.sleep(0.3)  # проверка уже в потоке
            started = time.monotonic()
            assert (await client.get("/v1/tasks")).status_code == 200
            latency.append(time.monotonic() - started)

        async with anyio.create_task_group() as tg:
            tg.start_soon(check)
            tg.start_soon(poll)
    assert latency[0] < 0.2

# -----------------------
# Постановка и жизнь задачи
# -----------------------


def _enqueue(stand, name: str = "demo_run", **body) -> httpx.Response:
    return stand.client.post("/v1/runs", json={"scenario": scenario_yaml(name), **body})


def test_enqueue_assigns_run_id_and_queues_task(stand):
    response = _enqueue(stand, scenario_id="demo_run", tags=["night"])

    assert response.status_code == 202
    run_id = response.json()["id"]
    assert run_id == "demo_run_20261125T101540Z"
    assert response.headers["location"] == f"/v1/runs/{run_id}"
    assert response.json()["state"] == "queued" and response.json()["position"] == 1

    record = anyio.run(stand.registry.get_run, run_id)
    assert record.scenario_id == "demo_run" and record.tags == ["night"]
    assert record.run_dir == stand.settings.artifacts_dir / run_id
    assert record.scenario["run"]["artifacts_dir"] == str(stand.settings.artifacts_dir)  # клиентский путь заменён
    spec = stand.jobs._tasks[TaskRef(TaskKind.RUN, run_id)].spec
    assert spec.run_dir == record.run_dir and spec.config["name"] == "demo_run"


def test_same_second_enqueues_get_suffixes(stand):
    ids = [_enqueue(stand).json()["id"] for _ in range(3)]
    assert ids == ["demo_run_20261125T101540Z", "demo_run_20261125T101540Z-2", "demo_run_20261125T101540Z-3"]
    assert all(re.fullmatch(r"demo_run_\d{8}T\d{6}Z(-\d+)?", run_id) for run_id in ids)


def test_rejected_run_leaves_no_trace(stand):
    response = stand.client.post("/v1/runs", json={"scenario": scenario_yaml(metrics={"quality": {"k": [99]}})})
    assert response.status_code == 422
    assert anyio.run(stand.jobs.queue) == []
    assert anyio.run(stand.registry.list_runs, RunQuery()).total == 0


def test_run_status_follows_the_task(stand):
    run_id = _enqueue(stand).json()["id"]
    ref = TaskRef(TaskKind.RUN, run_id)
    get = lambda: stand.client.get(f"/v1/runs/{run_id}").json()  # noqa: E731

    queued = get()
    assert (queued["state"], queued["position"], queued["phase"]) == ("queued", 1, None)
    assert queued["plan"]["total"] == 2 * queued["plan"]["queries_per_repeat"]

    anyio.run(stand.jobs.start, ref)
    assert (get()["state"], get()["phase"]) == ("running", "preparing")

    stand.jobs.report(ref, "queries", 3, 10)
    status = get()
    assert status["phase"] == "progress"
    assert (status["progress"]["done"], status["progress"]["total"]) == (3, 10)

    stand.jobs.report(ref, "queries", 10, 10)
    assert get()["phase"] == "finishing"

    failure = TaskFailure(type="ValidationError", message="index changed", title="Index is not compatible",
                          issues=(Issue("index.corpus", Severity.ERROR, "target default (chunks_v1)", "corpus differs"),))
    anyio.run(stand.jobs.finish, ref, TaskState.FAILED, failure)
    failed = get()
    assert failed["state"] == "failed" and failed["phase"] is None
    assert failed["error"]["problem"]["issues"][0]["code"] == "index.corpus"
    assert failed["progress"]["done"] == 10  # прогресс на момент конца
    assert failed["finished_at"] == "2026-11-25T10:15:40Z"


def test_cancel_queued_then_finished(stand):
    run_id = _enqueue(stand).json()["id"]

    cancelled = stand.client.post(f"/v1/runs/{run_id}/cancel")

    assert cancelled.status_code == 200 and cancelled.json()["state"] == "cancelled"
    again = stand.client.post(f"/v1/runs/{run_id}/cancel")
    assert again.status_code == 409
    assert _issues(again)[0][0] == "task.finished"
    assert stand.client.post("/v1/runs/nope_20261125T101540Z/cancel").status_code == 404


def test_log_by_offset(stand):
    run_id = _enqueue(stand).json()["id"]
    stand.jobs.append_log(TaskRef(TaskKind.RUN, run_id), "line 1\nline 2\n")

    first = stand.client.get(f"/v1/runs/{run_id}/log").json()
    assert first == {"text": "line 1\nline 2\n", "offset": 0, "next_offset": 14, "complete": False}
    rest = stand.client.get(f"/v1/runs/{run_id}/log", params={"offset": 7}).json()
    assert rest["text"] == "line 2\n"


def test_tasks_lists_running_then_queue(stand):
    a, b, c = (_enqueue(stand, name).json()["id"] for name in ("a", "b", "c"))
    anyio.run(stand.jobs.start, TaskRef(TaskKind.RUN, b))

    tasks = stand.client.get("/v1/tasks").json()

    assert [(t["id"], t["state"], t["position"]) for t in tasks] == [(b, "running", None), (a, "queued", 1),
                                                                     (c, "queued", 2)]


def test_repeat_from_snapshot(stand):
    run_id = _enqueue(stand, scenario_id="demo_run").json()["id"]

    status = stand.client.get(f"/v1/runs/{run_id}", params={"include": "scenario,summary,manifest"}).json()
    assert status["summary"] is None and status["manifest"] is None  # наполняет ragglet-30
    again = stand.client.post("/v1/runs", json={"scenario": status["scenario_yaml"], "scenario_id": "demo_run"})

    assert again.status_code == 202
    assert again.json()["id"] == f"{run_id}-2"
    bad = stand.client.get(f"/v1/runs/{run_id}", params={"include": "everything"})
    assert bad.status_code == 422 and _issues(bad) == [("request.include", "include", None)]

# -----------------------
# Сборки
# -----------------------


def test_build_check_enriches_targets(stand):
    stand.qdrant.collections[("http://localhost:6333", "chunks_v1")] = collection(index_meta(), points=9696)
    text = profile_yaml(storage={"endpoint": "http://localhost:6333", "collection": "chunks_v1"})

    response = stand.client.post("/v1/builds", json={"profile": text, "dry_run": True})

    assert response.status_code == 200, response.json()
    body = response.json()
    assert (body["profile"], body["corpus_sha256"]) == ("demo_build", DEMO_SHA)
    assert body["documents"] > 0
    [target] = body["targets"]
    assert (target["reachable"], target["exists"], target["points"]) == (True, True, 9696)
    assert target["built_at"] == "2026-10-20T12:00:00Z" and target["index_profile_id"] == "demo"
    assert "baseline_qdrant" in target["used_by"]


def test_build_check_unreachable_endpoint_is_data(stand):
    stand.qdrant.down.add(QDRANT)
    response = stand.client.post("/v1/builds", json={"profile": profile_yaml(), "dry_run": True})
    assert response.status_code == 200
    [target] = response.json()["targets"]
    assert (target["reachable"], target["exists"], target["error"]) == (False, None, "ConnectError: refused")


def test_build_check_broken_dataset(stand, tmp_path):
    text = profile_yaml(dataset={"id": str(tmp_path / "nope")})
    response = stand.client.post("/v1/builds", json={"profile": text, "dry_run": True})
    assert response.status_code == 422
    assert [(code, path) for code, path, _ in _issues(response)] == [("dataset.not_found", "dataset.id")]


def test_build_lifecycle(stand):
    response = stand.client.post("/v1/builds", json={"profile": profile_yaml(), "profile_id": "demo_build"})

    assert response.status_code == 202
    build_id = response.json()["id"]
    assert build_id == "demo_build_20261125T101540Z"
    ref = TaskRef(TaskKind.BUILD, build_id)
    anyio.run(stand.jobs.start, ref)
    stand.jobs.report(ref, "corpus", 2, 5)
    status = stand.client.get(f"/v1/builds/{build_id}", params={"include": "profile"}).json()
    assert (status["state"], status["phase"], status["profile_id"]) == ("running", "progress", "demo_build")
    assert status["targets"] == [{"endpoint": QDRANT, "collection": "chunks_v1"}]
    assert "demo_build" in status["profile_yaml"]

    anyio.run(lambda: stand.jobs.finish(ref, TaskState.COMPLETED, result={"documents": 5}))
    done = stand.client.get(f"/v1/builds/{build_id}").json()
    assert (done["state"], done["result"]) == ("completed", {"documents": 5})
    assert stand.client.post(f"/v1/builds/{build_id}/cancel").status_code == 409

    profiles = {p["id"]: p for p in stand.client.get("/v1/index-profiles").json()}
    assert "demo_build" not in profiles  # профиль не сохранён в библиотеке: ссылка в реестре всё равно есть
