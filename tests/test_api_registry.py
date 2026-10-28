"""API поверх FileRegistry: списки прогонов и сборок, деталь прогона, восстановление при старте."""
from __future__ import annotations

import json

import anyio
import pytest
from api_support import make_stand, profile_yaml, scenario_yaml
from fastapi.testclient import TestClient
from registry_support import RUN_ID, demo_scenario, make_feb_run_dir, make_run_dir

from ragglet.api.app import create_app
from ragglet.jobs.memory import InMemoryJobManager
from ragglet.jobs.protocol import TaskKind, TaskRef, TaskState
from ragglet.registry.files import FileRegistry

INCOMPLETE = "demo_run_20261024T180918Z"
FEB = "baseline_qdrant_20260221T135048Z"


@pytest.fixture
def stand(tmp_path, monkeypatch):
    return make_stand(tmp_path, monkeypatch, file_registry=True)


@pytest.fixture
def runs(stand):
    """Три импортированных каталога: полный, упавший до summary и февральский."""
    root = stand.settings.artifacts_dir
    make_run_dir(root)
    make_run_dir(root, INCOMPLETE, complete=False)
    make_feb_run_dir(root, FEB)
    return root


def _list(stand, **params) -> dict:
    response = stand.client.get("/v1/runs", params=params)
    assert response.status_code == 200, response.json()
    return response.json()


def _ids(body: dict) -> list[str]:
    return [item["id"] for item in body["items"]]


def _enqueue(stand, text: str | None = None, **body) -> str:
    response = stand.client.post("/v1/runs", json={"scenario": text or scenario_yaml(), **body})
    assert response.status_code == 202, response.json()
    return response.json()["id"]


def test_list_shows_summary_columns_without_recomputing(stand, runs):
    body = _list(stand, k=3)

    assert (body["k"], body["total"], _ids(body)) == (3, 3, [INCOMPLETE, RUN_ID, FEB])
    items = {item["id"]: item for item in body["items"]}
    done = items[RUN_ID]
    assert (done["origin"], done["state"], done["scenario"], done["scenario_id"]) == (
        "import", "completed", "demo_run", None)
    assert (done["started_at"], done["finished_at"], done["duration_s"]) == (None, None, None)
    assert done["tags"] == ["demo"]
    assert done["repeats"] == 2
    assert done["ndcg"] == pytest.approx((1 + 0.6309297535714575) / 3)
    assert done["recall"] == pytest.approx(2 / 3)
    assert (done["p50_ms"], done["timeout_rate"]) == (320.0, pytest.approx(1 / 3))
    assert done["qps"] > 0

    assert _list(stand, k=1)["items"][1]["ndcg"] == pytest.approx(1 / 3)
    assert _list(stand, k=10)["items"][1]["ndcg"] is None  # k=10 в прогоне не считали

    feb = items[FEB]  # recall@k_mean старого поколения — это нынешний hit@k: не переименовываем
    assert (feb["recall"], feb["ndcg"], feb["qps"], feb["timeout_rate"], feb["p50_ms"]) == (
        None, None, None, None, 12.0)
    failed = items[INCOMPLETE]
    assert (failed["state"], failed["ndcg"], failed["repeats"]) == ("failed", None, None)


def test_list_filters(stand, runs):
    assert _ids(_list(stand, state="completed")) == [RUN_ID, FEB]
    assert _ids(_list(stand, scenario="baseline_qdrant")) == [FEB]
    assert _ids(_list(stand, tag="demo")) == [INCOMPLETE, RUN_ID]  # тег сценария из snapshot
    assert _ids(_list(stand, collection="chunks_v1"))[:2] == [INCOMPLETE, RUN_ID]
    assert _ids(_list(stand, dataset="local:data/queries/baseline_qdrant.jsonl")) == [FEB]
    assert _ids(_list(stand, since="2026-10-24T18:09:18Z")) == [INCOMPLETE]
    assert _ids(_list(stand, until="2026-10-24T00:00:00Z")) == [FEB]
    assert _ids(_list(stand, q="BASELINE")) == [FEB]
    assert _list(stand, scenario_id="demo_run")["total"] == 0  # у импортированных ссылки нет

    page = _list(stand, sort="-ndcg", k=3, limit=2)
    assert (page["total"], _ids(page)) == (3, [RUN_ID, INCOMPLETE])  # null — в конце
    assert _ids(_list(stand, sort="p50"))[:2] == [FEB, RUN_ID]
    assert _ids(_list(stand, offset=2)) == [FEB]

    bad = stand.client.get("/v1/runs", params={"sort": "bogus"})
    assert bad.status_code == 422
    assert bad.json()["issues"][0]["code"] == "request.sort"


def test_dataset_filter_with_and_without_revision(stand, runs):
    label = anyio.run(stand.registry.get_run, RUN_ID).dataset

    assert RUN_ID in _ids(_list(stand, dataset=label))
    assert _ids(_list(stand, dataset=label + "@abc")) == []


def test_api_run_gets_record_in_its_dir(stand):
    run_id = _enqueue(stand, scenario_id="demo_run", tags=["night"])

    run_dir = stand.settings.artifacts_dir / run_id
    assert sorted(p.name for p in run_dir.iterdir()) == ["run.json"]
    item = _list(stand)["items"][0]
    assert (item["id"], item["origin"], item["state"], item["tags"]) == (run_id, "api", "queued", ["night"])
    status = stand.client.get(f"/v1/runs/{run_id}").json()
    assert (status["origin"], status["plan"]["total"]) == ("api", 200)  # 100 запросов demo × 2 повтора
    assert item["run_dir"] == status["run_dir"] == str(run_dir)  # путь, как его видит сервер


def test_scenario_file_follows_library(stand):
    text = scenario_yaml("demo_lib")
    assert stand.client.post("/v1/scenarios", json={"yaml": text}).status_code == 201
    run_id = _enqueue(stand, text, scenario_id="demo_lib")
    get = lambda: stand.client.get(f"/v1/runs/{run_id}").json()["scenario_file"]  # noqa: E731

    assert get() == "unchanged"
    etag = stand.client.get("/v1/scenarios/demo_lib").json()["etag"]
    edited = text.replace("repeats: 2", "repeats: 3")
    assert stand.client.put("/v1/scenarios/demo_lib", json={"yaml": edited, "etag": etag}).status_code == 200
    assert get() == "changed"
    assert stand.client.delete("/v1/scenarios/demo_lib").status_code == 204
    assert get() == "deleted"
    assert stand.client.get(f"/v1/runs/{_enqueue(stand)}").json()["scenario_file"] is None  # без ссылки


def test_repeat_from_snapshot_keeps_library_version(stand):
    text = scenario_yaml("demo_lib")
    assert stand.client.post("/v1/scenarios", json={"yaml": text}).status_code == 201
    first = _enqueue(stand, text, scenario_id="demo_lib")
    snapshot = stand.client.get(f"/v1/runs/{first}", params={"include": "scenario"}).json()["scenario_yaml"]
    assert snapshot != text

    # «Повторить» (ragglet-28): snapshot содержимым и та же ссылка на библиотеку
    repeat = _enqueue(stand, snapshot, scenario_id="demo_lib")
    unsaved = _enqueue(stand, text.replace("repeats: 2", "repeats: 3"), scenario_id="demo_lib")
    get = lambda run_id: stand.client.get(f"/v1/runs/{run_id}").json()["scenario_file"]  # noqa: E731

    assert (get(first), get(repeat), get(unsaved)) == ("unchanged", "unchanged", "changed")
    etag = stand.client.get("/v1/scenarios/demo_lib").json()["etag"]
    edited = text.replace("repeats: 2", "repeats: 4")
    assert stand.client.put("/v1/scenarios/demo_lib", json={"yaml": edited, "etag": etag}).status_code == 200
    assert (get(first), get(repeat)) == ("changed", "changed")


def test_include_fills_summary_and_manifest(stand, runs):
    body = stand.client.get(f"/v1/runs/{RUN_ID}", params={"include": "summary,manifest,scenario"}).json()

    assert (body["origin"], body["etag"], body["plan"], body["scenario_file"]) == ("import", None, None, None)
    assert [r["repeat_idx"] for r in body["summary"]["repeats"]] == [0, 1]
    assert body["summary"]["aggregate"]["lat_total_p50_mean_over_repeats"] == 320.0
    assert body["manifest"]["index"]["built_at"] == "2026-10-20T12:00:00Z"
    assert "name: demo_run" in body["scenario_yaml"]

    failed = stand.client.get(f"/v1/runs/{INCOMPLETE}", params={"include": "summary"}).json()
    assert failed["summary"] is None
    assert (failed["state"], failed["error"]["type"]) == ("failed", "RunIncomplete")


def test_last_run_brings_ndcg_at_10(stand):
    text = scenario_yaml("demo_lib")
    stand.client.post("/v1/scenarios", json={"yaml": text})
    run_id = _enqueue(stand, text, scenario_id="demo_lib")
    run_dir = stand.settings.artifacts_dir / run_id
    (run_dir / "summary.aggregate.json").write_text(
        json.dumps({"repeats": 1, "doc_ndcg@10_mean_mean_over_repeats": 0.315}), encoding="utf-8")
    ref = TaskRef(TaskKind.RUN, run_id)
    anyio.run(stand.jobs.start, ref)
    anyio.run(stand.jobs.finish, ref, TaskState.COMPLETED)

    items = {item["id"]: item for item in stand.client.get("/v1/scenarios").json()}
    assert items["demo_lib"]["last_run"]["ndcg_at_10"] == 0.315
    assert _list(stand)["items"][0]["ndcg"] == 0.315


def test_builds_list_and_dir(stand):
    response = stand.client.post("/v1/builds", json={"profile": profile_yaml(), "profile_id": "demo_build"})
    build_id = response.json()["id"]
    spec = stand.jobs._tasks[TaskRef(TaskKind.BUILD, build_id)].spec
    assert spec.run_dir == stand.settings.builds_dir / build_id  # каталог задачи сборки
    assert (spec.run_dir / "build.json").is_file()

    ref = TaskRef(TaskKind.BUILD, build_id)
    anyio.run(stand.jobs.start, ref)
    anyio.run(stand.jobs.finish, ref, TaskState.COMPLETED, None, {"documents": 50})

    body = stand.client.get("/v1/builds", params={"collection": "chunks_v1"}).json()
    item = body["items"][0]
    assert (body["total"], item["id"], item["profile"], item["state"], item["documents"]) == (
        1, build_id, "demo_build", "completed", 50)
    assert item["targets"] == [{"endpoint": "http://q:6333", "collection": "chunks_v1"}]
    build_dir = str(stand.settings.builds_dir / build_id)
    assert item["build_dir"] == stand.client.get(f"/v1/builds/{build_id}").json()["build_dir"] == build_dir
    assert stand.client.get("/v1/builds", params={"collection": "other"}).json()["total"] == 0
    assert stand.client.get("/v1/builds", params={"sort": "ndcg"}).status_code == 422


def test_restart_marks_orphans_failed(stand, tmp_path):
    run_id = _enqueue(stand)
    build_id = stand.client.post("/v1/builds", json={"profile": profile_yaml()}).json()["id"]

    # новый процесс сервера: те же каталоги, пустой менеджер; lifespan запускает recover
    registry = FileRegistry(stand.settings.artifacts_dir, stand.settings.builds_dir)
    app = create_app(stand.settings, jobs=InMemoryJobManager(registry), registry=registry)
    with TestClient(app) as client:
        run = client.get(f"/v1/runs/{run_id}").json()
        build = client.get(f"/v1/builds/{build_id}").json()

    assert (run["state"], run["error"]["type"]) == ("failed", "ServerRestart")
    assert (build["state"], build["error"]["type"]) == ("failed", "ServerRestart")
    assert run["finished_at"] is not None


def test_cfg_fixture_is_a_valid_scenario():
    assert demo_scenario().metrics.quality.k == [1, 3]


def test_process_manager_with_file_registry_as_in_serve(tmp_path, monkeypatch):
    """Как `ragglet serve`: менеджер задач ragglet-29 и реестр на файлах, затем рестарт сервера."""
    import sys
    import time
    from pathlib import Path

    from ragglet.jobs.process import ProcessJobManager

    stand = make_stand(tmp_path, monkeypatch, file_registry=True)
    fake = [sys.executable, str(Path(__file__).with_name("jobs_fake_worker.py"))]
    jobs = ProcessJobManager(stand.registry, cancel_grace_s=0.5, command=fake, builds_dir=stand.settings.builds_dir)

    def poll(client: TestClient, url: str) -> dict:
        deadline = time.monotonic() + 10
        while (body := client.get(url).json())["state"] in ("queued", "running"):
            assert time.monotonic() < deadline, body
            time.sleep(0.05)
        return body

    with TestClient(create_app(stand.settings, jobs=jobs, registry=stand.registry)) as client:
        run_id = client.post("/v1/runs", json={"scenario": scenario_yaml()}).json()["id"]
        build_id = client.post("/v1/builds", json={"profile": profile_yaml()}).json()["id"]
        assert poll(client, f"/v1/runs/{run_id}")["state"] == "completed"
        assert poll(client, f"/v1/builds/{build_id}")["state"] == "completed"

    run_dir = stand.settings.artifacts_dir / run_id
    assert json.loads((run_dir / "run.json").read_text())["state"] == "completed"
    assert json.loads((stand.settings.builds_dir / build_id / "build.json").read_text())["result"] == {"documents": 3}

    registry = FileRegistry(stand.settings.artifacts_dir, stand.settings.builds_dir)
    restarted = create_app(stand.settings, jobs=InMemoryJobManager(registry), registry=registry)
    with TestClient(restarted) as client:
        item = client.get("/v1/runs").json()["items"][0]
        log = client.get(f"/v1/runs/{run_id}/log").json()  # менеджер задачу уже не знает: task.log из каталога
        names = [a["name"] for a in client.get(f"/v1/runs/{run_id}/artifacts").json()]

    assert (item["id"], item["origin"], item["state"]) == (run_id, "api", "completed")  # recover не тронул
    assert log["complete"] and log["text"].startswith(f"== run {run_id}")
    assert "task.log" in names and "run.json" in names
    assert not {"task.json", "progress.json", "result.json"} & set(names)
