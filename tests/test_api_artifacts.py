"""Файлы прогона: листинг, отдача как есть, zip, лог задачи из каталога после рестарта."""
from __future__ import annotations

import io
import zipfile

import pytest
from api_support import make_stand, profile_yaml, scenario_yaml
from fastapi.testclient import TestClient
from registry_support import RUN_ID, make_run_dir

from ragglet.api.app import create_app
from ragglet.jobs.memory import InMemoryJobManager
from ragglet.registry.files import FileRegistry, read_task_log

ARTIFACTS = [
    "manifest.json", "per_query.repeat0.csv", "per_query.repeat0.parquet", "per_query.repeat1.csv",
    "per_query.repeat1.parquet", "scenario.snapshot.json", "scenario.snapshot.yaml", "summary.aggregate.csv",
    "summary.aggregate.json", "summary.csv", "summary.json", "task.log",
]


@pytest.fixture
def stand(tmp_path, monkeypatch):
    return make_stand(tmp_path, monkeypatch, file_registry=True)


@pytest.fixture
def run_dir(stand):
    run_dir = make_run_dir(stand.settings.artifacts_dir)
    (run_dir / "task.log").write_text("started\n", encoding="utf-8")
    for service in ("task.json", "progress.json", "result.json", ".run.json.0123.tmp"):
        (run_dir / service).write_text("{}", encoding="utf-8")
    return run_dir


def test_listing_hides_service_files(stand, run_dir):
    body = stand.client.get(f"/v1/runs/{RUN_ID}/artifacts").json()

    assert [a["name"] for a in body] == ARTIFACTS
    sizes = {a["name"]: a["size"] for a in body}
    assert sizes["summary.json"] == (run_dir / "summary.json").stat().st_size
    assert body[0]["modified_at"].endswith("Z")


def test_file_is_served_as_is(stand, run_dir):
    summary = stand.client.get(f"/v1/runs/{RUN_ID}/artifacts/summary.json")
    assert summary.content == (run_dir / "summary.json").read_bytes()
    assert summary.headers["content-type"] == "application/json"
    parquet = stand.client.get(f"/v1/runs/{RUN_ID}/artifacts/per_query.repeat0.parquet")
    assert parquet.content == (run_dir / "per_query.repeat0.parquet").read_bytes()
    assert parquet.headers["content-type"] == "application/vnd.apache.parquet"

    missing = stand.client.get(f"/v1/runs/{RUN_ID}/artifacts/task.json")  # служебный — как будто нет
    assert (missing.status_code, missing.json()["issues"][0]["code"]) == (404, "artifact.not_found")
    assert stand.client.get(f"/v1/runs/{RUN_ID}/artifacts/..%2F..%2Fapi_support.py").status_code == 404
    assert stand.client.get("/v1/runs/nope_20261024T180917Z/artifacts").status_code == 404


def test_zip_holds_the_listing_under_run_id(stand, run_dir):
    response = stand.client.get(f"/v1/runs/{RUN_ID}/artifacts.zip")

    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == f'attachment; filename="{RUN_ID}.zip"'
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert sorted(archive.namelist()) == [f"{RUN_ID}/{name}" for name in ARTIFACTS]
    assert archive.read(f"{RUN_ID}/summary.json") == (run_dir / "summary.json").read_bytes()


def test_log_is_read_from_task_dir_after_restart(stand):
    run_id = stand.client.post("/v1/runs", json={"scenario": scenario_yaml()}).json()["id"]
    build_id = stand.client.post("/v1/builds", json={"profile": profile_yaml()}).json()["id"]
    text = "загрузка модели\nготово\n"
    (stand.settings.artifacts_dir / run_id / "task.log").write_text(text, encoding="utf-8")
    (stand.settings.builds_dir / build_id / "task.log").write_text("build log\n", encoding="utf-8")

    # прежний менеджер знает задачу: лог у него (у заглушки пустой)
    assert stand.client.get(f"/v1/runs/{run_id}/log").json()["text"] == ""

    registry = FileRegistry(stand.settings.artifacts_dir, stand.settings.builds_dir)
    restarted = create_app(stand.settings, jobs=InMemoryJobManager(registry), registry=registry)
    with TestClient(restarted) as client:
        first = client.get(f"/v1/runs/{run_id}/log").json()
        rest = client.get(f"/v1/runs/{run_id}/log", params={"offset": first["next_offset"]}).json()
        build = client.get(f"/v1/builds/{build_id}/log").json()

    assert (first["text"], first["complete"]) == (text, True)
    assert first["next_offset"] == len(text.encode("utf-8"))  # смещение в байтах
    assert (rest["text"], rest["complete"]) == ("", True)
    assert (build["text"], build["complete"]) == ("build log\n", True)


def test_task_log_chunks_do_not_split_utf8(tmp_path):
    path = tmp_path / "task.log"
    path.write_bytes("ok привет".encode())

    first = read_task_log(path, 0, 4)  # "ok " и первый байт «п»
    second = read_task_log(path, first.next_offset, 100)

    assert (first.text, first.next_offset, first.complete) == ("ok ", 3, False)
    assert (second.text, second.complete) == ("привет", True)
    assert read_task_log(tmp_path / "missing.log", 7, 100) == read_task_log(tmp_path / "missing.log", 7, 1)
    path.write_bytes(b"ok \xd0")  # оборванный последний символ не держит чтение на месте
    tail = read_task_log(path, 3, 100)
    assert (tail.text, tail.complete) == ("�", True)
