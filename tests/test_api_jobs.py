"""API целиком с ProcessJobManager и фиктивным воркером: прогон и сборка доходят до completed."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from api_support import make_stand, profile_yaml, scenario_yaml
from fastapi.testclient import TestClient

from ragglet.api.app import create_app
from ragglet.jobs import files
from ragglet.jobs.process import ProcessJobManager

FAKE = Path(__file__).with_name("jobs_fake_worker.py")


def poll(client: TestClient, url: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        body = client.get(url).json()
        if body["state"] not in ("queued", "running"):
            return body
        assert time.monotonic() < deadline, body
        time.sleep(0.05)


def test_run_and_build_complete_through_the_api(tmp_path, monkeypatch):
    stand = make_stand(tmp_path, monkeypatch)
    jobs = ProcessJobManager(stand.registry, cancel_grace_s=0.5, command=[sys.executable, str(FAKE)],
                             builds_dir=tmp_path / "builds")
    app = create_app(stand.settings, jobs=jobs, registry=stand.registry)

    with TestClient(app) as client:  # lifespan входит в менеджер
        run = client.post("/v1/runs", json={"scenario": scenario_yaml("demo_run")})
        build = client.post("/v1/builds", json={"profile": profile_yaml("demo_build")})
        assert (run.status_code, build.status_code) == (202, 202)
        assert run.json()["state"] == "running"  # стенд был свободен
        assert build.json()["position"] == 1

        run_status = poll(client, run.headers["location"])
        build_status = poll(client, build.headers["location"])
        log = client.get(f"{run.headers['location']}/log").json()
        tasks = client.get("/v1/tasks").json()

    assert run_status["state"] == "completed"
    assert run_status["progress"]["done"] == run_status["progress"]["total"] == 6  # repeats 2 × 3
    assert run_status["started_at"] and run_status["finished_at"]
    assert build_status["state"] == "completed"
    assert build_status["result"] == {"documents": 3}
    assert log["complete"] and log["text"].startswith(f"== run {run.json()['id']} (demo_run) started")
    assert tasks == []
    run_dir = stand.settings.artifacts_dir / run.json()["id"]
    assert files.read_outcome(run_dir).state.value == "completed"
