"""
Сквозная проверка стенда через API (ragglet-35): настоящие `ProcessJobManager` и `FileRegistry`
во временных каталогах, настоящий воркер и ядро. Индекс `demo` собирается через API во временную
коллекцию, прогоны — сценарий `baseline_qdrant` по ней. Нужны Qdrant на localhost:6333, Redis на
localhost:6379 и модель all-MiniLM-L6-v2 (скачается с Hub при первом запуске); без сервисов тесты
пропускаются. Идут около трёх минут: каждый воркер импортирует torch и грузит модель.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ragglet.api.app import create_app
from ragglet.api.settings import ServerSettings
from ragglet.jobs.process import ProcessJobManager
from ragglet.registry.files import FileRegistry

ROOT = Path(__file__).resolve().parents[1]
QDRANT = "http://localhost:6333"
ACTIVE = ("queued", "running")
PHASES = ["preparing", "progress", "finishing"]


def _up(port: int) -> bool:
    try:
        socket.create_connection(("localhost", port), timeout=1).close()
        return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (_up(6333) and _up(6379)), reason="Qdrant (6333) or Redis (6379) is not running"),
]


@dataclass
class E2E:
    """Стенд во временном каталоге: своя библиотека с временной коллекцией, свои прогоны и сборки."""

    root: Path
    collection: str
    settings: ServerSettings
    build: dict = field(default_factory=dict)  # ответы сборки из фикстуры: их проверяет test_build_index

    @property
    def scenario(self) -> str:
        return (self.settings.scenarios_dir / "baseline_qdrant.yaml").read_text(encoding="utf-8")

    @property
    def profile(self) -> str:
        return (self.settings.index_profiles_dir / "demo.yaml").read_text(encoding="utf-8")

    def app(self):
        """Новый процесс сервера, как `ragglet serve`: пустой менеджер, реестр сканирует каталоги."""
        registry = FileRegistry(self.settings.artifacts_dir, self.settings.builds_dir)
        jobs = ProcessJobManager(registry, cancel_grace_s=5, builds_dir=self.settings.builds_dir)
        return create_app(self.settings, jobs=jobs, registry=registry)


def _with_collection(source: Path, target: Path, collection: str, **replace: str) -> None:
    text = source.read_text(encoding="utf-8")
    for old, new in {'collection: "chunks_v1"': f'collection: "{collection}"', **replace}.items():
        assert text.count(old) == 1, (source, old)
        text = text.replace(old, new)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _poll(get: Callable[[], dict], timeout: float = 300) -> tuple[dict, list[str]]:
    """Опрос раз в 0,2 с до конца задачи; фазы — в том порядке, в каком их видел опрос."""
    deadline = time.monotonic() + timeout
    phases: list[str] = []
    while (body := get())["state"] in ACTIVE:
        if body["phase"] and body["phase"] not in phases:
            phases.append(body["phase"])
        assert time.monotonic() < deadline, body
        time.sleep(0.2)
    return body, phases


def _wait_progress(get: Callable[[], dict], above: int, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while not ((body := get())["progress"] and body["progress"]["done"] > above):
        assert body["state"] in ACTIVE and time.monotonic() < deadline, body
        time.sleep(0.2)
    return body


def _json(response: httpx.Response, status: int = 200):
    assert response.status_code == status, response.text
    return response.json()


@pytest.fixture(scope="module")
def e2e(tmp_path_factory) -> Iterator[E2E]:
    from qdrant_client import QdrantClient

    root = tmp_path_factory.mktemp("e2e")
    collection = f"ragglet_e2e_{uuid.uuid4().hex[:8]}"
    _with_collection(ROOT / "configs/scenarios/baseline_qdrant.yaml", root / "scenarios/baseline_qdrant.yaml",
                     collection)
    _with_collection(ROOT / "configs/indexes/demo.yaml", root / "indexes/demo.yaml", collection)
    settings = ServerSettings(
        _env_file=None,
        scenarios_dir=root / "scenarios",
        index_profiles_dir=root / "indexes",
        artifacts_dir=root / "runs",
        builds_dir=root / "builds",
        dataset_roots=[ROOT / "datasets"],
        qdrant_endpoints=[QDRANT],
        redis_url="redis://localhost:6379/0",
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)  # относительный datasets/demo — как у сервера, запущенного из корня
        stand = E2E(root=root, collection=collection, settings=settings)
        try:
            with TestClient(stand.app()) as client:
                _build(stand, client)
            yield stand
        finally:
            QdrantClient(QDRANT).delete_collection(collection)


def _build(stand: E2E, client: TestClient) -> None:
    """Пункт 2: сборка индекса через API; ответы сохраняются, проверяет их test_build_index."""
    stand.build["check"] = client.post("/v1/builds", json={"profile": stand.profile, "dry_run": True})
    accepted = client.post("/v1/builds", json={"profile": stand.profile, "profile_id": "demo"})
    stand.build["accepted"] = accepted
    build_id = _json(accepted, 202)["id"]
    stand.build["status"], stand.build["phases"] = _poll(lambda: client.get(f"/v1/builds/{build_id}").json())
    stand.build["indexes"] = _json(client.get("/v1/indexes"))
    stand.build["list"] = _json(client.get("/v1/builds"))
    stand.build["log"] = _json(client.get(f"/v1/builds/{build_id}/log"))


def test_stand_sees_services(e2e):
    with TestClient(e2e.app()) as client:
        stand = _json(client.get("/v1/stand"))

    assert [q["available"] for q in stand["qdrant"]] == [True]
    assert stand["redis"]["available"] and stand["redis"]["version"]
    assert stand["compute"]["device"] and stand["commit"]
    assert (stand["current"], stand["queued"]) == (None, 0)


def test_build_index(e2e):
    check = _json(e2e.build["check"])
    assert (check["profile"], check["documents"]) == ("demo", 50)
    assert [(t["collection"], t["reachable"]) for t in check["targets"]] == [(e2e.collection, True)]

    accepted = _json(e2e.build["accepted"], 202)
    assert e2e.build["accepted"].headers["location"] == f"/v1/builds/{accepted['id']}"
    status = e2e.build["status"]
    assert status["state"] == "completed", e2e.build["log"]["text"]
    assert e2e.build["phases"][0] == "preparing"
    assert status["progress"]["stage"] == "corpus"
    assert status["progress"]["done"] == status["progress"]["total"] == 50
    assert status["result"]["documents"] == 50

    collection = next(c for c in e2e.build["indexes"]["collections"] if c["name"] == e2e.collection)
    assert (collection["state"], collection["profiles"]) == ("ready", ["demo"])
    assert collection["ragglet"]["index_profile_id"] == "demo"
    assert collection["ragglet"]["built_at"] == status["result"]["index"]["built_at"]
    assert [(b["id"], b["state"], b["documents"]) for b in e2e.build["list"]["items"]] == [
        (accepted["id"], "completed", 50)]
    assert e2e.build["log"]["text"].startswith(f"== build {accepted['id']}")


def test_run_queue_cancel_and_results(e2e):
    bad = e2e.scenario.replace("k: [1, 3, 5, 10]", "k: [50]")
    with TestClient(e2e.app()) as client:
        check = _json(client.post("/v1/runs", json={"scenario": e2e.scenario, "dry_run": True}))
        invalid = _json(client.post("/v1/runs", json={"scenario": bad, "dry_run": True}), 422)

        accepted = client.post("/v1/runs", json={"scenario": e2e.scenario, "scenario_id": "baseline_qdrant"})
        a = _json(accepted, 202)
        b = _json(client.post("/v1/runs", json={"scenario": e2e.scenario, "scenario_id": "baseline_qdrant"}), 202)
        tasks = _json(client.get("/v1/tasks"))
        cancelled = _json(client.post(f"/v1/runs/{b['id']}/cancel"))
        again = client.post(f"/v1/runs/{b['id']}/cancel")
        status, phases = _poll(lambda: client.get(f"/v1/runs/{a['id']}").json())

        full = _json(client.get(f"/v1/runs/{a['id']}", params={"include": "summary,manifest,scenario"}))
        listed = {item["id"]: item for item in _json(client.get("/v1/runs"))["items"]}
        queries = _json(client.get(f"/v1/runs/{a['id']}/queries", params={"outcome": "ok"}))
        qid = queries["items"][0]["qid"]
        detail = _json(client.get(f"/v1/runs/{a['id']}/queries/{qid}"))
        artifacts = [x["name"] for x in _json(client.get(f"/v1/runs/{a['id']}/artifacts"))]
        summary = client.get(f"/v1/runs/{a['id']}/artifacts/summary.json")
        archive = client.get(f"/v1/runs/{a['id']}/artifacts.zip")
        log = _json(client.get(f"/v1/runs/{a['id']}/log", params={"offset": 0}))
        tail = _json(client.get(f"/v1/runs/{a['id']}/log", params={"offset": log["next_offset"]}))

    run_dir = e2e.settings.artifacts_dir / a["id"]
    # dry-run: отчёт preflight и план; испорченный конфиг — 422 с полем
    assert (check["dataset"]["queries"], check["plan"]["total"]) == (100, 200)
    assert check["index"]["index_profile_id"] == "demo"
    assert [(i["code"], i["path"]) for i in invalid["issues"]] == [("config.value_error", "metrics.quality.k")]

    # очередь: A на стенде, B ждёт; B убрана из очереди
    assert accepted.headers["location"] == f"/v1/runs/{a['id']}"
    assert (b["state"], b["position"]) == ("queued", 1)
    assert [(t["id"], t["state"], t["position"]) for t in tasks] == [
        (a["id"], "running", None), (b["id"], "queued", 1)]
    assert (cancelled["state"], cancelled["position"]) == ("cancelled", None)
    assert (again.status_code, again.json()["issues"][0]["code"]) == (409, "task.finished")

    # A: фазы по порядку, итог, тяжёлые поля
    assert status["state"] == "completed", log["text"]
    assert phases[0] == "preparing" and "progress" in phases
    assert phases == [p for p in PHASES if p in phases]
    assert status["progress"]["done"] == status["progress"]["total"] == 200
    aggregate = full["summary"]["aggregate"]
    assert aggregate["doc_ndcg@10_mean_mean_over_repeats"] > 0.5
    assert aggregate["lat_total_p50_mean_over_repeats"] > 0
    assert aggregate["throughput_qps_mean_over_repeats"] > 0
    assert len(full["summary"]["repeats"]) == 2
    assert full["manifest"]["index"]["profile_id"] == "demo"
    assert f'collection: {e2e.collection}' in full["scenario_yaml"]
    assert (full["origin"], full["scenario_file"]) == ("api", "unchanged")

    # список: колонки сводки у A, B отменена
    assert listed[a["id"]]["state"] == "completed"
    assert listed[a["id"]]["ndcg"] == pytest.approx(aggregate["doc_ndcg@10_mean_mean_over_repeats"])
    assert None not in (listed[a["id"]]["p50_ms"], listed[a["id"]]["qps"], listed[a["id"]]["duration_s"])
    assert listed[b["id"]]["state"] == "cancelled"

    # per-query и деталь запроса: оценки из датасета, тексты чанков из Qdrant
    assert (queries["repeats"], queries["total"]) == ([0, 1], 100)
    assert detail["grades"]["available"] and detail["texts"]["available"], detail
    assert detail["truth"] and all("grade" in t for t in detail["truth"])
    assert detail["results"][0]["text"]

    # файлы как есть, zip, лог
    assert "task.log" in artifacts and "task.json" not in artifacts and "summary.json" in artifacts
    assert summary.content == (run_dir / "summary.json").read_bytes()
    names = zipfile.ZipFile(io.BytesIO(archive.content)).namelist()
    assert sorted(names) == sorted(f"{a['id']}/{name}" for name in artifacts)
    assert log["text"].startswith(f"== run {a['id']}") and "completed" in log["text"]
    assert (tail["text"], tail["complete"]) == ("", True)


def test_cancel_running_run(e2e):
    scenario = e2e.scenario.replace("repeats: 2", "repeats: 5")
    with TestClient(e2e.app()) as client:
        run_id = _json(client.post("/v1/runs", json={"scenario": scenario}), 202)["id"]
        get = lambda: client.get(f"/v1/runs/{run_id}").json()  # noqa: E731
        _wait_progress(get, above=100)  # первый повтор записан
        answer = _json(client.post(f"/v1/runs/{run_id}/cancel"))
        status, _ = _poll(get, timeout=30)
        summary = _json(client.get(f"/v1/runs/{run_id}", params={"include": "summary"}))["summary"]

    run_dir = e2e.settings.artifacts_dir / run_id
    assert answer["state"] in ACTIVE + ("cancelled",)  # running, пока процесс не вышел
    assert status["state"] == "cancelled", (run_dir / "task.log").read_text()
    assert 100 < status["progress"]["done"] < status["progress"]["total"] == 500
    assert (run_dir / "scenario.snapshot.json").is_file()
    assert (run_dir / "per_query.repeat0.csv").is_file()
    assert summary is None and not (run_dir / "summary.json").exists()


def test_shutdown_during_run_marks_failed(e2e):
    scenario = e2e.scenario.replace("repeats: 2", "repeats: 30")
    with TestClient(e2e.app()) as client:
        running = _json(client.post("/v1/runs", json={"scenario": scenario}), 202)["id"]
        queued = _json(client.post("/v1/runs", json={"scenario": scenario}), 202)["id"]
        _wait_progress(lambda: client.get(f"/v1/runs/{running}").json(), above=0)
    # выход из lifespan — штатная остановка сервера; дальше — новый сервер на тех же каталогах
    with TestClient(e2e.app()) as client:
        first = _json(client.get(f"/v1/runs/{running}"))
        second = _json(client.get(f"/v1/runs/{queued}"))

    assert (first["state"], first["error"]["type"]) == ("failed", "ServerShutdown")
    assert first["progress"]["done"] > 0
    assert (second["state"], second["error"]["type"], second["progress"]) == ("failed", "ServerShutdown", None)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_server_crash_then_restart(e2e):
    """`python -m ragglet.server` настоящим процессом, SIGKILL во время прогона, затем новый сервер."""
    port = _free_port()
    env = os.environ | {
        "RAGGLET_PORT": str(port),
        "RAGGLET_SCENARIOS_DIR": str(e2e.settings.scenarios_dir),
        "RAGGLET_INDEX_PROFILES_DIR": str(e2e.settings.index_profiles_dir),
        "RAGGLET_ARTIFACTS_DIR": str(e2e.settings.artifacts_dir),
        "RAGGLET_BUILDS_DIR": str(e2e.settings.builds_dir),
    }
    server_log = e2e.root / f"serve_{port}.log"
    scenario = e2e.scenario.replace("repeats: 2", "repeats: 30")
    with server_log.open("wb") as out:
        server = subprocess.Popen([sys.executable, "-m", "ragglet.server"],
                                  cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}/v1", timeout=30) as api:
            deadline = time.monotonic() + 60
            while True:
                try:
                    api.get("/stand")
                    break
                except httpx.TransportError:
                    assert server.poll() is None and time.monotonic() < deadline, server_log.read_text()
                    time.sleep(0.5)
            run_id = _json(api.post("/runs", json={"scenario": scenario}), 202)["id"]
            _wait_progress(lambda: api.get(f"/runs/{run_id}").json(), above=0)
    finally:
        server.send_signal(signal.SIGKILL)
        server.wait()

    run_dir = e2e.settings.artifacts_dir / run_id
    header = (run_dir / "task.log").read_text().splitlines()[0]
    worker = int(header.rsplit("pid ", 1)[1].rstrip(" ="))
    deadline = time.monotonic() + 15  # сторожевой поток воркера: смена родителя → отмена, 12 с максимум
    while _alive(worker):
        assert time.monotonic() < deadline, f"worker {worker} outlived the server"
        time.sleep(0.5)

    with TestClient(e2e.app()) as client:
        status = _json(client.get(f"/v1/runs/{run_id}"))
        log = _json(client.get(f"/v1/runs/{run_id}/log"))
        listed = [item["id"] for item in _json(client.get("/v1/runs"))["items"]]
        tasks = _json(client.get("/v1/tasks"))
        repeat = client.post(f"/v1/runs/{run_id}/cancel")

    assert (status["state"], status["error"]["type"]) == ("failed", "ServerRestart")
    # последний прогресс воркера: итог, который записал его сторожевой поток
    outcome = json.loads((run_dir / "result.json").read_text())
    assert outcome["state"] == "cancelled"
    assert status["progress"]["done"] == outcome["final_progress"]["done"] > 0
    assert log["complete"] and log["text"].startswith(f"== run {run_id}") and "parent process" in log["text"]
    assert run_id in listed and tasks == []
    assert repeat.status_code == 409


def test_client_start_run_stop(e2e, monkeypatch, tmp_path, capsys):
    """Клиент в корне (ragglet-37): start поднимает настоящий сервер, run идёт через API, stop его гасит."""
    loader = importlib.machinery.SourceFileLoader("ragglet_client", str(ROOT / "ragglet"))
    cli = importlib.util.module_from_spec(importlib.util.spec_from_loader("ragglet_client", loader))
    loader.exec_module(cli)
    for name, value in {
        "RAGGLET_SCENARIOS_DIR": e2e.settings.scenarios_dir, "RAGGLET_INDEX_PROFILES_DIR": e2e.settings.index_profiles_dir,
        "RAGGLET_ARTIFACTS_DIR": e2e.settings.artifacts_dir, "RAGGLET_BUILDS_DIR": e2e.settings.builds_dir,
    }.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.delenv("RAGGLET_API_URL", raising=False)
    monkeypatch.setattr(cli, "RESULTS", tmp_path)  # server.json и лог сервера — не в results/ репозитория
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(e2e.scenario, encoding="utf-8")

    try:
        assert cli.main(["start", "--port", str(_free_port()), "--no-services"]) == 0
        server = cli.read_state()["pid"]
        assert cli.main(["run", str(scenario), "--tag", "client"]) == 0, capsys.readouterr().err
        out = capsys.readouterr().out
        assert cli.main(["runs", "list", "--tag", "client", "--json"]) == 0
        listed = json.loads(capsys.readouterr().out)["items"]
    finally:
        assert cli.main(["stop"]) == 0

    run_dir = Path(out.split("Artifacts: ", 1)[1].split(" (on the server)", 1)[0])
    (item,) = listed
    assert (item["origin"], item["state"], item["tags"], item["run_dir"]) == ("api", "completed",
                                                                             ["client"], str(run_dir))
    assert None not in (item["ndcg"], item["recall"], item["p50_ms"], item["qps"])
    assert (run_dir / "run.json").is_file() and (run_dir / "summary.aggregate.json").is_file()
    assert not cli.alive(server) and cli.read_state() == {}


def test_repeat_from_snapshot(e2e):
    with TestClient(e2e.app()) as client:
        body = {"scenario": e2e.scenario, "scenario_id": "baseline_qdrant", "tags": ["e2e"]}
        first = _json(client.post("/v1/runs", json=body), 202)["id"]
        assert _poll(lambda: client.get(f"/v1/runs/{first}").json())[0]["state"] == "completed"
        original = _json(client.get(f"/v1/runs/{first}", params={"include": "scenario"}))

        # «Повторить»: snapshot содержимым и та же ссылка на библиотеку
        repeat = {"scenario": original["scenario_yaml"], "scenario_id": original["scenario_id"]}
        second = _json(client.post("/v1/runs", json=repeat), 202)["id"]
        status, _ = _poll(lambda: client.get(f"/v1/runs/{second}").json())
        again = _json(client.get(f"/v1/runs/{second}", params={"include": "scenario,summary"}))

    assert status["state"] == "completed"
    assert (again["scenario_id"], again["scenario_file"]) == ("baseline_qdrant", "unchanged")
    assert again["scenario_yaml"] == original["scenario_yaml"]
    assert again["summary"]["aggregate"]["doc_ndcg@10_mean_mean_over_repeats"] > 0.5
    assert json.loads((e2e.settings.artifacts_dir / second / "run.json").read_text())["scenario_id"] == (
        "baseline_qdrant")
