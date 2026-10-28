from __future__ import annotations

import time
from pathlib import Path

import anyio
from api_support import QDRANT, collection, index_meta, make_stand, scenario_yaml

from ragglet.api.settings import ServerSettings
from ragglet.config.scenario import ScenarioConfig


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def test_scenario_meta_adds_what_the_schema_lacks(stand):
    meta = stand.client.get("/v1/meta/scenario").json()

    assert "$defs" in meta["json_schema"]
    assert "concurrency" in meta["load_models"]["closed"]["properties"]
    assert meta["dataset_sources"] == [{"id": "local", "supports_revision": False},
                                       {"id": "hf", "supports_revision": True}]
    assert meta["storage_backends"] == ["qdrant"]
    assert meta["retrieval_strategies"] == ["single", "fanout_merge"]
    assert {"ndcg", "recall", "r_cap"} & set(meta["metrics"]["chunk"]) == set()
    assert {"ndcg", "recall", "r_cap", "hit", "mrr", "precision"} == set(meta["metrics"]["doc"])
    assert "merge" not in meta["latency_steps"] and "total" in meta["latency_steps"]
    defaults = meta["defaults"]
    assert defaults["metrics"]["quality"]["k"] == [1, 3, 5, 10]
    assert defaults["run"]["load"] == {"type": "closed", "concurrency": 1}
    assert defaults["retrieval"]["async"] is True

    filled = _merge(defaults, {
        "name": "x",
        "dataset": {"id": "datasets/demo"},
        "storage": {"backend": {"type": "qdrant", "endpoint": QDRANT, "collection": "c"}},
    })
    ScenarioConfig.model_validate(filled)  # умолчания — валидная основа пустого сценария


def test_index_profile_meta(stand):
    meta = stand.client.get("/v1/meta/index-profile").json()
    assert meta["defaults"]["chunking"]["method"] == "recursive"
    assert meta["storage_backends"] == ["qdrant"]


def test_stand_answers_when_services_are_down(stand, monkeypatch):
    from ragglet.api.routers import stand as stand_module

    stand.qdrant.down.add(QDRANT)
    monkeypatch.setattr(stand_module, "predicted_compute", lambda: {"device": "cpu", "accelerator": None,
                                                                    "torch_threads": 2})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    stand.client.put("/v1/scenarios/gen", json={"yaml": scenario_yaml(
        "gen", context={"enabled": True},
        generation={"enabled": True, "provider": "openai_compatible", "model": "m", "api_key_env": "MY_KEY"},
    )})
    monkeypatch.delenv("MY_KEY", raising=False)

    response = stand.client.get("/v1/stand")

    assert response.status_code == 200
    body = response.json()
    assert body["qdrant"] == [{"url": QDRANT, "available": False, "version": None,
                               "error": "ConnectError: refused", "collections": None}]
    assert body["redis"]["available"] is False
    assert body["compute"] == {"device": "cpu", "accelerator": None, "torch_threads": 2}
    assert body["env"] == {"MY_KEY": False, "OPENAI_API_KEY": True}
    assert "sk-secret" not in response.text
    assert (body["current"], body["queued"]) == (None, 0)
    assert {"platform", "cpus"} <= set(body["system"])


def test_stand_probe_respects_timeout(stand, monkeypatch):
    from ragglet.api import probes

    async def hangs(endpoint):
        await anyio.sleep(10)

    monkeypatch.setattr(probes, "list_collections", hangs)
    stand.settings.probe_timeout_s = 0.2
    started = time.monotonic()
    body = stand.client.get("/v1/stand").json()
    assert time.monotonic() - started < 2
    assert body["qdrant"][0]["error"] == "TimeoutError: no answer in 0.2 s"


def test_stand_shows_current_task_and_queue(stand):
    first = stand.client.post("/v1/runs", json={"scenario": scenario_yaml("a")}).json()["id"]
    stand.client.post("/v1/runs", json={"scenario": scenario_yaml("b")})
    from ragglet.jobs.protocol import TaskKind, TaskRef

    anyio.run(stand.jobs.start, TaskRef(TaskKind.RUN, first))

    body = stand.client.get("/v1/stand").json()
    assert body["current"]["id"] == first and body["current"]["state"] == "running"
    assert body["queued"] == 1


def test_collections_get_states_and_library_users(stand):
    meta = index_meta()
    stand.qdrant.collections.update({
        ("http://localhost:6333", "chunks_v1"): collection(meta),
        ("http://localhost:6333", "nfcorpus_v1"): collection(meta, status="yellow"),
        ("http://localhost:6333", "half"): collection({"ragglet": {"built_at": "x"}}),
        ("http://localhost:6333", "named"): collection(None, params=None),
        ("http://localhost:6333", "other_app"): collection({"owner": "someone"}),
    })
    stand.settings.qdrant_endpoints = ["http://localhost:6333/", "http://down:6333"]
    stand.qdrant.down.add("http://down:6333")

    body = stand.client.get("/v1/indexes").json()

    states = {c["name"]: c["state"] for c in body["collections"]}
    assert states == {"chunks_v1": "ready", "nfcorpus_v1": "optimizing", "half": "incomplete",
                      "named": "foreign", "other_app": "foreign"}
    by_name = {c["name"]: c for c in body["collections"]}
    assert "baseline_qdrant" in by_name["chunks_v1"]["used_by"]
    assert by_name["nfcorpus_v1"]["used_by"] == ["nfcorpus_hf"]
    assert by_name["chunks_v1"]["profiles"] == ["demo"]
    assert by_name["chunks_v1"]["ragglet"]["index_profile_id"] == "demo"
    assert by_name["half"]["ragglet"] is None
    assert [e["available"] for e in body["endpoints"]] == [True, False]


def test_datasets_lists_local_dirs_and_hf_ids(stand):
    stand.qdrant.collections[(QDRANT, "nf")] = collection(index_meta(dataset_source="hf", dataset_id="BeIR/scifact"))

    body = stand.client.get("/v1/datasets").json()

    local = {d["id"]: d for d in body["local"]}
    demo = str(Path(stand.settings.dataset_roots[0]) / "demo")
    assert local[demo]["splits"] == ["test"]
    hf = {d["id"]: d["seen_in"] for d in body["hf"]}
    assert hf["BeIR/nfcorpus"] == ["profile:nfcorpus_hf", "scenario:nfcorpus_hf"]
    assert hf["BeIR/scifact"] == [f"collection:{QDRANT}/nf"]


def test_settings_from_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAGGLET_QDRANT_ENDPOINTS", "http://a:6333, http://b:6334")
    monkeypatch.setenv("RAGGLET_PORT", "9000")
    monkeypatch.setenv("RAGGLET_REDIS_URL", "")
    (tmp_path / ".env").write_text("RAGGLET_HOST=0.0.0.0\n", encoding="utf-8")

    settings = ServerSettings()
    assert settings.qdrant_endpoints == ["http://a:6333", "http://b:6334"]
    assert (settings.host, settings.port, settings.redis_url) == ("0.0.0.0", 9000, None)
    assert ServerSettings(port=8100).port == 8100  # флаг ragglet serve важнее переменной


def test_server_passes_flags_to_uvicorn(monkeypatch, tmp_path, capsys):
    import uvicorn

    from ragglet import server

    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda application, **kw: calls.append(kw))

    server.main(["--port", "8123"])

    assert calls == [{"host": "127.0.0.1", "port": 8123, "workers": 1}]
    assert "listening: http://127.0.0.1:8123" in capsys.readouterr().out


def test_ui_is_served_with_spa_fallback(tmp_path, monkeypatch):
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<html>ragglet</html>", encoding="utf-8")
    (ui / "app.js").write_text("console.log(1)", encoding="utf-8")
    client = make_stand(tmp_path, monkeypatch, ui_dir=ui).client

    assert client.get("/app.js").text == "console.log(1)"
    assert client.get("/runs/x?tab=queries").text == "<html>ragglet</html>"
    missing = client.get("/v1/nope")
    assert missing.status_code == 404 and missing.json()["issues"][0]["code"] == "http.not_found"
    assert client.get("/v1/scenarios").status_code == 200


def test_root_without_ui_points_to_api(stand):
    assert stand.client.get("/").json() == {"api": "/v1", "docs": "/v1/docs"}
    assert stand.client.get("/v1/openapi.json").json()["info"]["title"] == "Ragglet"
