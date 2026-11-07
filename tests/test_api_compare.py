"""Сравнение прогонов через API: различия по классам, вердикт, прогоны без manifest, ошибки запроса."""
from __future__ import annotations

import json

import pytest
from api_support import make_stand
from registry_support import demo_scenario, make_run_dir

A, B, C = "demo_run_20261024T180917Z", "demo_run_20261024T181000Z", "demo_run_20261024T182000Z"


@pytest.fixture
def stand(tmp_path, monkeypatch):
    return make_stand(tmp_path, monkeypatch, file_registry=True)


def _compare(stand, *ids: str) -> dict:
    response = stand.client.get("/v1/runs/compare", params={"ids": ",".join(ids)})
    assert response.status_code == 200, response.json()
    return response.json()


def _edit_manifest(run_dir, edit):
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    edit(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_config_difference_only_is_comparable(stand):
    root = stand.settings.artifacts_dir
    make_run_dir(root, A)
    make_run_dir(root, B, cfg=demo_scenario(run={"repeats": 2, "shuffle": False}, retrieval={"top_k": 20}))

    body = _compare(stand, A, B)

    assert [run["id"] for run in body["runs"]] == [A, B]
    assert all(run["has_manifest"] and run["summary"]["repeats"] for run in body["runs"])
    assert body["differences"] == [{"kind": "config", "path": "scenario.retrieval.top_k", "values": [30, 20]}]
    assert (body["quality_comparable"], body["latency_comparable"], body["confounders"]) == (True, True, [])


def test_three_runs_with_data_and_env_differences(stand):
    root = stand.settings.artifacts_dir
    make_run_dir(root, A)
    _edit_manifest(make_run_dir(root, B), lambda m: m["dataset"].update(eval_sha256="e" * 64))
    _edit_manifest(make_run_dir(root, C), lambda m: m["env"].update(commit="f" * 40, system={"cpu": "x86"}))

    body = _compare(stand, A, B, C)

    kinds = {d["path"]: (d["kind"], d["values"]) for d in body["differences"]}
    assert kinds["dataset.eval_sha256"][0] == "data"
    assert kinds["dataset.eval_sha256"][1][1] == "e" * 64
    assert kinds["env.system"] == ("env", [None, None, {"cpu": "x86"}])
    assert (body["quality_comparable"], body["latency_comparable"]) == (False, False)
    assert body["confounders"] == ["env.commit"]


def test_run_without_manifest_compares_config_only(stand):
    root = stand.settings.artifacts_dir
    make_run_dir(root, A)
    make_run_dir(root, B, manifest=False, complete=False)

    body = _compare(stand, A, B)

    assert body["missing_manifest"] == [B]
    assert all(d["path"].startswith("scenario.") for d in body["differences"])
    assert (body["quality_comparable"], body["latency_comparable"]) == (None, None)
    assert body["runs"][1]["summary"] is None  # упал до summary


@pytest.mark.parametrize("ids", [A, f"{A},{A}", ",".join([A] * 7)])
def test_bad_ids_are_422(stand, ids):
    make_run_dir(stand.settings.artifacts_dir, A)
    response = stand.client.get("/v1/runs/compare", params={"ids": ids})
    assert response.status_code == 422
    assert response.json()["issues"][0]["code"] == "request.compare"


def test_unknown_run_is_404_and_route_is_not_shadowed(stand):
    make_run_dir(stand.settings.artifacts_dir, A)
    response = stand.client.get("/v1/runs/compare", params={"ids": f"{A},nope_20260101T000000Z"})
    assert response.status_code == 404
    assert response.json()["issues"][0]["code"] == "run.not_found"
    # /v1/runs/compare без ids — не прогон с id «compare»
    assert stand.client.get("/v1/runs/compare").status_code == 422
