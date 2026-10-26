from __future__ import annotations

import os

import pytest
import yaml
from api_support import profile_yaml, scenario_yaml


def _issues(response) -> list[tuple[str, str | None]]:
    return [(i["code"], i["path"]) for i in response.json()["issues"]]


def test_list_shows_valid_scenarios_drafts_and_renamed_files(stand):
    (stand.settings.scenarios_dir / "draft.yaml").write_text("name: draft\nstorage: {}\n", encoding="utf-8")
    (stand.settings.scenarios_dir / "renamed.yaml").write_text(scenario_yaml("other"), encoding="utf-8")

    items = {item["id"]: item for item in stand.client.get("/v1/scenarios").json()}

    nfcorpus = items["nfcorpus_hf"]
    assert nfcorpus["valid"] is True
    assert nfcorpus["dataset"] == "hf:BeIR/nfcorpus"
    assert nfcorpus["collections"] == [{"endpoint": "http://localhost:6333", "collection": "nfcorpus_v1"}]
    assert nfcorpus["pipeline"] == {"strategy": "single", "sources": 1, "rerank": False, "context": False,
                                    "generation": False, "cache": False, "concurrency": 1, "repeats": 1}
    assert items["draft"]["valid"] is False and items["draft"]["pipeline"] is None
    assert items["renamed"]["valid"] is False


def test_get_returns_file_text_as_is(stand):
    original = (stand.settings.scenarios_dir / "nfcorpus_hf.yaml").read_text(encoding="utf-8")
    doc = stand.client.get("/v1/scenarios/nfcorpus_hf").json()
    assert doc["yaml"] == original  # комментарии на месте
    assert doc["valid"] is True and doc["problem"] is None and len(doc["etag"]) == 16


def test_create_from_yaml_and_conflict_on_existing_id(stand):
    response = stand.client.post("/v1/scenarios", json={"yaml": scenario_yaml("fresh")})
    assert response.status_code == 201
    assert response.json()["id"] == "fresh"
    assert (stand.settings.scenarios_dir / "fresh.yaml").exists()

    again = stand.client.post("/v1/scenarios", json={"yaml": scenario_yaml("fresh")})
    assert again.status_code == 409
    assert _issues(again) == [("library.conflict", None)]


def test_put_detects_changes_made_on_disk(stand):
    doc = stand.client.get("/v1/scenarios/nfcorpus_hf").json()
    path = stand.settings.scenarios_dir / "nfcorpus_hf.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "# edited in IDE\n", encoding="utf-8")

    stale = stand.client.put("/v1/scenarios/nfcorpus_hf", json={"yaml": doc["yaml"], "etag": doc["etag"]})
    assert stale.status_code == 409

    fresh = stand.client.get("/v1/scenarios/nfcorpus_hf").json()
    saved = stand.client.put("/v1/scenarios/nfcorpus_hf", json={"yaml": doc["yaml"], "etag": fresh["etag"]})
    assert saved.status_code == 200
    assert path.read_text(encoding="utf-8") == doc["yaml"]


def test_put_requires_name_to_match_id(stand):
    doc = stand.client.get("/v1/scenarios/nfcorpus_hf").json()
    response = stand.client.put("/v1/scenarios/nfcorpus_hf", json={"yaml": scenario_yaml("other"), "etag": doc["etag"]})
    assert response.status_code == 422
    assert _issues(response) == [("library.name_mismatch", "name")]


def test_schema_invalid_text_is_saved_as_draft(stand):
    response = stand.client.put("/v1/scenarios/half", json={"yaml": "name: half\nrerank: {enabled: true}\n"})
    assert response.status_code == 201
    doc = response.json()
    assert doc["valid"] is False
    assert ("config.missing", "dataset") in _issues_from(doc)
    assert (stand.settings.scenarios_dir / "half.yaml").exists()


def _issues_from(doc) -> list[tuple[str, str | None]]:
    return [(i["code"], i["path"]) for i in doc["problem"]["issues"]]


def test_broken_yaml_is_not_saved(stand):
    path = stand.settings.scenarios_dir / "nfcorpus_hf.yaml"
    before = path.read_bytes()
    doc = stand.client.get("/v1/scenarios/nfcorpus_hf").json()

    response = stand.client.put("/v1/scenarios/nfcorpus_hf", json={"yaml": "name: [oops", "etag": doc["etag"]})

    assert response.status_code == 422
    assert _issues(response) == [("config.yaml", None)]
    assert path.read_bytes() == before


def test_delete_then_not_found(stand):
    doc = stand.client.get("/v1/scenarios/s0_baseline_rag").json()
    assert stand.client.delete("/v1/scenarios/s0_baseline_rag", params={"etag": "0" * 16}).status_code == 409
    assert stand.client.delete("/v1/scenarios/s0_baseline_rag", params={"etag": doc["etag"]}).status_code == 204
    response = stand.client.get("/v1/scenarios/s0_baseline_rag")
    assert response.status_code == 404
    assert _issues(response) == [("library.not_found", None)]


@pytest.mark.parametrize("bad_id", ["bad.name", "..%2Fsecret"])
def test_ids_outside_the_pattern_never_reach_the_disk(stand, bad_id):
    assert stand.client.get(f"/v1/scenarios/{bad_id}").status_code in (404, 422)


def test_writes_leave_no_temporary_files(stand):
    stand.client.put("/v1/scenarios/clean", json={"yaml": scenario_yaml("clean")})
    assert not [name for name in os.listdir(stand.settings.scenarios_dir) if name.endswith(".tmp")]


def _enqueue(stand, text: str) -> str:
    response = stand.client.post("/v1/runs", json={"scenario": text, "scenario_id": "demo_run"})
    assert response.status_code == 202, response.json()
    return response.json()["id"]


def test_create_from_run_snapshot(stand):
    run_id = _enqueue(stand, scenario_yaml("demo_run", retrieval={"async": False}))

    response = stand.client.post("/v1/scenarios", json={"from_run": run_id, "name": "demo_copy"})

    assert response.status_code == 201
    data = yaml.safe_load(response.json()["yaml"])
    assert data["name"] == "demo_copy"
    assert data["retrieval"]["async"] is False  # алиас, как в YAML руками
    assert "artifacts_dir" not in data["run"]  # серверный путь не переносится
    assert response.json()["valid"] is True

    items = {item["id"]: item for item in stand.client.get("/v1/scenarios").json()}
    assert items["demo_copy"]["last_run"] is None


def test_create_requires_exactly_one_source(stand):
    response = stand.client.post("/v1/scenarios", json={})
    assert response.status_code == 422
    assert _issues(response) == [("request.value_error", "yaml")]
    response = stand.client.post("/v1/scenarios", json={"from_run": "x"})
    assert _issues(response) == [("request.value_error", "name")]


def test_scenario_list_shows_last_run_by_library_ref(stand):
    stand.client.put("/v1/scenarios/demo_run", json={"yaml": scenario_yaml("demo_run")})
    run_id = _enqueue(stand, scenario_yaml("demo_run"))
    items = {item["id"]: item for item in stand.client.get("/v1/scenarios").json()}
    assert items["demo_run"]["last_run"]["id"] == run_id


def test_index_profiles_crud(stand):
    items = {item["id"]: item for item in stand.client.get("/v1/index-profiles").json()}
    assert items["demo"]["chunking"] == "recursive 800/120"
    assert items["demo"]["targets"] == [{"endpoint": "http://localhost:6333", "collection": "chunks_v1"}]

    created = stand.client.post("/v1/index-profiles", json={"yaml": profile_yaml("p_new")})
    assert created.status_code == 201
    draft = stand.client.put("/v1/index-profiles/p_new", json={
        "yaml": profile_yaml("p_new", chunking={"chunk_size": 100, "chunk_overlap": 100}),
        "etag": created.json()["etag"],
    })
    assert draft.status_code == 200
    assert _issues_from(draft.json()) == [("config.value_error", "chunking.chunk_overlap")]
    assert stand.client.delete("/v1/index-profiles/p_new").status_code == 204
