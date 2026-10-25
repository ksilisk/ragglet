from __future__ import annotations

import pytest
import yaml
from api_support import scenario_yaml
from fastapi.testclient import TestClient

from ragglet.api.problems import loc_to_path


def _issues(response) -> list[tuple[str, str | None, str | None]]:
    return [(i["code"], i["path"], i["where"]) for i in response.json()["issues"]]


def _check(stand, text: str):
    return stand.client.post("/v1/runs", json={"scenario": text, "dry_run": True})


def test_loc_to_path():
    assert loc_to_path(("retrieval", "sources", 1, "params")) == "retrieval.sources[1].params"
    assert loc_to_path(()) is None


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"bogus": 1}, [("config.extra_forbidden", "bogus", None)]),
        ({"retrieval": {"strategy": "fanout_merge", "sources": [{"name": "a"}, {"name": "b", "params": 5}]}},
         [("config.dict_type", "retrieval.sources[1].params", None)]),
        ({"retrieval": {"async": "maybe"}}, [("config.bool_parsing", "retrieval.async", None)]),
        ({"metrics": {"quality": {"k": [50]}}}, [("config.value_error", "metrics.quality.k", None)]),
        ({"generation": {"enabled": True, "model": "m"}}, [("config.value_error", "generation.enabled", None)]),
        ({"rerank": {"enabled": True}}, [("config.value_error", "rerank.model", None)]),
        ({"run": {"load": {"type": "open"}}}, [("config.value_error", "run.load.type", None)]),
        ({"run": {"load": {"concurrency": 0}}}, [("config.greater_than_equal", "run.load.concurrency", None)]),
    ],
    ids=["extra", "list_index", "alias", "root_validator", "root_validator_2", "nested_validator", "load_type",
         "load_param"],
)
def test_schema_errors_point_to_fields(stand, overrides, expected):
    response = _check(stand, scenario_yaml(**overrides))
    assert response.status_code == 422
    body = response.json()
    assert body["title"] == "Scenario config is invalid"
    assert _issues(response) == expected


def test_metrics_message_has_no_pydantic_prefix(stand):
    response = _check(stand, scenario_yaml(metrics={"quality": {"k": [50]}}))
    assert response.json()["issues"][0]["message"] == "metrics.quality.k=50 exceeds final list depth 30 (retrieval.top_k)"


@pytest.mark.parametrize(
    "text, where",
    [("a: 1\nb: [1, 2\nc: 3\n", "line 3, column 2"), ("", None), ("- 1\n- 2\n", None)],
    ids=["broken", "empty", "list_root"],
)
def test_yaml_errors(stand, text, where):
    response = _check(stand, text)
    assert response.status_code == 422
    assert _issues(response) == [("config.yaml", None, where)]


def test_request_body_errors(stand):
    response = stand.client.post("/v1/runs", json={"dry_run": True})
    assert response.status_code == 422
    assert _issues(response) == [("request.missing", "scenario", None)]


def test_unknown_route_and_run_are_problems(stand):
    assert _issues(stand.client.get("/v1/nope")) == [("http.not_found", None, "/v1/nope")]
    response = stand.client.get("/v1/runs/x_20261125T101540Z")
    assert response.status_code == 404
    assert _issues(response) == [("run.not_found", None, "runs/x_20261125T101540Z")]


def test_unexpected_error_is_a_problem(stand, monkeypatch):
    async def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(stand.jobs, "queue", broken)
    response = TestClient(stand.client.app, raise_server_exceptions=False).get("/v1/tasks")
    assert response.status_code == 500
    assert response.json()["issues"] == [
        {"code": "server.error", "severity": "error", "path": None, "where": None, "message": "RuntimeError: boom"},
    ]


def test_yaml_scalar_name_is_still_a_schema_error(stand):
    response = _check(stand, yaml.safe_dump({"name": 5}))
    assert response.status_code == 422
    assert {code for code, _, _ in _issues(response)} == {"config.string_type", "config.missing"}
