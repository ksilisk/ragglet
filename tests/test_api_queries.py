"""Запросы прогона: per-query parquet и CSV, фильтры, ранг первого релевантного, деталь запроса."""
from __future__ import annotations

import pytest
from api_support import make_stand, scenario_yaml
from registry_support import RUN_ID, TRUTH, make_feb_run_dir, make_run_dir

FEB = "baseline_qdrant_20260221T135048Z"


@pytest.fixture
def stand(tmp_path, monkeypatch):
    return make_stand(tmp_path, monkeypatch, file_registry=True)


def _page(stand, run_id: str = RUN_ID, **params) -> dict:
    response = stand.client.get(f"/v1/runs/{run_id}/queries", params=params)
    assert response.status_code == 200, response.json()
    return response.json()


def _qids(body: dict) -> list[str]:
    return [item["qid"] for item in body["items"]]


def _problem(stand, url: str, **params) -> tuple[int, str, str | None]:
    response = stand.client.get(url, params=params)
    issue = response.json()["issues"][0]
    return response.status_code, issue["code"], issue["path"]


def test_parquet_and_csv_give_the_same_rows(stand):
    root = stand.settings.artifacts_dir
    make_run_dir(root, "pq_20261024T180917Z", formats=("parquet",))
    make_run_dir(root, "csv_20261024T180917Z", formats=("csv",))

    pq, csv = _page(stand, "pq_20261024T180917Z"), _page(stand, "csv_20261024T180917Z")

    assert (pq["source"], csv["source"]) == ("per_query.repeat0.parquet", "per_query.repeat0.csv")
    assert pq["items"] == csv["items"]
    assert pq["columns"] == csv["columns"]
    detail = lambda run_id: stand.client.get(f"/v1/runs/{run_id}/queries/q001").json()  # noqa: E731
    pq_row, csv_row = detail("pq_20261024T180917Z"), detail("csv_20261024T180917Z")
    # CSV не отличает пустую строку от пустого значения: пустой текст из CSV — null
    assert (pq_row["row"].pop("answer"), csv_row["row"].pop("answer")) == ("", None)
    assert pq_row == csv_row  # списки из repr и из parquet


def test_rows_are_scalars_with_nan_as_null_and_rank(stand):
    make_run_dir(stand.settings.artifacts_dir)

    body = _page(stand)

    assert (body["repeat"], body["repeats"], body["total"]) == (0, [0, 1], 3)
    assert body["source"] == "per_query.repeat0.parquet"  # parquet предпочтительнее CSV
    first, _, timeout = body["items"]
    assert "final_ids" not in first and "context_text" not in first  # списки и тексты — в детали
    assert "first_relevant_rank" in body["columns"]
    assert [item["first_relevant_rank"] for item in body["items"]] == [1, 2, None]
    assert (first["timeout_stage"], first["lat_generation_ms"]) == (None, None)
    assert (timeout["outcome"], timeout["timeout_stage"], timeout["lat_rerank_ms"]) == ("timeout", "rerank", None)


def test_filters_sort_and_pages(stand):
    make_run_dir(stand.settings.artifacts_dir)

    assert _qids(_page(stand, outcome="timeout")) == ["q007"]
    assert _qids(_page(stand, miss_at=1)) == ["q003", "q007"]  # то же, что doc_hit@1 = 0
    assert _qids(_page(stand, miss_at=3)) == ["q007"]
    assert _qids(_page(stand, min_total_ms=300)) == ["q003", "q007"]
    assert _qids(_page(stand, q="FAILURE")) == ["q003"]
    assert _qids(_page(stand, q="[demo]")) == ["q007"]  # текст со скобкой остался текстом
    assert _qids(_page(stand, sort="-first_relevant_rank")) == ["q003", "q001", "q007"]  # null в конце
    assert _qids(_page(stand, sort="-lat_total_ms")) == ["q007", "q003", "q001"]
    page = _page(stand, sort="lat_total_ms", offset=1, limit=1)
    assert (page["total"], _qids(page)) == (3, ["q003"])
    assert _page(stand, repeat=1)["items"][0]["lat_total_ms"] == 74.0

    assert _problem(stand, f"/v1/runs/{RUN_ID}/queries", sort="bogus")[:2] == (422, "request.sort")
    assert _problem(stand, f"/v1/runs/{RUN_ID}/queries", repeat=5) == (404, "run.repeat_not_found", "repeat")


def test_qid_from_csv_stays_a_string(stand):
    make_run_dir(stand.settings.artifacts_dir, formats=("csv",), qids=("0042", "q003", "q007"))

    assert _qids(_page(stand))[0] == "0042"
    assert stand.client.get(f"/v1/runs/{RUN_ID}/queries/0042").json()["truth"][0]["doc_rank"] == 1


def test_older_format_has_fewer_columns(stand):
    make_feb_run_dir(stand.settings.artifacts_dir, FEB)

    body = _page(stand, FEB)
    assert "first_relevant_rank" not in body["columns"] and "outcome" not in body["columns"]
    assert body["items"][0]["recall@1"] == 1.0
    assert _problem(stand, f"/v1/runs/{FEB}/queries", outcome="timeout") == (422, "request.filter", "outcome")
    assert _problem(stand, f"/v1/runs/{FEB}/queries", miss_at=10) == (422, "request.filter", "miss_at")

    detail = stand.client.get(f"/v1/runs/{FEB}/queries/q1").json()
    assert (detail["results"], detail["truth"]) == ([], [{"doc_id": "d1", "doc_rank": None, "grade": None,
                                                          "title": None}])


def test_missing_and_unreadable_files(stand):
    run_id = stand.client.post("/v1/runs", json={"scenario": scenario_yaml()}).json()["id"]  # в очереди
    assert _problem(stand, f"/v1/runs/{run_id}/queries")[:2] == (404, "run.no_per_query")
    assert _problem(stand, "/v1/runs/nope_20261024T180917Z/queries")[:2] == (404, "run.not_found")

    run_dir = make_run_dir(stand.settings.artifacts_dir)
    (run_dir / "per_query.repeat1.parquet").write_bytes(b"PAR1 half written")
    assert _problem(stand, f"/v1/runs/{RUN_ID}/queries", repeat=1)[:2] == (409, "run.file_unreadable")


def test_query_detail_from_the_row(stand):
    make_run_dir(stand.settings.artifacts_dir)

    body = stand.client.get(f"/v1/runs/{RUN_ID}/queries/q001").json()

    ta = TRUTH["q001"]
    assert (body["qid"], body["repeat"]) == ("q001", 0)
    assert body["query"].startswith("What is Retrieval-Augmented Generation")
    assert [(t["doc_id"], t["doc_rank"]) for t in body["truth"]] == [(ta, 1)]
    assert [(r["rank"], r["doc_rank"], r["relevant"], r["rerank_score"], r["in_context"]) for r in body["results"]] == [
        (1, 1, True, 0.9, True),
        (2, 1, True, 0.8, True),  # второй чанк того же документа: doc-ранг тот же
        (3, 2, False, 0.1, False),
    ]
    assert body["results"][2]["doc_id"] == "rag_evaluation_overview_metrics_and_protocols"
    assert body["row"]["context_text"] == "chunk text" and body["row"]["final_ids"][0] == f"{ta}#0"

    timeout = stand.client.get(f"/v1/runs/{RUN_ID}/queries/q007", params={"repeat": 1}).json()
    assert (timeout["results"], timeout["truth"][0]["doc_rank"]) == ([], None)
    assert timeout["row"]["outcome"] == "timeout"

    assert _problem(stand, f"/v1/runs/{RUN_ID}/queries/q999")[:2] == (404, "query.not_found")
