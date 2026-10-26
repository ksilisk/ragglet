"""Обогащение детали запроса: оценки и заголовки из датасета прогона, тексты чанков из Qdrant."""
from __future__ import annotations

import json
import shutil

import pytest
from api_support import DEMO, QDRANT, collection, index_meta, make_stand
from registry_support import RUN_ID, TRUTH, demo_scenario, make_run_dir

TA = TRUTH["q001"]
OTHER = "rag_evaluation_overview_metrics_and_protocols"


@pytest.fixture
def stand(tmp_path, monkeypatch):
    return make_stand(tmp_path, monkeypatch, file_registry=True)


@pytest.fixture
def dataset(tmp_path):
    """Копия demo с заголовками документов: в самом demo они пустые."""
    root = tmp_path / "ds"
    shutil.copytree(DEMO, root)
    docs = [json.loads(line) for line in (root / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    (root / "corpus.jsonl").write_text(
        "".join(json.dumps({**doc, "title": f"Title {doc['_id']}"}) + "\n" for doc in docs), encoding="utf-8")
    return root


def _detail(stand, qid: str = "q001", run_id: str = RUN_ID) -> dict:
    response = stand.client.get(f"/v1/runs/{run_id}/queries/{qid}")
    assert response.status_code == 200, response.json()
    return response.json()


def _codes(body: dict) -> tuple[str | None, str | None]:
    return body["grades"]["code"], body["texts"]["code"]


def test_grades_titles_and_texts_when_sources_match(stand, dataset):
    make_run_dir(stand.settings.artifacts_dir, cfg=demo_scenario(dataset={"id": str(dataset), "limit": 100}))
    stand.qdrant.points[(QDRANT, "chunks_v1")] = {
        f"{TA}#0": {"chunk_id": f"{TA}#0", "title": "", "text": "RAG is retrieval plus generation"},
        f"{OTHER}#0": {"chunk_id": f"{OTHER}#0", "title": "Payload title", "text": "Evaluation protocols"},
    }

    body = _detail(stand)

    assert (body["grades"]["available"], body["texts"]["available"]) == (True, True)
    assert body["truth"] == [{"doc_id": TA, "doc_rank": 1, "grade": 1, "title": f"Title {TA}"}]
    results = [(r["grade"], r["title"], r["text"]) for r in body["results"]]
    assert results == [
        (1, f"Title {TA}", "RAG is retrieval plus generation"),  # пустой title в payload — заголовок из корпуса
        (1, f"Title {TA}", None),  # чанка нет в коллекции
        (0, "Payload title", "Evaluation protocols"),  # не релевантен: оценка 0
    ]


def test_changed_dataset_hides_grades_and_titles(stand, dataset):
    make_run_dir(stand.settings.artifacts_dir, cfg=demo_scenario(dataset={"id": str(dataset), "limit": 100}))

    qrels = dataset / "qrels" / "test.tsv"
    qrels.write_text(qrels.read_text(encoding="utf-8").replace(f"q001\t{TA}\t1", f"q001\t{TA}\t2"),
                     encoding="utf-8")
    body = _detail(stand)
    assert body["grades"]["code"] == "dataset.changed"
    assert (body["truth"][0]["grade"], body["truth"][0]["title"]) == (None, f"Title {TA}")  # корпус тот же

    (dataset / "corpus.jsonl").write_text((DEMO / "corpus.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    assert _detail(stand)["truth"][0]["title"] is None  # корпус другой — заголовков нет

    shutil.rmtree(dataset)
    assert _detail(stand)["grades"]["code"] == "dataset.not_local"


def test_texts_follow_the_index_build(stand):
    make_run_dir(stand.settings.artifacts_dir)
    target = (QDRANT, "chunks_v1")

    stand.qdrant.collections[target] = collection(index_meta(built_at="2026-11-02T09:00:00Z"))
    body = _detail(stand)
    assert body["texts"]["code"] == "index.rebuilt" and "2026-11-02T09:00:00Z" in body["texts"]["message"]
    assert [r["text"] for r in body["results"]] == [None, None, None]
    assert body["grades"]["available"]  # оценки — из датасета, не из индекса

    stand.qdrant.collections[target] = collection(None)
    assert _codes(_detail(stand))[1] == "index.no_metadata"
    del stand.qdrant.collections[target]
    assert _codes(_detail(stand))[1] == "index.missing"
    stand.qdrant.down.add(QDRANT)
    assert _codes(_detail(stand))[1] == "index.unreachable"


def test_without_manifest_nothing_is_guessed(stand):
    make_run_dir(stand.settings.artifacts_dir, manifest=False)

    body = _detail(stand)

    assert _codes(body) == ("run.no_manifest", "run.no_manifest")
    assert [r["relevant"] for r in body["results"]] == [True, True, False]  # из truth_ids строки


def test_outdated_snapshot_has_no_texts(stand):
    run_dir = make_run_dir(stand.settings.artifacts_dir)
    snapshot = json.loads((run_dir / "scenario.snapshot.json").read_text(encoding="utf-8"))
    snapshot["dataset"]["query_field"] = "question"  # поле старого поколения: текущая схема его не знает
    (run_dir / "scenario.snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    assert _codes(_detail(stand)) == (None, "run.snapshot_outdated")


def test_timeout_query_needs_no_index(stand):
    make_run_dir(stand.settings.artifacts_dir)
    stand.qdrant.down.add(QDRANT)

    body = _detail(stand, "q007")

    assert (body["results"], body["texts"]["available"]) == ([], True)  # выдачи нет — в Qdrant не ходим
