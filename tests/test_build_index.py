from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ragglet.cli.app import app
from ragglet.config.index_profile import IndexProfileConfig
from ragglet.core.issues import ValidationError
from ragglet.indexing import build_index as build_module
from ragglet.indexing.metadata import extract_index_metadata

DEMO = Path(__file__).resolve().parents[1] / "datasets" / "demo"


class _Indexer:
    events: list = []
    fail_on_upsert: int | None = None

    def __init__(self, spec):
        self.collection = spec.collection

    def recreate_collection(self):
        self.events.append(("recreate", self.collection))

    def upsert(self, ids, vectors, payloads):
        upserts = sum(1 for e in self.events if e[0] == "upsert")
        if upserts == self.fail_on_upsert:
            raise ConnectionError("qdrant went away")
        self.events.append(("upsert", self.collection))

    def write_metadata(self, metadata):
        self.events.append(("metadata", self.collection, metadata))


class _Embedder:
    def __init__(self, model_name, normalize=True, revision=None):
        self.revision = revision or "f" * 40

    def compute_info(self):
        return {"device": "cuda:0", "accelerator": "NVIDIA L4", "torch_threads": 4}

    def embed_batch(self, texts):
        return [[0.0, 1.0] for _ in texts]


@pytest.fixture(autouse=True)
def fake_qdrant(monkeypatch):
    _Indexer.events = []
    _Indexer.fail_on_upsert = None
    monkeypatch.setattr(build_module, "QdrantIndexer", _Indexer)
    monkeypatch.setattr(build_module, "SentenceTransformerEmbedder", _Embedder)


def _profile_data(dataset_id: str = str(DEMO), **storage) -> dict:
    return {
        "id": "demo",
        "dataset": {"id": dataset_id},
        "embedding": {"model": "all-MiniLM-L6-v2", "batch_size": 16},
        "storage": storage or {"endpoint": "http://q:6333", "collection": "chunks_v1"},
    }


def _profile(dataset_id: str = str(DEMO), **storage) -> IndexProfileConfig:
    return IndexProfileConfig.model_validate(_profile_data(dataset_id, **storage))


def test_metadata_written_once_after_all_upserts():
    res = build_module.build_index(_profile())

    assert res.warnings == []  # demo без предупреждений

    kinds = [e[0] for e in _Indexer.events]
    assert kinds[0] == "recreate" and kinds[-1] == "metadata"
    assert kinds.count("metadata") == 1 and kinds.count("upsert") > 1
    meta = extract_index_metadata(_Indexer.events[-1][2])
    assert res.index == meta
    assert meta["corpus_sha256"] == "893aa67b4bb4c811afb0afc144ac6fc0b081ae724a39c3f6899b789225c71477"
    assert meta["embedding_revision"] == "f" * 40
    assert (meta["storage_mode"], meta["storage_backend_count"]) == ("single", 1)
    assert meta["build_env"]["compute"] == {"device": "cuda:0", "accelerator": "NVIDIA L4", "torch_threads": 4}
    assert {"commit", "python", "system"} <= set(meta["build_env"])


def test_failed_build_leaves_no_metadata():
    _Indexer.fail_on_upsert = 1
    with pytest.raises(ConnectionError):
        build_module.build_index(_profile())
    assert not any(e[0] == "metadata" for e in _Indexer.events)


def test_all_targets_share_build_metadata():
    build_module.build_index(
        _profile(
            mode="sharded",
            endpoint="http://q:6333",
            collection="c",
            backends=[
                {"id": "s0", "endpoint": "http://q:6333", "collection": "c0"},
                {"id": "s1", "endpoint": "http://q:6334", "collection": "c1"},
            ],
        )
    )
    written = {e[1]: e[2] for e in _Indexer.events if e[0] == "metadata"}
    assert set(written) == {"c0", "c1"}
    assert written["c0"] == written["c1"]
    assert written["c0"]["ragglet"]["storage_backend_count"] == 2


def test_invalid_dataset_fails_before_collections_are_touched(tmp_path):
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text('{"_id":"d1","text":"a"}\n{"_id":"d1","text":"b"}\n', encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"a?"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text("q1\td1\t1\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="corpus.duplicate_id"):
        build_module.build_index(_profile(str(tmp_path)))
    assert _Indexer.events == []


def test_dry_run_checks_dataset_without_model_or_collections(monkeypatch):
    def no_model(*args, **kwargs):
        raise AssertionError("dry run must not load the model")

    monkeypatch.setattr(build_module, "SentenceTransformerEmbedder", no_model)

    res = build_module.build_index(_profile(), dry_run=True)

    assert res.corpus_sha256 == "893aa67b4bb4c811afb0afc144ac6fc0b081ae724a39c3f6899b789225c71477"
    assert res.documents == 50
    assert [t.collection for t in res.targets] == ["chunks_v1"]
    assert res.index is None
    assert _Indexer.events == []


def test_dry_run_rejects_invalid_dataset(tmp_path):
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text('{"_id":"d1","text":"a"}\n{"_id":"d1","text":"b"}\n', encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"a?"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text("q1\td1\t1\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="corpus.duplicate_id"):
        build_module.build_index(_profile(str(tmp_path)), dry_run=True)


def test_progress_reported_per_document():
    calls = []

    res = build_module.build_index(_profile(), progress=lambda *a: calls.append(a))

    assert calls == [("corpus", n, res.documents) for n in range(1, res.documents + 1)]


def test_unsupported_backend_fails_before_collections_are_touched():
    with pytest.raises(ValidationError) as exc:
        build_module.build_index(_profile(type="pgvector", endpoint="http://pg:5432", collection="c"))
    assert [(i.code, i.where) for i in exc.value.issues] == [("storage.not_implemented", "target default")]
    assert _Indexer.events == []


def test_cli_dry_run_lists_collections_to_recreate(tmp_path):
    path = tmp_path / "index.yaml"
    path.write_text(yaml.safe_dump(_profile_data()), encoding="utf-8")

    result = CliRunner().invoke(app, ["index", "build", str(path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "will recreate: chunks_v1 at http://q:6333 (target default)" in result.output
    assert _Indexer.events == []
