from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ragglet.datasets import sources
from ragglet.datasets.beir import read_corpus
from ragglet.datasets.sources import HfSource, check_source, describe_dataset, resolve_dataset

DATA_SHA = "a" * 40
QRELS_SHA = "b" * 40


def _beir_hub(root: Path) -> dict[tuple[str, str], Path]:
    """Фейковый кэш HF: BeIR/x (parquet) + BeIR/x-qrels (tsv)."""
    data = root / "BeIR__x" / DATA_SHA
    (data / "corpus").mkdir(parents=True)
    (data / "queries").mkdir()
    pd.DataFrame({"_id": ["d1", "d2"], "title": ["T", None], "text": ["alpha", "beta"]}).to_parquet(
        data / "corpus" / "corpus-0.parquet"
    )
    pd.DataFrame({"_id": ["q1"], "text": ["alpha?"]}).to_parquet(data / "queries" / "queries-0.parquet")
    qrels = root / "BeIR__x-qrels" / QRELS_SHA
    qrels.mkdir(parents=True)
    (qrels / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
    return {("BeIR/x", DATA_SHA): data, ("BeIR/x-qrels", QRELS_SHA): qrels}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    snapshots = _beir_hub(tmp_path / "hub")
    calls: list[tuple[str, str | None]] = []

    def fake_snapshot(repo_id: str, revision: str | None) -> Path:
        calls.append((repo_id, revision))
        for (repo, sha), path in snapshots.items():
            if repo == repo_id and revision in (None, sha):
                return path
        raise FileNotFoundError(f"{repo_id}@{revision}")

    monkeypatch.setattr(sources, "_snapshot", fake_snapshot)
    return calls


def test_local_source_has_no_revisions():
    check_source("local", None)
    with pytest.raises(ValueError, match="not supported for source 'local'"):
        check_source("local", DATA_SHA)
    with pytest.raises(ValueError, match=r"unknown dataset source 's3'; available: \['hf', 'local'\]"):
        check_source("s3", None)


def test_config_rejects_revision_for_local():
    from ragglet.config.scenario import DatasetConfig

    assert DatasetConfig(source="hf", id="BeIR/x", revision=f"{DATA_SHA}+{QRELS_SHA}").revision
    with pytest.raises(ValueError, match="content hash"):
        DatasetConfig(source="local", id="datasets/demo", revision="v2")


def test_hf_beir_resolves_composite_revision_and_materializes(hub, tmp_path):
    resolved = resolve_dataset("hf", "BeIR/x")

    assert resolved.revision == f"{DATA_SHA}+{QRELS_SHA}"
    assert resolved.path == tmp_path / "home" / ".cache" / "ragglet" / "beir" / f"BeIR__x@{DATA_SHA}+{QRELS_SHA}"
    assert [d.title for d in read_corpus(resolved.path)] == ["T", ""]  # null -> "", не "None"
    assert (resolved.path / "qrels" / "test.tsv").exists()
    assert describe_dataset("hf", "BeIR/x", resolved.revision) == f"hf:BeIR/x@{'a' * 12}+{'b' * 12}"


def test_hf_pinned_revision_is_passed_to_both_repositories(hub):
    resolve_dataset("hf", "BeIR/x", f"{DATA_SHA}+{QRELS_SHA}")
    assert hub == [("BeIR/x", DATA_SHA), ("BeIR/x-qrels", QRELS_SHA)]


def test_hf_beir_requires_both_parts_when_pinned(hub):
    with pytest.raises(ValueError, match="pin both"):
        resolve_dataset("hf", "BeIR/x", DATA_SHA)


def test_hf_materialization_is_atomic(hub, monkeypatch, tmp_path):
    real = sources._read_parquet_dir

    def broken(directory: Path):
        if directory.name == "queries":
            raise OSError("disk full")
        return real(directory)

    monkeypatch.setattr(sources, "_read_parquet_dir", broken)
    with pytest.raises(OSError, match="disk full"):
        resolve_dataset("hf", "BeIR/x")
    cache = tmp_path / "home" / ".cache" / "ragglet" / "beir"
    assert list(cache.iterdir()) == []  # ни неполной директории, ни tmp

    monkeypatch.setattr(sources, "_read_parquet_dir", real)
    assert (resolve_dataset("hf", "BeIR/x").path / "queries.jsonl").exists()


def test_hf_cache_is_reused(hub, monkeypatch):
    first = resolve_dataset("hf", "BeIR/x")
    monkeypatch.setattr(sources, "_materialize_from_parquet", lambda *a: pytest.fail("rematerialized"))
    assert resolve_dataset("hf", "BeIR/x") == first


def test_hf_raw_mirror_is_used_in_place(tmp_path, monkeypatch):
    snap = tmp_path / "hub" / DATA_SHA
    snap.mkdir(parents=True)
    (snap / "corpus.jsonl").write_text('{"_id":"d1","text":"x"}\n', encoding="utf-8")
    monkeypatch.setattr(sources, "_snapshot", lambda repo_id, revision: snap)

    assert HfSource().resolve("mirror/x", None) == sources.ResolvedDataset(snap, DATA_SHA)
    with pytest.raises(ValueError, match="single commit"):
        HfSource().resolve("mirror/x", f"{DATA_SHA}+{QRELS_SHA}")


def test_model_revision_from_hf_cache(tmp_path, monkeypatch):
    import huggingface_hub

    from ragglet.modules.embedders.st_embedder import hub_model_id
    from ragglet.modules.hub import cached_model_revision

    assert hub_model_id("all-MiniLM-L6-v2") == "sentence-transformers/all-MiniLM-L6-v2"
    assert hub_model_id("BAAI/bge-small-en") == "BAAI/bge-small-en"
    assert hub_model_id(str(tmp_path)) == str(tmp_path)

    seen = []

    def fake_cache(repo_id, filename, revision=None):
        seen.append((repo_id, revision))
        return None if revision == "missing" else f"/hf/models--x/snapshots/{DATA_SHA}/{filename}"

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake_cache)
    assert cached_model_revision("sentence-transformers/all-MiniLM-L6-v2", None) == DATA_SHA
    assert seen == [("sentence-transformers/all-MiniLM-L6-v2", None)]
    with pytest.raises(RuntimeError, match="pin a full commit sha"):
        cached_model_revision("sentence-transformers/all-MiniLM-L6-v2", "missing")
    assert cached_model_revision(str(tmp_path), None) is None  # локальный путь


def test_redis_embedding_key_includes_model_revision():
    from ragglet.cache.embedding_redis import RedisEmbeddingCache

    cache = RedisEmbeddingCache("redis://localhost:6379/0", 60, "m", DATA_SHA, True, namespace="ns")
    assert cache._key("q").startswith(f"emb:m@{DATA_SHA}:1:ns:")
