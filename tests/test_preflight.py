from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ragglet.bench import run_scenario as run_module
from ragglet.bench.compare import section_kind
from ragglet.bench.preflight import preflight
from ragglet.cli.app import app
from ragglet.config.index_profile import IndexProfileConfig
from ragglet.config.scenario import ScenarioConfig
from ragglet.core.issues import ValidationError
from ragglet.indexing.metadata import build_index_metadata
from ragglet.stores.qdrant_store_async import CollectionInfo

DEMO = Path(__file__).resolve().parents[1] / "datasets" / "demo"
DEMO_SHA = "893aa67b4bb4c811afb0afc144ac6fc0b081ae724a39c3f6899b789225c71477"
MODEL_SHA = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BUILD_ENV = {"commit": "c" * 40, "python": "3.12.7", "system": {}, "compute": {"device": "cpu"}}
PARAMS = {"vector_size": 384, "distance": "Cosine", "hnsw": {"m": 16, "ef_construct": 128}}


def _scenario_data(tmp_path: Path | None = None, **overrides) -> dict:
    return {
        "name": "s",
        "dataset": {"id": str(DEMO)},
        "storage": {
            "mode": "single",
            "backend": {"type": "qdrant", "endpoint": "http://q:6333", "collection": "chunks_v1"},
        },
        "embedding": {"model": "all-MiniLM-L6-v2", "normalize": True},
        "run": {"repeats": 1, "shuffle": False, "artifacts_dir": str(tmp_path or "results/runs")},
        **overrides,
    }


def _scenario(tmp_path: Path | None = None, **overrides) -> ScenarioConfig:
    return ScenarioConfig.model_validate(_scenario_data(tmp_path, **overrides))


def _meta(**overrides) -> dict:
    profile = IndexProfileConfig.model_validate(
        {
            "id": "demo",
            "dataset": {"id": "datasets/demo"},
            "embedding": {"model": "all-MiniLM-L6-v2"},
            "storage": {"endpoint": "http://q:6333", "collection": "chunks_v1"},
        }
    )
    meta = build_index_metadata(
        profile,
        built_at="2026-10-20T12:00:00Z",
        corpus_sha256=DEMO_SHA,
        embedding_revision=MODEL_SHA,
        storage_mode="single",
        storage_backend_count=1,
        build_env=BUILD_ENV,
    )
    meta["ragglet"].update(overrides)
    return meta


class _Store:
    def __init__(self, metadata, status="green"):
        self.info = CollectionInfo(
            metadata=metadata, params=PARAMS, status=status, points=53, indexed_vectors=0, segments=2
        )

    async def collection_info(self):
        return self.info

    async def describe(self):
        return {"type": "qdrant", "version": "1.17.0", "status": self.info.status}


def _codes(exc) -> list[tuple[str, str]]:
    return [(i.code, i.where) for i in exc.value.issues]


async def test_compatible_index_passes_and_builds_manifest():
    checked = await preflight(_scenario(), {"default": _Store(_meta())})

    assert len(checked.dataset.items) == 100
    assert checked.manifest() == {
        "dataset": {
            "revision": None,
            "corpus_sha256": DEMO_SHA,
            "eval_sha256": "b68bae536f369e98dedd1f9d9a16e2c90245d347f535c7515b27ccf7b480a376",
        },
        "index": {
            "profile_id": "demo",
            "built_at": "2026-10-20T12:00:00Z",
            "embedding_revision": MODEL_SHA,
            "chunking": {"method": "recursive", "chunk_size": 800, "chunk_overlap": 120},
            "params": PARAMS,
            "build_env": BUILD_ENV,
        },
    }
    assert checked.warnings == []


async def test_collection_not_green_warns_but_passes():
    checked = await preflight(_scenario(), {"default": _Store(_meta(), status="yellow")})

    assert [(i.code, i.severity, i.where) for i in checked.warnings] == [
        ("index.not_green", "warning", "target default (chunks_v1)")
    ]


@pytest.mark.parametrize(
    "metadata",
    [
        None,  # сборка не дошла до записи метаданных
        {"ragglet": {"index_profile_id": "demo", "embedding_model": "all-MiniLM-L6-v2"}},  # старый контракт
        {"ragglet": {k: v for k, v in _meta()["ragglet"].items() if k != "build_env"}},  # до ragglet-23
    ],
)
async def test_incomplete_or_outdated_index_is_rejected(metadata):
    with pytest.raises(ValidationError) as exc:
        await preflight(_scenario(), {"default": _Store(metadata)})
    assert _codes(exc) == [("index.no_metadata", "target default (chunks_v1)")]
    assert "Rebuild the index" in str(exc.value)


class _MissingStore:
    async def collection_info(self):
        return None


async def test_missing_collection_is_rejected_once():
    with pytest.raises(ValidationError) as exc:
        await preflight(_scenario(), {"default": _MissingStore()})
    assert _codes(exc) == [("index.no_collection", "target default (chunks_v1)")]
    assert "Rebuild the index" in str(exc.value)


async def test_all_index_mismatches_reported_at_once():
    meta = _meta(
        corpus_sha256="c" * 64,
        dataset_id="datasets/other",
        storage_mode="sharded",
        storage_backend_count=2,
        embedding_normalize=False,
    )
    with pytest.raises(ValidationError) as exc:
        await preflight(_scenario(), {"default": _Store(meta)})

    assert sorted(i.code for i in exc.value.issues) == [
        "index.corpus",
        "index.embedding",
        "index.storage_layout",
    ]
    message = str(exc.value)
    assert "index is sharded with 2 backend(s), scenario expects single with 1" in message
    assert f"index built from local:datasets/other (cccccccccccc), scenario dataset local:{DEMO} ({DEMO_SHA[:12]})" in message
    assert "embedding_normalize: index has False, scenario expects True" in message


async def test_targets_from_different_builds_are_rejected():
    cfg = _scenario(
        storage={
            "mode": "sharded",
            "backend": {"type": "qdrant", "endpoint": "http://q:6333", "collection": "c"},
            "backends": [
                {"id": "s0", "type": "qdrant", "endpoint": "http://q:6333", "collection": "c0"},
                {"id": "s1", "type": "qdrant", "endpoint": "http://q:6334", "collection": "c1"},
            ],
        }
    )
    shard = {"storage_mode": "sharded", "storage_backend_count": 2}
    stores = {
        "s0": _Store(_meta(**shard)),
        "s1": _Store(_meta(**shard, built_at="2026-10-21T09:00:00Z")),
    }
    with pytest.raises(ValidationError) as exc:
        await preflight(cfg, stores)
    assert _codes(exc) == [("index.mixed_builds", "storage")]

    stores["s1"] = _Store(None)
    with pytest.raises(ValidationError) as exc:
        await preflight(cfg, stores)
    assert _codes(exc) == [("index.no_metadata", "target s1 (c1)")]


async def test_invalid_dataset_fails_before_index_checks(tmp_path):
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text('{"_id":"d1","text":"a"}\n', encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"a?"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text("q1\td2\t1\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="Dataset validation failed") as exc:
        await preflight(_scenario(dataset={"id": str(tmp_path)}), {"default": _Store(None)})
    assert _codes(exc) == [("qrels.unknown_doc", "qrels/test.tsv:1")]


class _RunStore(_Store):
    metadata_for: dict = {}

    def __init__(self, endpoint, collection):
        super().__init__(self.metadata_for.get(collection))

    async def search(self, vector, top_k, params=None):
        from ragglet.stores.qdrant_store_async import Candidate

        return [Candidate(id="p", score_raw=1.0, payload={"chunk_id": "x#0", "doc_id": "x", "text": "t"})]

    async def close(self):
        pass


class _Embedder:
    created: list = []
    fail = False

    def __init__(self, model_name, normalize=True, revision=None):
        if self.fail:
            raise OSError("model not found")
        self.revision = revision
        self.created.append(revision)

    def compute_info(self):
        return {"device": "mps", "accelerator": None, "torch_threads": 8}

    def embed_batch(self, texts):
        return [[0.0] for _ in texts]

    async def embed_query(self, text):
        return self.embed_batch([text])[0]


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(run_module, "AsyncQdrantStore", _RunStore)
    monkeypatch.setattr(run_module, "SentenceTransformerEmbedder", _Embedder)
    _Embedder.created.clear()
    _Embedder.fail = False
    return _RunStore


async def test_run_without_valid_index_creates_no_run_dir(tmp_path, fake_runtime):
    fake_runtime.metadata_for = {}
    with pytest.raises(ValidationError):
        await run_module.run_scenario(_scenario(tmp_path / "runs"))
    assert not (tmp_path / "runs").exists()


async def test_model_load_failure_creates_no_run_dir(tmp_path, fake_runtime):
    fake_runtime.metadata_for = {"chunks_v1": _meta()}
    _Embedder.fail = True
    with pytest.raises(OSError):
        await run_module.run_scenario(_scenario(tmp_path / "runs"))
    assert not (tmp_path / "runs").exists()


async def test_run_writes_manifest_and_loads_index_model_revision(tmp_path, fake_runtime):
    fake_runtime.metadata_for = {"chunks_v1": _meta()}
    cfg = _scenario(tmp_path / "runs", dataset={"id": str(DEMO), "limit": 3})

    res = await run_module.run_scenario(cfg)

    manifest = json.loads((res.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["corpus_sha256"] == DEMO_SHA
    assert manifest["index"]["embedding_revision"] == MODEL_SHA
    assert _Embedder.created == [MODEL_SHA]
    assert (res.run_dir / "scenario.snapshot.json").exists()
    assert manifest["index"]["build_env"] == BUILD_ENV
    assert manifest["rerank"] is None
    env = manifest["env"]
    assert env["compute"] == {"device": "mps", "accelerator": None, "torch_threads": 8}
    assert {"commit", "python", "system"} <= set(env)
    assert env["services"] == {"storage": {"default": {"type": "qdrant", "version": "1.17.0", "status": "green"}}}
    for section in manifest:
        section_kind(section)  # каждая секция участвует в compare_runs


async def test_dry_run_checks_index_without_models_or_run_dir(tmp_path, fake_runtime):
    fake_runtime.metadata_for = {"chunks_v1": _meta()}

    res = await run_module.run_scenario(_scenario(tmp_path / "runs"), dry_run=True)

    assert res.preflight.index["embedding_revision"] == MODEL_SHA
    assert (res.run_dir, res.summary, res.repeats) == (None, None, [])
    assert _Embedder.created == []
    assert not (tmp_path / "runs").exists()


async def test_progress_counts_queries_over_all_repeats(tmp_path, fake_runtime):
    fake_runtime.metadata_for = {"chunks_v1": _meta()}
    cfg = _scenario(
        dataset={"id": str(DEMO), "limit": 3},
        run={"repeats": 2, "shuffle": False, "artifacts_dir": str(tmp_path / "runs")},
    )
    calls = []

    await run_module.run_scenario(cfg, progress=lambda *a: calls.append(a))

    assert calls == [("queries", n, 6) for n in range(1, 7)]


async def test_unsupported_backend_is_a_validation_error(fake_runtime):
    backend = {"type": "pgvector", "endpoint": "http://pg:5432", "collection": "chunks_v1"}

    with pytest.raises(ValidationError) as exc:
        await run_module.run_scenario(_scenario(storage={"mode": "single", "backend": backend}), dry_run=True)
    assert _codes(exc) == [("storage.not_implemented", "target default")]


def _scenario_file(tmp_path: Path) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(_scenario_data(tmp_path / "runs")), encoding="utf-8")
    return path


def test_cli_dry_run_prints_preflight_report(tmp_path, fake_runtime):
    fake_runtime.metadata_for = {"chunks_v1": _meta()}

    result = CliRunner().invoke(app, ["bench", "run", str(_scenario_file(tmp_path)), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Preflight: OK" in result.output
    assert f"corpus sha256: {DEMO_SHA}" in result.output
    assert f"embedding: all-MiniLM-L6-v2@{MODEL_SHA}" in result.output
    assert not (tmp_path / "runs").exists()


def test_cli_validation_error_exits_with_code_2(tmp_path, fake_runtime):
    fake_runtime.metadata_for = {}

    result = CliRunner().invoke(app, ["bench", "run", str(_scenario_file(tmp_path)), "--dry-run"])

    assert result.exit_code == 2
    assert "index.no_metadata" in result.output


def _paths(exc) -> list[tuple[str, str | None]]:
    return [(i.code, i.path) for i in exc.value.issues]


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (
            {"retrieval": {"strategy": "cascade", "sources": [{"name": "v"}]}},
            [("retrieval.not_implemented", "retrieval.strategy")],
        ),
        (
            {"retrieval": {"strategy": "fanout_merge", "sources": [
                {"name": "v"}, {"name": "k1", "kind": "keyword"}, {"name": "k2", "kind": "keyword"},
            ]}},
            [("retrieval.keyword_sources", "retrieval.sources[2]")],
        ),
        (
            {"retrieval": {"strategy": "fanout_merge", "sources": [
                {"name": "v", "params": {"backend_ref": "default"}},
                {"name": "w", "params": {"backend_ref": "shard_9"}},
            ]}},
            [("retrieval.backend_ref", "retrieval.sources[1].params.backend_ref")],
        ),
        (
            {"storage": {"mode": "sharded", "backend": {"type": "qdrant", "endpoint": "http://q:6333", "collection": "c"},
                         "backends": [{"id": "s", "type": "qdrant", "endpoint": "http://q:6333", "collection": "c0"},
                                      {"id": "s", "type": "qdrant", "endpoint": "http://q:6334", "collection": "c1"}]}},
            [("storage.layout", "storage.backends")],
        ),
    ],
    ids=["cascade", "two_keyword_sources", "unknown_backend_ref", "duplicate_target_ids"],
)
async def test_unsupported_scenario_settings_fail_before_any_client(overrides, expected, tmp_path, fake_runtime):
    with pytest.raises(ValidationError) as exc:
        await run_module.run_scenario(_scenario(tmp_path / "runs", **overrides), dry_run=True)
    assert _paths(exc) == expected


async def test_config_problems_are_reported_together(fake_runtime):
    backend = {"type": "pgvector", "endpoint": "http://pg:5432", "collection": "chunks_v1"}
    cfg = _scenario(
        storage={"mode": "single", "backend": backend},
        retrieval={"strategy": "cascade", "sources": [{"name": "v"}]},
    )
    with pytest.raises(ValidationError) as exc:
        await run_module.run_scenario(cfg, dry_run=True)
    assert _paths(exc) == [
        ("storage.not_implemented", "storage.backend.type"),
        ("retrieval.not_implemented", "retrieval.strategy"),
    ]


async def test_unreachable_qdrant_is_a_config_error():
    from ragglet.core.errors import ServiceUnavailable

    class _Down:
        async def collection_info(self):
            raise ServiceUnavailable("http://q:6333", "ConnectError: refused")

    with pytest.raises(ValidationError) as exc:
        await preflight(_scenario(), {"default": _Down()})
    assert _paths(exc) == [("storage.unreachable", "storage.backend.endpoint")]
    assert exc.value.issues[0].where == "target default (http://q:6333)"


async def test_unreachable_redis_is_a_config_error(monkeypatch):
    from ragglet.bench import preflight as preflight_module
    from ragglet.core.errors import ServiceUnavailable

    async def down(endpoint, timeout):
        raise ServiceUnavailable(endpoint, "ConnectionError: refused")

    monkeypatch.setattr(preflight_module, "redis_version", down)
    cfg = _scenario(cache={"enabled": True, "embedding": {"endpoint": "redis://r:6379/0"}})
    with pytest.raises(ValidationError) as exc:
        await preflight(cfg, {"default": _Store(_meta())})
    assert _paths(exc) == [("cache.unreachable", "cache.embedding.endpoint")]


async def test_redis_version_reports_refused_connection():
    from ragglet.cache.embedding_redis import redis_version
    from ragglet.core.errors import ServiceUnavailable

    with pytest.raises(ServiceUnavailable, match="redis://127.0.0.1:1/0"):
        await redis_version("redis://127.0.0.1:1/0", timeout=1.0)


async def test_missing_local_dataset_points_to_dataset_id(tmp_path):
    with pytest.raises(ValidationError) as exc:
        await preflight(_scenario(dataset={"id": str(tmp_path / "missing")}), {"default": _Store(_meta())})
    assert _paths(exc) == [("dataset.not_found", "dataset.id")]


async def test_preflight_keeps_state_of_every_target(fake_runtime):
    fake_runtime.metadata_for = {"chunks_v1": _meta()}
    res = await run_module.run_scenario(_scenario(), dry_run=True)
    assert set(res.preflight.collections) == {"default"}
    assert res.preflight.collections["default"].points == 53


async def test_run_id_names_the_run_dir_once(tmp_path, fake_runtime):
    fake_runtime.metadata_for = {"chunks_v1": _meta()}
    cfg = _scenario(tmp_path / "runs", dataset={"id": str(DEMO), "limit": 2})

    res = await run_module.run_scenario(cfg, run_id="s_20261125T101540Z-2")

    assert res.run_dir == tmp_path / "runs" / "s_20261125T101540Z-2"
    with pytest.raises(FileExistsError):
        await run_module.run_scenario(cfg, run_id="s_20261125T101540Z-2")


async def test_run_id_accepts_a_dir_prepared_for_the_run(tmp_path, fake_runtime):
    fake_runtime.metadata_for = {"chunks_v1": _meta()}
    cfg = _scenario(tmp_path / "runs", dataset={"id": str(DEMO), "limit": 2})
    run_dir = tmp_path / "runs" / "s_20261125T101540Z"
    run_dir.mkdir(parents=True)
    for name in ("run.json", "task.json", "task.log"):  # реестр и менеджер задач пишут их до старта
        (run_dir / name).write_text("{}")

    res = await run_module.run_scenario(cfg, run_id=run_dir.name)

    assert res.run_dir == run_dir
    assert (run_dir / "scenario.snapshot.json").exists()
    assert (run_dir / "run.json").read_text() == "{}"


def test_run_plan_matches_progress_total():
    from ragglet.bench.plan import run_plan

    cfg = _scenario(run={"repeats": 3, "load": {"concurrency": 4}})
    plan = run_plan(cfg, 5)
    assert (plan.queries_per_repeat, plan.total, plan.load) == (5, 15, {"type": "closed", "concurrency": 4})


def test_cli_unsupported_strategy_exits_with_code_2(tmp_path, fake_runtime):
    path = tmp_path / "scenario.yaml"
    data = _scenario_data(tmp_path / "runs", retrieval={"strategy": "cascade", "sources": [{"name": "v"}]})
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = CliRunner().invoke(app, ["bench", "run", str(path), "--dry-run"])

    assert result.exit_code == 2
    assert "retrieval.not_implemented" in result.output
