"""Общее для API-тестов: стенд без Qdrant, Redis и torch."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from ragglet.api.app import create_app
from ragglet.api.settings import ServerSettings
from ragglet.config.index_profile import IndexProfileConfig
from ragglet.core.errors import ServiceUnavailable
from ragglet.indexing.metadata import build_index_metadata
from ragglet.jobs.memory import InMemoryJobManager
from ragglet.registry.files import FileRegistry
from ragglet.registry.memory import InMemoryRegistry
from ragglet.stores.qdrant_store_async import CollectionInfo, ServerCollections

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "datasets" / "demo"
DEMO_SHA = "893aa67b4bb4c811afb0afc144ac6fc0b081ae724a39c3f6899b789225c71477"
MODEL_SHA = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
PARAMS = {"vector_size": 384, "distance": "Cosine", "hnsw": {"m": 16, "ef_construct": 128}}
QDRANT = "http://q:6333"


def index_meta(**overrides) -> dict:
    profile = IndexProfileConfig.model_validate({
        "id": "demo",
        "dataset": {"id": "datasets/demo"},
        "embedding": {"model": "all-MiniLM-L6-v2"},
        "storage": {"endpoint": QDRANT, "collection": "chunks_v1"},
    })
    meta = build_index_metadata(
        profile, built_at="2026-10-20T12:00:00Z", corpus_sha256=DEMO_SHA, embedding_revision=MODEL_SHA,
        storage_mode="single", storage_backend_count=1, build_env={"commit": "c" * 40},
    )
    meta["ragglet"].update(overrides)
    return meta


def collection(metadata=None, status="green", params=PARAMS, points=53) -> CollectionInfo:
    return CollectionInfo(metadata=metadata, params=params, status=status, points=points, indexed_vectors=0,
                          segments=2)


@dataclass
class FakeQdrant:
    """Qdrant в памяти: (endpoint, коллекция) -> состояние; down — endpoint-ы, которые не отвечают."""

    collections: dict[tuple[str, str], CollectionInfo] = field(default_factory=dict)
    down: set[str] = field(default_factory=set)

    def store_class(self):
        fake = self

        class Store:
            def __init__(self, endpoint: str, collection: str):
                self.endpoint, self.collection = endpoint, collection

            async def collection_info(self):
                if self.endpoint in fake.down:
                    raise ServiceUnavailable(self.endpoint, "ConnectError: refused")
                return fake.collections.get((self.endpoint, self.collection))

            async def describe(self):
                return {"type": "qdrant", "version": "1.17.0"}

            async def close(self):
                pass

        return Store

    async def list_collections(self, endpoint: str) -> ServerCollections:
        if endpoint in self.down:
            raise ServiceUnavailable(endpoint, "ConnectError: refused")
        found = {name: info for (ep, name), info in self.collections.items() if ep == endpoint.rstrip("/")}
        return ServerCollections(version="1.17.0", collections=found)


def scenario_yaml(name: str = "demo_run", **overrides) -> str:
    data = {
        "name": name,
        "dataset": {"id": str(DEMO)},
        "storage": {"backend": {"type": "qdrant", "endpoint": QDRANT, "collection": "chunks_v1"}},
        "embedding": {"model": "all-MiniLM-L6-v2", "normalize": True},
        "run": {"repeats": 2, "shuffle": False, "artifacts_dir": "/client/side/path"},
        **overrides,
    }
    return yaml.safe_dump(data, sort_keys=False)


def profile_yaml(profile_id: str = "demo_build", **overrides) -> str:
    data = {
        "id": profile_id,
        "dataset": {"id": str(DEMO)},
        "embedding": {"model": "all-MiniLM-L6-v2"},
        "storage": {"endpoint": QDRANT, "collection": "chunks_v1"},
        **overrides,
    }
    return yaml.safe_dump(data, sort_keys=False)


class Clock:
    def __init__(self, at: datetime = datetime(2026, 11, 25, 10, 15, 40, tzinfo=UTC)):
        self.at = at

    def __call__(self) -> datetime:
        return self.at


@dataclass
class Stand:
    client: TestClient
    settings: ServerSettings
    registry: InMemoryRegistry | FileRegistry
    jobs: InMemoryJobManager
    qdrant: FakeQdrant
    clock: Clock


def make_stand(tmp_path: Path, monkeypatch, *, file_registry: bool = False, **settings) -> Stand:
    from ragglet.api import probes
    from ragglet.bench import preflight as preflight_module
    from ragglet.bench import run_scenario as run_module

    scenarios, profiles = tmp_path / "scenarios", tmp_path / "indexes"
    shutil.copytree(ROOT / "configs" / "scenarios", scenarios)
    shutil.copytree(ROOT / "configs" / "indexes", profiles)
    qdrant = FakeQdrant({(QDRANT, "chunks_v1"): collection(index_meta())})
    monkeypatch.setattr(run_module, "AsyncQdrantStore", qdrant.store_class())
    monkeypatch.setattr(probes, "AsyncQdrantStore", qdrant.store_class())
    monkeypatch.setattr(probes, "list_collections", qdrant.list_collections)

    async def redis_down(endpoint, timeout):
        raise ServiceUnavailable(endpoint, "ConnectionError: refused")

    monkeypatch.setattr(probes, "redis_version", redis_down)
    monkeypatch.setattr(preflight_module, "redis_version", redis_down)

    config = ServerSettings(
        _env_file=None,
        scenarios_dir=scenarios,
        index_profiles_dir=profiles,
        artifacts_dir=tmp_path / "runs",
        dataset_roots=[ROOT / "datasets"],
        qdrant_endpoints=[QDRANT],
        builds_dir=tmp_path / "builds",
        **settings,
    )
    clock = Clock()
    registry = (FileRegistry(config.artifacts_dir, config.builds_dir, now=clock) if file_registry
                else InMemoryRegistry(config.artifacts_dir, now=clock))
    jobs = InMemoryJobManager(registry, now=clock)
    client = TestClient(create_app(config, jobs=jobs, registry=registry))
    return Stand(client=client, settings=config, registry=registry, jobs=jobs, qdrant=qdrant, clock=clock)
