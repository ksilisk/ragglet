from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any

from ragglet.core.issues import Issue, error, raise_on_errors

# схема конфига знает и другие типы, но реализован только qdrant
IMPLEMENTED_BACKENDS = {"qdrant"}


@dataclass(frozen=True)
class StorageTarget:
    id: str
    type: str
    endpoint: str
    collection: str
    path: str = "storage"  # где цель в конфиге: storage.backend, storage.backends[1]


@dataclass(frozen=True)
class StorageLayout:
    mode: str
    primary: StorageTarget
    targets: list[StorageTarget]
    routing_key: str

    def route_write(self, payload: dict[str, Any]) -> list[StorageTarget]:
        if self.mode == "single":
            return [self.primary]
        if self.mode == "replicated":
            return list(self.targets)
        if self.mode == "sharded":
            raw_key = str(payload.get(self.routing_key) or payload.get("chunk_id") or "")
            idx = int(sha1(raw_key.encode("utf-8")).hexdigest(), 16) % len(self.targets)
            return [self.targets[idx]]
        raise ValueError(f"Unsupported storage mode: {self.mode}")

    def target_by_id(self, target_id: str | None) -> StorageTarget:
        if target_id is None:
            return self.primary
        for target in self.targets:
            if target.id == target_id:
                return target
        raise KeyError(f"Unknown storage backend_ref: {target_id}")


def _target_from_backend(backend: Any, fallback_id: str, path: str) -> StorageTarget:
    target_id = getattr(backend, "id", None) or fallback_id
    return StorageTarget(
        id=str(target_id),
        type=str(backend.type),
        endpoint=str(backend.endpoint),
        collection=str(backend.collection),
        path=path,
    )


def _primary_backend(storage: Any) -> tuple[Any, str]:
    backend = getattr(storage, "backend", None)
    if backend is not None:
        return backend, "storage.backend"
    return storage, "storage"


def build_storage_layout(storage: Any) -> StorageLayout:
    mode = str(getattr(storage, "mode", "single"))
    routing = getattr(storage, "routing", None)
    routing_key = str(getattr(routing, "key", "chunk_id"))

    backend, path = _primary_backend(storage)
    primary = _target_from_backend(backend, "default", path)
    configured = [
        _target_from_backend(backend, f"backend_{idx}", f"storage.backends[{idx}]")
        for idx, backend in enumerate(getattr(storage, "backends", []) or [])
    ]

    targets = configured if mode != "single" else [primary]
    if not targets:
        targets = [primary]
    target_ids = [target.id for target in targets]
    issues = [
        error("storage.layout", "storage.backends", f"backend id {tid!r} is used more than once",
              path="storage.backends")
        for tid in sorted({tid for tid in target_ids if target_ids.count(tid) > 1})
    ]
    if mode != "single" and len(targets) < 2:
        issues.append(error("storage.layout", "storage.backends",
                            f"storage.mode={mode!r} requires at least two storage.backends", path="storage.backends"))
    raise_on_errors("Storage layout is invalid", issues)

    return StorageLayout(
        mode=mode,
        primary=targets[0] if mode != "single" else primary,
        targets=targets,
        routing_key=routing_key,
    )


def unimplemented_targets(layout: StorageLayout) -> list[Issue]:
    return [
        error(
            "storage.not_implemented",
            f"target {target.id}",
            f"backend {target.type!r} is not implemented, use one of {sorted(IMPLEMENTED_BACKENDS)}",
            path=f"{target.path}.type",
        )
        for target in layout.targets
        if target.type not in IMPLEMENTED_BACKENDS
    ]


def require_implemented(layout: StorageLayout) -> None:
    raise_on_errors("Storage is not supported", unimplemented_targets(layout))
