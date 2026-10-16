from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any


@dataclass(frozen=True)
class StorageTarget:
    id: str
    type: str
    endpoint: str
    collection: str


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


def _target_from_backend(backend: Any, fallback_id: str) -> StorageTarget:
    target_id = getattr(backend, "id", None) or fallback_id
    return StorageTarget(
        id=str(target_id),
        type=str(getattr(backend, "type")),
        endpoint=str(getattr(backend, "endpoint")),
        collection=str(getattr(backend, "collection")),
    )


def _primary_backend(storage: Any) -> Any:
    backend = getattr(storage, "backend", None)
    if backend is not None:
        return backend
    return storage


def build_storage_layout(storage: Any) -> StorageLayout:
    mode = str(getattr(storage, "mode", "single"))
    routing = getattr(storage, "routing", None)
    routing_key = str(getattr(routing, "key", "chunk_id"))

    primary = _target_from_backend(_primary_backend(storage), "default")
    configured = [
        _target_from_backend(backend, f"backend_{idx}")
        for idx, backend in enumerate(getattr(storage, "backends", []) or [])
    ]

    targets = configured if mode != "single" else [primary]
    if not targets:
        targets = [primary]
    target_ids = [target.id for target in targets]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError(f"Storage backend ids must be unique: {target_ids}")
    if mode != "single" and len(targets) < 2:
        raise ValueError(f"storage.mode='{mode}' requires at least two storage.backends")

    return StorageLayout(
        mode=mode,
        primary=targets[0] if mode != "single" else primary,
        targets=targets,
        routing_key=routing_key,
    )
