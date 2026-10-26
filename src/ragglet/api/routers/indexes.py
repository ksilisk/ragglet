from __future__ import annotations

from typing import Literal

import anyio
from fastapi import APIRouter

from ragglet.api import probes
from ragglet.api.deps import Svc
from ragglet.api.refs import library_users
from ragglet.api.schemas import CollectionItem, IndexList
from ragglet.indexing.metadata import RAGGLET_METADATA_KEY, extract_index_metadata
from ragglet.stores.qdrant_store_async import CollectionInfo

router = APIRouter(tags=["indexes"])

State = Literal["ready", "optimizing", "incomplete", "foreign"]


def collection_state(info: CollectionInfo) -> State:
    """Состояния из ragglet-15: недособранную и собранную по старому контракту по коду не различить."""
    raw = info.metadata or {}
    if info.params is None:  # не один безымянный вектор: не ragglet
        return "foreign"
    if extract_index_metadata(raw) is not None:
        return "ready" if info.status == "green" else "optimizing"
    if RAGGLET_METADATA_KEY in raw or not raw:
        return "incomplete"
    return "foreign"


@router.get("/indexes")
async def list_indexes(svc: Svc) -> IndexList:
    """Коллекции со всех Qdrant из конфига сервера и кто из библиотеки на них ссылается."""
    settings = svc.settings
    servers = await probes.qdrant_all(settings.qdrant_endpoints, settings.probe_timeout_s)
    scenario_users, profile_users = await anyio.to_thread.run_sync(library_users, svc)
    collections = []
    for status, server in servers:
        if server is None:
            continue
        for name, info in server.collections.items():
            key = (probes.normalize_endpoint(status.url), name)
            collections.append(CollectionItem(
                endpoint=status.url,
                name=name,
                state=collection_state(info),
                status=info.status,
                points=info.points,
                indexed_vectors=info.indexed_vectors,
                segments=info.segments,
                params=info.params,
                ragglet=extract_index_metadata(info.metadata),
                used_by=scenario_users.get(key, []),
                profiles=profile_users.get(key, []),
            ))
    return IndexList(endpoints=[status for status, _ in servers], collections=collections)
