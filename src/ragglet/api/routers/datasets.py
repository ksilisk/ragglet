from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import anyio
from fastapi import APIRouter

from ragglet.api import probes
from ragglet.api.deps import Services, Svc
from ragglet.api.schemas import DatasetList, HfDataset, LocalDataset
from ragglet.datasets.beir import available_splits
from ragglet.indexing.metadata import extract_index_metadata

router = APIRouter(tags=["datasets"])

SCAN_DEPTH = 2


def _is_beir(path: Path) -> bool:
    return (path / "corpus.jsonl").is_file() or (path / "corpus.jsonl.gz").is_file()


def local_datasets(roots: list[Path]) -> list[LocalDataset]:
    """BEIR-каталоги под корнями, без разбора файлов; id — путь так, как его писать в dataset.id."""
    found: dict[str, LocalDataset] = {}
    for root in roots:
        level = [root] if root.is_dir() else []
        for _ in range(SCAN_DEPTH + 1):
            for path in level:
                if _is_beir(path):
                    found.setdefault(path.as_posix(), LocalDataset(id=path.as_posix(), splits=available_splits(path)))
            level = sorted(child for path in level for child in path.iterdir() if child.is_dir())
    return sorted(found.values(), key=lambda d: d.id)


def _library_hf(svc: Services) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = defaultdict(list)
    for kind, library in (("scenario", svc.scenarios), ("profile", svc.profiles)):
        for entry in library.list():
            if entry.config is not None and entry.config.dataset.source == "hf":
                seen[entry.config.dataset.id].append(f"{kind}:{entry.id}")
    return seen


@router.get("/datasets")
async def list_datasets(svc: Svc) -> DatasetList:
    """Локальные датасеты под корнями сервера и hf-id из библиотеки и метаданных коллекций."""
    settings = svc.settings
    local = await anyio.to_thread.run_sync(local_datasets, settings.dataset_roots)
    seen = await anyio.to_thread.run_sync(_library_hf, svc)
    for status, server in await probes.qdrant_all(settings.qdrant_endpoints, settings.probe_timeout_s):
        for name, info in (server.collections.items() if server else []):
            meta = extract_index_metadata(info.metadata)
            if meta is not None and meta["dataset_source"] == "hf":
                seen[meta["dataset_id"]].append(f"collection:{status.url.rstrip('/')}/{name}")
    hf = [HfDataset(id=dataset_id, seen_in=sorted(refs)) for dataset_id, refs in sorted(seen.items())]
    return DatasetList(local=local, hf=hf)
