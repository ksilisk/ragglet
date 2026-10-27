"""
Обогащение детали запроса: оценки и заголовки из датасета прогона, тексты чанков из Qdrant.
Источники ищутся без сети и сверяются с manifest прогона: если датасет изменился или коллекция
пересобрана, деталь говорит об этом, а не показывает чужие данные.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import anyio
from pydantic import ValidationError as SchemaError

from ragglet.bench.dataset import eval_items, eval_sha256
from ragglet.config.scenario import ScenarioConfig
from ragglet.core.errors import ServiceUnavailable
from ragglet.core.issues import ValidationError
from ragglet.datasets.beir import parse_dataset, read_corpus
from ragglet.datasets.sources import resolve_dataset
from ragglet.indexing.metadata import extract_index_metadata
from ragglet.registry.protocol import RunRecord
from ragglet.storage.layout import build_storage_layout
from ragglet.stores.qdrant_store_async import AsyncQdrantStore


@dataclass(frozen=True)
class Availability:
    available: bool
    code: str | None = None  # почему нет: dataset.not_local, index.rebuilt, …
    message: str | None = None


AVAILABLE = Availability(True)


@dataclass(frozen=True)
class DatasetView:
    """Что из датасета совпало с прогоном: оценки — при том же eval_sha256, заголовки — при том же корпусе."""

    qrels: dict[str, dict[str, int]] | None  # qid -> doc_id -> оценка > 0
    titles: dict[str, str] | None
    grades: Availability


@dataclass(frozen=True)
class _Parsed:
    corpus_sha256: str
    eval_sha256: str
    qrels: dict[str, dict[str, int]]
    titles: dict[str, str]


def _unavailable(code: str, message: str) -> DatasetView:
    return DatasetView(qrels=None, titles=None, grades=Availability(False, code, message))


@lru_cache(maxsize=2)
def _parse(path: str, split: str, limit: int, stamp: tuple[int, ...]) -> _Parsed:
    """Разбор датасета; stamp — mtime и размеры файлов: правка датасета сбрасывает кэш."""
    ds = parse_dataset(Path(path), [split])
    if ds.issues:
        raise ValueError(f"dataset has {len(ds.issues)} parse errors, first: {ds.issues[0].message}")
    items = eval_items(ds, split, limit)
    return _Parsed(
        corpus_sha256=ds.corpus_sha256,
        eval_sha256=eval_sha256(items),
        qrels={item.qid: item.qrels for item in items},
        titles={doc.doc_id: doc.title for doc in read_corpus(Path(path))},
    )


def _stamp(path: Path) -> tuple[int, ...]:
    files = [p for p in sorted(path.rglob("*")) if p.is_file()]
    return tuple(n for p in files for n in (p.stat().st_mtime_ns, p.stat().st_size))


def dataset_view(scenario: dict[str, Any], manifest: dict[str, Any] | None) -> DatasetView:
    if manifest is None:
        return _unavailable("run.no_manifest", "the run has no manifest.json: nothing to check the dataset against")
    expected = manifest.get("dataset") or {}
    dataset = scenario.get("dataset") or {}
    try:
        resolved = resolve_dataset(str(dataset.get("source") or "local"), str(dataset.get("id")),
                                   expected.get("revision"), offline=True)
    except Exception as e:  # нет на диске, нет в кэше HF: сеть по клику на строку не трогаем
        return _unavailable("dataset.not_local", f"dataset is not available locally: {type(e).__name__}: {e}")
    try:
        parsed = _parse(str(resolved.path), str(dataset.get("split") or "test"), int(dataset.get("limit") or 0),
                        _stamp(resolved.path))
    except (OSError, ValueError, KeyError, ValidationError) as e:  # битая запись корпуса — ValidationError ядра
        return _unavailable("dataset.invalid", f"cannot read the dataset: {type(e).__name__}: {e}")
    titles = parsed.titles if parsed.corpus_sha256 == expected.get("corpus_sha256") else None
    if parsed.eval_sha256 != expected.get("eval_sha256"):
        return DatasetView(qrels=None, titles=titles, grades=Availability(
            False, "dataset.changed", "queries or qrels differ from the run (eval_sha256 mismatch)"))
    return DatasetView(qrels=parsed.qrels, titles=titles, grades=AVAILABLE)


class QueryInspector:
    """Датасет и Qdrant для детали запроса; один на сервер."""

    def __init__(self, timeout_s: float):
        self.timeout_s = timeout_s

    async def dataset(self, record: RunRecord, manifest: dict[str, Any] | None) -> DatasetView:
        return await anyio.to_thread.run_sync(dataset_view, record.scenario, manifest)

    async def texts(
            self, record: RunRecord, manifest: dict[str, Any] | None, chunk_ids: list[str]
    ) -> tuple[dict[str, dict[str, Any]], Availability]:
        """Payload чанков, если коллекции — та же сборка, что у прогона (built_at и профиль из manifest)."""
        if manifest is None:
            return {}, Availability(False, "run.no_manifest", "the run has no manifest.json: index build unknown")
        if not chunk_ids:
            return {}, AVAILABLE
        try:
            layout = build_storage_layout(ScenarioConfig.model_validate(record.scenario).storage)
        except (SchemaError, ValueError) as e:
            return {}, Availability(False, "run.snapshot_outdated", f"the snapshot does not fit the current schema: {e}")
        index = manifest.get("index") or {}
        payloads: dict[str, dict[str, Any]] = {}
        for target in layout.targets:  # sharded: чанк лежит в одной из целей
            where = f"{target.endpoint}/{target.collection}"
            store = AsyncQdrantStore(target.endpoint, target.collection)
            try:
                with anyio.fail_after(self.timeout_s):
                    info = await store.collection_info()
                    if info is None:
                        return {}, Availability(False, "index.missing", f"collection {where} does not exist")
                    meta = extract_index_metadata(info.metadata)
                    if meta is None:
                        return {}, Availability(False, "index.no_metadata",
                                                f"collection {where} has no ragglet metadata: it is being rebuilt")
                    if (meta["built_at"], meta["index_profile_id"]) != (index.get("built_at"), index.get("profile_id")):
                        return {}, Availability(False, "index.rebuilt", f"collection {where} was rebuilt at "
                                                f"{meta['built_at']}; the run used the build of {index.get('built_at')}")
                    payloads |= await store.payloads(chunk_ids)
            except (ServiceUnavailable, TimeoutError) as e:
                return {}, Availability(False, "index.unreachable", f"{where}: {e}")
            finally:
                await store.close()
        return payloads, AVAILABLE
