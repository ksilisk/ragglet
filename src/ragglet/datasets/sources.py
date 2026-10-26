from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from ragglet.config.errors import FieldError
from ragglet.core.issues import ValidationError, error


@dataclass(frozen=True)
class ResolvedDataset:
    path: Path  # локальная директория с BEIR-триадой
    revision: str | None  # разрешённая ревизия источника; None — у источника нет ревизий


class DatasetSource(Protocol):
    supports_revision: bool

    def resolve(self, dataset_id: str, revision: str | None, offline: bool = False) -> ResolvedDataset:
        """offline — только то, что уже на диске: нет — FileNotFoundError, без сети."""
        ...


class LocalSource:
    """`id` — путь к BEIR-директории. Ревизий нет: версия local-датасета — хэш содержимого."""

    supports_revision = False

    def resolve(self, dataset_id: str, revision: str | None, offline: bool = False) -> ResolvedDataset:
        path = Path(dataset_id)
        if not path.is_dir():
            # путь — на машине, где идёт проверка; через API это сервер, а не клиент
            raise _not_found(str(path), f"local dataset directory not found: {path.resolve()}", "dataset.id")
        return ResolvedDataset(path=path, revision=None)


class HfSource:
    """
    `id` — repo id датасета на HuggingFace Hub, ревизия — коммит репозитория.
    У BeIR/* qrels лежат в соседнем `{id}-qrels`, поэтому ревизия составная:
    `<data_sha>+<qrels_sha>`.
    """

    supports_revision = True

    def resolve(self, dataset_id: str, revision: str | None, offline: bool = False) -> ResolvedDataset:
        data_rev, sep, qrels_rev = (revision or "").partition("+")
        fetch = _cached_snapshot if offline else _snapshot
        snap = fetch(dataset_id, data_rev or None)

        # Зеркала, которые раздают BEIR как есть (corpus.jsonl/queries.jsonl/qrels).
        if (snap / "corpus.jsonl").exists() or (snap / "corpus.jsonl.gz").exists():
            if sep:
                raise ValueError(f"{dataset_id} has no separate qrels repository; pin a single commit")
            return ResolvedDataset(path=snap, revision=snap.name)

        if (snap / "qrels").exists():
            if sep:
                raise ValueError(f"{dataset_id} keeps qrels in the same repository; pin a single commit")
            qrels_src, resolved = snap / "qrels", snap.name
        else:
            if revision is not None and not sep:
                raise ValueError(
                    f"{dataset_id} keeps qrels in {dataset_id}-qrels; pin both: '<data_sha>+<qrels_sha>'"
                )
            qrels_snap = fetch(f"{dataset_id}-qrels", qrels_rev or None)
            qrels_src, resolved = qrels_snap, f"{snap.name}+{qrels_snap.name}"

        out = _cache_dir(dataset_id, resolved)
        if not out.exists():
            _materialize_from_parquet(dataset_id, snap, qrels_src, out)
        return ResolvedDataset(path=out, revision=resolved)


SOURCES: dict[str, DatasetSource] = {"local": LocalSource(), "hf": HfSource()}


def check_source(source: str, revision: str | None) -> None:
    if source not in SOURCES:
        raise FieldError("source", f"unknown dataset source {source!r}; available: {sorted(SOURCES)}")
    if revision is not None and not SOURCES[source].supports_revision:
        raise FieldError(
            "revision", f"dataset.revision is not supported for source {source!r}: its version is the content hash"
        )


def resolve_dataset(
        source: str, dataset_id: str, revision: str | None = None, *, offline: bool = False
) -> ResolvedDataset:
    """offline=True — без сети: hf только из локального кэша, иначе FileNotFoundError."""
    check_source(source, revision)
    return SOURCES[source].resolve(dataset_id, revision, offline)


def describe_dataset(source: str, dataset_id: str, revision: str | None) -> str:
    if revision is None:
        return f"{source}:{dataset_id}"
    short = "+".join(part[:12] for part in revision.split("+"))
    return f"{source}:{dataset_id}@{short}"


def _not_found(where: str, message: str, path: str) -> ValidationError:
    return ValidationError("Dataset not found", [error("dataset.not_found", where, message, path=path)])


def _cached_snapshot(repo_id: str, revision: str | None) -> Path:
    """Только из локального кэша HF: нет — LocalEntryNotFoundError (подкласс FileNotFoundError)."""
    return _snapshot(repo_id, revision, local_files_only=True)


def _snapshot(repo_id: str, revision: str | None, **download: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
    except ImportError as e:
        raise ImportError("source='hf' requires 'huggingface_hub' (project dependency).") from e
    try:
        return Path(snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision, **download))
    except RevisionNotFoundError as e:
        raise _not_found(f"{repo_id}@{revision}", f"revision {revision!r} not found in {repo_id}",
                         "dataset.revision") from e
    except RepositoryNotFoundError as e:
        # сетевые ошибки Hub остаются сбоем: это стенд, а не конфиг
        raise _not_found(repo_id, f"dataset repository {repo_id!r} not found on HuggingFace Hub "
                                  "(or access is denied)", "dataset.id") from e


def _cache_dir(repo_id: str, revision: str) -> Path:
    return Path.home() / ".cache" / "ragglet" / "beir" / f"{repo_id.replace('/', '__')}@{revision}"


def _read_parquet_dir(directory: Path):
    import pandas as pd

    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {directory}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _write_jsonl(df, path: Path, fields: dict[str, str]) -> None:
    columns = {
        key: (df[col].fillna("").astype(str).tolist() if col in df else None) for key, col in fields.items()
    }
    n = len(df)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            obj = {key: (vals[i] if vals is not None else "") for key, vals in columns.items()}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _materialize_from_parquet(repo_id: str, snap: Path, qrels_src: Path, out: Path) -> None:
    """Конвенция BeIR/*: corpus/queries как parquet. Пишем во временную директорию и
    переименовываем: существующая `out` всегда полная."""
    tmp = out.with_name(f".{out.name}.{uuid4().hex}.tmp")
    try:
        (tmp / "qrels").mkdir(parents=True)
        corpus_df = _read_parquet_dir(snap / "corpus")
        _write_jsonl(corpus_df, tmp / "corpus.jsonl", {"_id": "_id", "title": "title", "text": "text"})
        queries_df = _read_parquet_dir(snap / "queries")
        _write_jsonl(queries_df, tmp / "queries.jsonl", {"_id": "_id", "text": "text"})

        tsvs = sorted(qrels_src.glob("*.tsv")) + sorted((qrels_src / "qrels").glob("*.tsv"))
        if not tsvs:
            raise FileNotFoundError(f"No qrels tsv found for {repo_id} (looked in {qrels_src})")
        for tsv in tsvs:
            shutil.copyfile(tsv, tmp / "qrels" / tsv.name)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
