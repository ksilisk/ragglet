from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4


@dataclass(frozen=True)
class ResolvedDataset:
    path: Path  # локальная директория с BEIR-триадой
    revision: str | None  # разрешённая ревизия источника; None — у источника нет ревизий


class DatasetSource(Protocol):
    supports_revision: bool

    def resolve(self, dataset_id: str, revision: str | None) -> ResolvedDataset: ...


class LocalSource:
    """`id` — путь к BEIR-директории. Ревизий нет: версия local-датасета — хэш содержимого."""

    supports_revision = False

    def resolve(self, dataset_id: str, revision: str | None) -> ResolvedDataset:
        path = Path(dataset_id)
        if not path.exists():
            raise FileNotFoundError(f"Local dataset directory not found: {path}")
        return ResolvedDataset(path=path, revision=None)


class HfSource:
    """
    `id` — repo id датасета на HuggingFace Hub, ревизия — коммит репозитория.
    У BeIR/* qrels лежат в соседнем `{id}-qrels`, поэтому ревизия составная:
    `<data_sha>+<qrels_sha>`.
    """

    supports_revision = True

    def resolve(self, dataset_id: str, revision: str | None) -> ResolvedDataset:
        data_rev, sep, qrels_rev = (revision or "").partition("+")
        snap = _snapshot(dataset_id, data_rev or None)

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
            qrels_snap = _snapshot(f"{dataset_id}-qrels", qrels_rev or None)
            qrels_src, resolved = qrels_snap, f"{snap.name}+{qrels_snap.name}"

        out = _cache_dir(dataset_id, resolved)
        if not out.exists():
            _materialize_from_parquet(dataset_id, snap, qrels_src, out)
        return ResolvedDataset(path=out, revision=resolved)


SOURCES: dict[str, DatasetSource] = {"local": LocalSource(), "hf": HfSource()}


def check_source(source: str, revision: str | None) -> None:
    if source not in SOURCES:
        raise ValueError(f"unknown dataset source {source!r}; available: {sorted(SOURCES)}")
    if revision is not None and not SOURCES[source].supports_revision:
        raise ValueError(
            f"dataset.revision is not supported for source {source!r}: its version is the content hash"
        )


def resolve_dataset(source: str, dataset_id: str, revision: str | None = None) -> ResolvedDataset:
    check_source(source, revision)
    return SOURCES[source].resolve(dataset_id, revision)


def describe_dataset(source: str, dataset_id: str, revision: str | None) -> str:
    if revision is None:
        return f"{source}:{dataset_id}"
    short = "+".join(part[:12] for part in revision.split("+"))
    return f"{source}:{dataset_id}@{short}"


def _snapshot(repo_id: str, revision: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("source='hf' requires 'huggingface_hub' (project dependency).") from e
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision))


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
