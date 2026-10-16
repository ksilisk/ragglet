from __future__ import annotations

import json
import shutil
from pathlib import Path


def resolve_dataset_dir(source: str, dataset_id: str) -> Path:
    """
    Приводит (source, id) к локальной директории с BEIR-триадой.

    - local: `id` — путь к директории датасета;
    - hf:    `id` — repo id на HuggingFace Hub; скачиваем и при необходимости
             материализуем в локальный BEIR-кэш.
    """
    if source == "local":
        path = Path(dataset_id)
        if not path.exists():
            raise FileNotFoundError(f"Local dataset directory not found: {path}")
        return path
    if source == "hf":
        return _resolve_hf(dataset_id)
    raise ValueError(f"Unknown dataset source: {source!r} (expected 'local' or 'hf')")


def _cache_dir(repo_id: str) -> Path:
    return Path.home() / ".cache" / "ragglet" / "beir" / repo_id.replace("/", "__")


def _resolve_hf(repo_id: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("source='hf' requires 'huggingface_hub' (project dependency).") from e

    out = _cache_dir(repo_id)
    if (out / "corpus.jsonl").exists() and (out / "queries.jsonl").exists() and (out / "qrels").exists():
        return out  # уже материализовано

    snap = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))

    # Некоторые зеркала раздают BEIR как есть (corpus.jsonl/queries.jsonl/qrels).
    if (snap / "corpus.jsonl").exists() or (snap / "corpus.jsonl.gz").exists():
        return snap

    # Конвенция BeIR/*: corpus/queries как parquet, qrels — в соседнем репозитории {id}-qrels.
    _materialize_from_parquet(repo_id, snap, out)
    return out


def _read_parquet_dir(directory: Path):
    import pandas as pd

    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {directory}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _write_jsonl(df, path: Path, fields: dict[str, str]) -> None:
    columns = {key: (df[col].astype(str).tolist() if col in df else None) for key, col in fields.items()}
    n = len(df)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            obj = {key: (vals[i] if vals is not None else "") for key, vals in columns.items()}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _materialize_from_parquet(repo_id: str, snap: Path, out: Path) -> None:
    from huggingface_hub import snapshot_download

    out.mkdir(parents=True, exist_ok=True)
    (out / "qrels").mkdir(exist_ok=True)

    corpus_df = _read_parquet_dir(snap / "corpus")
    _write_jsonl(corpus_df, out / "corpus.jsonl", {"_id": "_id", "title": "title", "text": "text"})

    queries_df = _read_parquet_dir(snap / "queries")
    _write_jsonl(queries_df, out / "queries.jsonl", {"_id": "_id", "text": "text"})

    # qrels: внутри основного репо или в соседнем {id}-qrels
    qrels_src = snap / "qrels" if (snap / "qrels").exists() else Path(
        snapshot_download(repo_id=f"{repo_id}-qrels", repo_type="dataset")
    )
    tsvs = list(qrels_src.glob("*.tsv")) + list((qrels_src / "qrels").glob("*.tsv"))
    if not tsvs:
        raise FileNotFoundError(f"No qrels tsv found for {repo_id} (looked in {qrels_src})")
    for tsv in tsvs:
        shutil.copyfile(tsv, out / "qrels" / tsv.name)
