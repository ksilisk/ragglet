"""
Чтение per_query.repeat{r}.parquet|csv: фильтры, сортировка, страница и строка одного запроса.
Колонки берутся по наличию: у старых поколений формата их меньше, ничего не пересчитывается.
Сверху сервер добавляет только first_relevant_rank — позицию первого документа из truth_ids в
final_doc_ids: ровно то условие, по которому считается doc_hit@k.
"""
from __future__ import annotations

import ast
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from ragglet.registry.protocol import QueryFilter, QueryPage, RegistryError

# в CSV списки записаны Python-repr: разбираем по набору колонок, а не по первой «[» — текст
# запроса тоже может начинаться со скобки; parquet хранит настоящие списки
LIST_COLUMNS = (
    "truth_ids", "retrieved_ids", "retrieved_doc_ids", "reranked_ids", "context_ids", "final_ids",
    "final_doc_ids", "rerank_scores",
)
TEXT_COLUMNS = ("context_text", "prompt_text", "answer", "raw_generation_response")  # только в строке запроса
STRING_COLUMNS = ("qid", "query", "outcome", "timeout_stage", "emb_cache", "ret_cache", "generation_error")
RANK = "first_relevant_rank"
FILE_NAME = re.compile(r"^per_query\.repeat(?P<repeat>\d+)\.(?P<ext>parquet|csv)$")


def per_query_files(run_dir: Path) -> dict[int, Path]:
    """Номер повтора -> файл; parquet, если он есть, иначе CSV."""
    found: dict[int, Path] = {}
    for path in sorted(run_dir.glob("per_query.repeat*.*")):
        match = FILE_NAME.match(path.name)
        if match is None:
            continue
        repeat = int(match["repeat"])
        if repeat not in found or match["ext"] == "parquet":
            found[repeat] = path
    return dict(sorted(found.items()))


def _as_list(value: Any) -> list[Any] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return list(value)  # массив parquet -> список


def first_relevant_rank(final_doc_ids: list[str] | None, truth_ids: list[str] | None) -> int | None:
    truth = set(truth_ids or ())
    return next((rank for rank, doc_id in enumerate(final_doc_ids or (), start=1) if doc_id in truth), None)


@lru_cache(maxsize=4)
def _load(path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    """Разобранный файл; ключ кэша — путь, mtime и размер. Кадр после этого не меняется."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={column: str for column in STRING_COLUMNS})  # qid «0042» — не число 42
    for column in LIST_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(_as_list)
    if {"final_doc_ids", "truth_ids"} <= set(df.columns):
        ranks = [first_relevant_rank(f, t) for f, t in zip(df["final_doc_ids"], df["truth_ids"], strict=True)]
        df[RANK] = pd.array(ranks, dtype="Int64")
    return df


def load_per_query(path: Path) -> pd.DataFrame:
    try:
        stat = path.stat()
        return _load(str(path), stat.st_mtime_ns, stat.st_size)
    except Exception as e:  # pyarrow, csv, literal_eval: файл ещё пишется или повреждён
        raise RegistryError(409, "run.file_unreadable", f"{path.name} is being written or damaged: {e}") from e


def _repeat_file(run_dir: Path, repeat: int) -> tuple[dict[int, Path], Path]:
    files = per_query_files(run_dir)
    if not files:
        raise RegistryError(404, "run.no_per_query", "no per-query files: the run has not finished a repeat yet, "
                                                     "or run.save_per_query is false")
    if repeat not in files:
        raise RegistryError(404, "run.repeat_not_found", f"repeat {repeat} is not written; written: {list(files)}",
                            path="repeat")
    return files, files[repeat]


def _require(df: pd.DataFrame, column: str, param: str) -> pd.Series:
    if column not in df.columns:
        raise RegistryError(422, "request.filter", f"this run has no column {column!r} (older artifacts format)",
                            path=param)
    return df[column]


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """NaN -> null, массивы -> списки: так же, как их отдал бы JSON."""
    return json.loads(df.to_json(orient="records", double_precision=15))


def query_page(run_dir: Path, f: QueryFilter) -> QueryPage:
    files, path = _repeat_file(run_dir, f.repeat)
    df = load_per_query(path)
    mask = pd.Series(True, index=df.index)
    if f.outcome is not None:
        mask &= _require(df, "outcome", "outcome") == f.outcome
    if f.miss_at is not None:
        rank = _require(df, RANK, "miss_at")
        mask &= (rank.isna() | (rank > f.miss_at)).fillna(True).astype(bool)
    if f.min_total_ms is not None:
        mask &= _require(df, "lat_total_ms", "min_total_ms") >= f.min_total_ms
    if f.text:
        needle = f.text.casefold()
        found = pd.Series(False, index=df.index)
        for column in ("qid", "query"):
            if column in df.columns:
                found |= df[column].fillna("").astype(str).str.casefold().str.contains(needle, regex=False)
        mask &= found

    columns = [c for c in df.columns if c not in LIST_COLUMNS and c not in TEXT_COLUMNS]
    out = df.loc[mask, columns]
    if f.sort:
        key = f.sort.removeprefix("-")
        if key not in columns:
            raise RegistryError(422, "request.sort", f"unknown sort column {key!r}; columns: {columns}", path="sort")
        out = out.sort_values(key, ascending=not f.sort.startswith("-"), na_position="last", kind="stable")
    return QueryPage(
        items=_records(out.iloc[f.offset:f.offset + f.limit]),
        total=len(out),
        repeat=f.repeat,
        repeats=list(files),
        source=path.name,
        columns=columns,
    )


def query_row(run_dir: Path, repeat: int, qid: str) -> dict[str, Any] | None:
    _, path = _repeat_file(run_dir, repeat)
    df = load_per_query(path)
    if "qid" not in df.columns:
        return None
    rows = df[df["qid"].astype(str) == qid]
    return _records(rows.iloc[:1])[0] if len(rows) else None


def doc_of(chunk_id: str) -> str:
    """Контракт сборки: chunk_id = «<doc_id>#<chunk_index>»."""
    return chunk_id.rsplit("#", 1)[0]


def truth_docs(row: dict[str, Any]) -> list[dict[str, Any]]:
    doc_rank = {doc_id: rank for rank, doc_id in enumerate(row.get("final_doc_ids") or (), start=1)}
    return [{"doc_id": doc_id, "doc_rank": doc_rank.get(doc_id)} for doc_id in row.get("truth_ids") or ()]


def result_chunks(row: dict[str, Any], context_enabled: bool) -> list[dict[str, Any]]:
    """Финальная выдача по чанкам; у прогонов без final_doc_ids (февраль, июнь) — пусто."""
    final_ids, final_doc_ids = row.get("final_ids"), row.get("final_doc_ids")
    if final_ids is None or final_doc_ids is None:
        return []
    truth = set(row.get("truth_ids") or ())
    doc_rank = {doc_id: rank for rank, doc_id in enumerate(final_doc_ids, start=1)}
    scores = dict(zip(row.get("reranked_ids") or (), row.get("rerank_scores") or (), strict=False))
    context = set(row.get("context_ids") or ())
    return [
        {
            "rank": rank,
            "chunk_id": chunk_id,
            "doc_id": doc_of(chunk_id),
            "doc_rank": doc_rank.get(doc_of(chunk_id)),
            "relevant": doc_of(chunk_id) in truth,
            "rerank_score": scores.get(chunk_id),
            "in_context": chunk_id in context if context_enabled else None,
        }
        for rank, chunk_id in enumerate(final_ids, start=1)
    ]
