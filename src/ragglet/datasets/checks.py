from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from ragglet.core.issues import Issue, error, raise_on_errors, warning
from ragglet.datasets.beir import ParsedDataset, QrelRecord, parse_dataset

DatasetCheck = Callable[[ParsedDataset], Iterable[Issue]]


def _judgments(ds: ParsedDataset) -> Iterator[tuple[str, str, QrelRecord]]:
    """(split, место в файле, запись) по всем qrels."""
    for split, records in ds.qrels.items():
        rel = ds.qrels_files[split]
        for record in records:
            yield split, f"{rel}:{record.line}", record


def _relevant_doc_ids(ds: ParsedDataset) -> set[str]:
    return {r.doc_id for _, _, r in _judgments(ds) if r.grade > 0}


def corpus_duplicate_id(ds: ParsedDataset) -> Iterator[Issue]:
    for record in ds.corpus:
        first = ds.doc_index[record.doc_id]
        if first is not record:
            yield error(
                "corpus.duplicate_id",
                f"{ds.corpus_file}:{record.line}",
                f"_id {record.doc_id!r} already defined at line {first.line}",
            )


def corpus_empty_doc(ds: ParsedDataset) -> Iterator[Issue]:
    relevant = _relevant_doc_ids(ds)
    for record in ds.corpus:
        if not record.indexable and record.doc_id not in relevant:
            yield warning(
                "corpus.empty_doc",
                f"{ds.corpus_file}:{record.line}",
                f"{record.doc_id!r} has empty title and text (not indexed)",
            )


def queries_duplicate_id(ds: ParsedDataset) -> Iterator[Issue]:
    for record in ds.queries:
        first = ds.query_index[record.qid]
        if first is not record:
            yield error(
                "queries.duplicate_id",
                f"{ds.queries_file}:{record.line}",
                f"_id {record.qid!r} already defined at line {first.line}",
            )


def queries_empty_text(ds: ParsedDataset) -> Iterator[Issue]:
    judged = {r.qid for _, _, r in _judgments(ds) if r.grade > 0}
    for qid in sorted(judged):
        record = ds.query_index.get(qid)
        if record is not None and not record.text.strip():
            yield error("queries.empty_text", f"{ds.queries_file}:{record.line}", f"query {qid!r} has empty text")


def qrels_duplicate_pair(ds: ParsedDataset) -> Iterator[Issue]:
    for split, records in ds.qrels.items():
        seen: dict[tuple[str, str], int] = {}
        for record in records:
            key = (record.qid, record.doc_id)
            if key in seen:
                yield error(
                    "qrels.duplicate_pair",
                    f"{ds.qrels_files[split]}:{record.line}",
                    f"{record.qid} -> {record.doc_id!r} already judged at line {seen[key]}",
                )
            else:
                seen[key] = record.line


def qrels_unknown_query(ds: ParsedDataset) -> Iterator[Issue]:
    reported: set[tuple[str, str]] = set()
    for split, where, record in _judgments(ds):
        key = (split, record.qid)
        if record.qid not in ds.query_index and key not in reported:
            reported.add(key)
            yield error("qrels.unknown_query", where, f"query {record.qid!r} is not in {ds.queries_file}")


def qrels_unknown_doc(ds: ParsedDataset) -> Iterator[Issue]:
    for _, where, record in _judgments(ds):
        if record.doc_id not in ds.doc_index:
            make = error if record.grade > 0 else warning
            yield make("qrels.unknown_doc", where, f"{record.qid} -> {record.doc_id!r}: no such document in corpus")


def qrels_unindexable_doc(ds: ParsedDataset) -> Iterator[Issue]:
    for _, where, record in _judgments(ds):
        doc = ds.doc_index.get(record.doc_id)
        if record.grade > 0 and doc is not None and not doc.indexable:
            yield error(
                "qrels.unindexable_doc",
                where,
                f"{record.qid} -> {record.doc_id!r}: relevant document has empty title and text",
            )


def qrels_no_relevant(ds: ParsedDataset) -> Iterator[Issue]:
    for split, records in ds.qrels.items():
        first_line: dict[str, int] = {}
        relevant: set[str] = set()
        for record in records:
            first_line.setdefault(record.qid, record.line)
            if record.grade > 0:
                relevant.add(record.qid)
        if not relevant:
            continue  # пустой split целиком — ошибка split.empty
        for qid, line in first_line.items():
            if qid not in relevant:
                yield warning(
                    "qrels.no_relevant",
                    f"{ds.qrels_files[split]}:{line}",
                    f"query {qid!r} has no relevant judgments (score > 0) and is excluded from evaluation; "
                    "BEIR counts such queries as 0",
                )


def split_empty(ds: ParsedDataset) -> Iterator[Issue]:
    for split, records in ds.qrels.items():
        if not any(r.grade > 0 for r in records):
            yield error("split.empty", ds.qrels_files[split], "no relevant judgments (score > 0)")


DATASET_CHECKS: list[DatasetCheck] = [
    corpus_duplicate_id,
    corpus_empty_doc,
    queries_duplicate_id,
    queries_empty_text,
    qrels_duplicate_pair,
    qrels_unknown_query,
    qrels_unknown_doc,
    qrels_unindexable_doc,
    qrels_no_relevant,
    split_empty,
]


def validate_dataset(root: Path, splits: list[str] | None, label: str) -> tuple[ParsedDataset, list[Issue]]:
    """Разбор + все проверки. Ошибки — ValidationError, иначе (датасет, предупреждения)."""
    ds = parse_dataset(root, splits)
    issues = list(ds.issues)
    if not any(i.code == "file.missing" for i in issues):
        issues += [issue for check in DATASET_CHECKS for issue in check(ds)]
    return ds, raise_on_errors(f"Dataset validation failed: {label}", issues)
