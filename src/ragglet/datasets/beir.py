from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, TextIO

from ragglet.core.issues import Issue, ValidationError, error


@dataclass(frozen=True)
class BeirDocument:
    doc_id: str
    title: str
    text: str


def document_text(doc: BeirDocument) -> str:
    """Текст, который индексируется: title + text."""
    return f"{doc.title}\n\n{doc.text}".strip() if doc.title else doc.text


@dataclass(frozen=True)
class DocRecord:
    line: int
    doc_id: str
    indexable: bool


@dataclass(frozen=True)
class QueryRecord:
    line: int
    qid: str
    text: str


@dataclass(frozen=True)
class QrelRecord:
    line: int
    qid: str
    doc_id: str
    grade: int


@dataclass
class ParsedDataset:
    """
    Разобранный BEIR-датасет без текстов документов: id, номера строк и хэш корпуса.
    `issues` — ошибки разбора; правила целостности живут в `datasets/checks.py`.
    """

    root: Path
    corpus_file: str
    corpus: list[DocRecord]
    corpus_sha256: str
    queries_file: str
    queries: list[QueryRecord]
    qrels_files: dict[str, str]
    qrels: dict[str, list[QrelRecord]]
    issues: list[Issue] = field(default_factory=list)

    @cached_property
    def doc_index(self) -> dict[str, DocRecord]:
        out: dict[str, DocRecord] = {}
        for record in self.corpus:
            out.setdefault(record.doc_id, record)
        return out

    @cached_property
    def query_index(self) -> dict[str, QueryRecord]:
        out: dict[str, QueryRecord] = {}
        for record in self.queries:
            out.setdefault(record.qid, record)
        return out


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _find(root: Path, *names: str) -> Path | None:
    for name in names:
        if (root / name).exists():
            return root / name
    return None


def _field(obj: dict[str, Any], key: str) -> str:
    value = obj.get(key)
    return "" if value is None else str(value)


def _iter_records(path: Path, rel: str, issues: list[Issue]) -> Iterator[tuple[int, str, dict]]:
    """(номер строки, _id, объект) для каждой корректной записи jsonl."""
    with _open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            where = f"{rel}:{line_no}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(error("json.invalid", where, f"invalid JSON: {e.msg}"))
                continue
            if not isinstance(obj, dict):
                issues.append(error("json.invalid", where, "record is not a JSON object"))
                continue
            record_id = _field(obj, "_id")
            if not record_id:
                issues.append(error("record.no_id", where, "record has no _id"))
                continue
            yield line_no, record_id, obj


def _canonical_document(doc: BeirDocument) -> bytes:
    payload = json.dumps([doc.doc_id, doc.title, doc.text], ensure_ascii=False, separators=(",", ":"))
    return payload.encode("utf-8") + b"\n"


def _iter_documents(path: Path, rel: str, issues: list[Issue]) -> Iterator[tuple[int, BeirDocument]]:
    for line_no, doc_id, obj in _iter_records(path, rel, issues):
        yield line_no, BeirDocument(doc_id=doc_id, title=_field(obj, "title"), text=_field(obj, "text"))


def _corpus_path(root: Path) -> Path | None:
    return _find(root, "corpus.jsonl", "corpus.jsonl.gz")


def read_corpus(root: Path) -> Iterator[BeirDocument]:
    """Документы корпуса в порядке файла; битая запись — ошибка, а не пропуск."""
    path = _corpus_path(root)
    if path is None:
        raise FileNotFoundError(f"BEIR corpus not found in {root}: tried corpus.jsonl, corpus.jsonl.gz")
    issues: list[Issue] = []
    for _, doc in _iter_documents(path, path.name, issues):
        if issues:
            break
        yield doc
    if issues:
        raise ValidationError(f"Corpus is invalid: {root}", issues)


def available_splits(root: Path) -> list[str]:
    qrels_dir = root / "qrels"
    if not qrels_dir.is_dir():
        return []
    names = {p.name.removesuffix(".gz").removesuffix(".tsv") for p in qrels_dir.glob("*.tsv*")}
    return sorted(names)


def _parse_qrels(path: Path, rel: str, issues: list[Issue]) -> list[QrelRecord]:
    """
    BEIR tsv `query-id corpus-id score` (заголовок допустим только первой строкой)
    или TREC `qid 0 docid rel`.
    """
    out: list[QrelRecord] = []
    first = True
    with _open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            is_first, first = first, False
            where = f"{rel}:{line_no}"
            if len(parts) == 3:
                qid, doc_id, score = parts
            elif len(parts) == 4:
                qid, _, doc_id, score = parts
            else:
                issues.append(
                    error("qrels.malformed", where, f"expected 3 (BEIR) or 4 (TREC) columns, got {len(parts)}")
                )
                continue
            try:
                grade = int(float(score))
            except ValueError:
                if not is_first:
                    issues.append(error("qrels.malformed", where, f"score {score!r} is not a number"))
                continue
            out.append(QrelRecord(line=line_no, qid=qid, doc_id=doc_id, grade=grade))
    return out


def parse_dataset(root: Path, splits: list[str] | None = None) -> ParsedDataset:
    """Разбирает corpus, queries и qrels выбранных split (None — все найденные)."""
    issues: list[Issue] = []

    corpus: list[DocRecord] = []
    digest = hashlib.sha256()
    corpus_path = _corpus_path(root)
    if corpus_path is None:
        issues.append(error("file.missing", "corpus.jsonl", f"no corpus.jsonl(.gz) in {root}"))
    else:
        for line_no, doc in _iter_documents(corpus_path, corpus_path.name, issues):
            digest.update(_canonical_document(doc))
            corpus.append(DocRecord(line_no, doc.doc_id, bool(document_text(doc).strip())))

    queries: list[QueryRecord] = []
    queries_path = _find(root, "queries.jsonl", "queries.jsonl.gz")
    if queries_path is None:
        issues.append(error("file.missing", "queries.jsonl", f"no queries.jsonl(.gz) in {root}"))
    else:
        for line_no, qid, obj in _iter_records(queries_path, queries_path.name, issues):
            queries.append(QueryRecord(line_no, qid, _field(obj, "text")))

    found = available_splits(root)
    qrels_files: dict[str, str] = {}
    qrels: dict[str, list[QrelRecord]] = {}
    if splits is None and not found:
        issues.append(error("file.missing", "qrels/", f"no qrels/<split>.tsv in {root}"))
    for split in found if splits is None else splits:
        path = _find(root, f"qrels/{split}.tsv", f"qrels/{split}.tsv.gz")
        if path is None:
            hint = ", ".join(found) if found else "none"
            issues.append(error("file.missing", f"qrels/{split}.tsv", f"no such split; available: {hint}"))
            continue
        rel = path.relative_to(root).as_posix()
        qrels_files[split] = rel
        qrels[split] = _parse_qrels(path, rel, issues)

    return ParsedDataset(
        root=root,
        corpus_file=corpus_path.name if corpus_path else "corpus.jsonl",
        corpus=corpus,
        corpus_sha256=digest.hexdigest(),
        queries_file=queries_path.name if queries_path else "queries.jsonl",
        queries=queries,
        qrels_files=qrels_files,
        qrels=qrels,
        issues=issues,
    )
