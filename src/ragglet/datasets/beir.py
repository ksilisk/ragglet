from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class BeirDocument:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class BeirQuery:
    qid: str
    text: str
    qrels: dict[str, int]  # relevant doc_id -> grade (> 0)


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _find(dataset_dir: Path, *names: str) -> Path:
    for name in names:
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate
    tried = ", ".join(names)
    raise FileNotFoundError(f"BEIR file not found in {dataset_dir}: tried {tried}")


def read_corpus(dataset_dir: Path) -> Iterator[BeirDocument]:
    path = _find(dataset_dir, "corpus.jsonl", "corpus.jsonl.gz")
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield BeirDocument(
                doc_id=str(obj["_id"]),
                title=str(obj.get("title", "")),
                text=str(obj.get("text", "")),
            )


def _read_query_texts(dataset_dir: Path) -> dict[str, str]:
    path = _find(dataset_dir, "queries.jsonl", "queries.jsonl.gz")
    out: dict[str, str] = {}
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[str(obj["_id"])] = str(obj.get("text", ""))
    return out


def _read_qrels(dataset_dir: Path, split: str) -> dict[str, dict[str, int]]:
    """
    Читает qrels (BEIR tsv `query-id corpus-id score` с заголовком или TREC `qid 0 docid rel`).
    Формат определяется по числу колонок; релевантными считаем записи со score > 0.
    При повторе пары (query, doc) берём максимальный grade.
    """
    path = _find(dataset_dir, f"qrels/{split}.tsv", f"qrels/{split}.tsv.gz")
    rel: dict[str, dict[str, int]] = {}
    with _open_text(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            if len(parts) == 3:
                qid, doc_id, score = parts
            else:  # TREC: qid 0 docid rel
                qid, _, doc_id, score = parts[:4]
            try:
                grade = int(float(score))
            except ValueError:
                continue  # header row (`query-id corpus-id score`) or garbage
            if grade > 0:
                doc_grades = rel.setdefault(str(qid), {})
                doc_grades[str(doc_id)] = max(grade, doc_grades.get(str(doc_id), 0))
    return rel


def read_queries(dataset_dir: Path, split: str) -> list[BeirQuery]:
    """
    Запросы для оценки = пересечение queries.jsonl и qrels/<split>: берём только запросы,
    у которых есть релевантные документы в выбранном split. Порядок детерминированный (по qid).
    """
    texts = _read_query_texts(dataset_dir)
    qrels = _read_qrels(dataset_dir, split)
    out: list[BeirQuery] = []
    for qid in sorted(qrels.keys()):
        text = texts.get(qid)
        if text is None:
            continue
        out.append(BeirQuery(qid=qid, text=text, qrels=qrels[qid]))
    return out
