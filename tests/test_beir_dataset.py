from __future__ import annotations

from pathlib import Path

from ragglet.datasets.beir import read_corpus, read_queries
from ragglet.datasets.import_dir import import_directory


def _write_beir(root: Path) -> None:
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        '{"_id":"d1","title":"T1","text":"alpha beta"}\n'
        '{"_id":"d2","title":"","text":"gamma"}\n',
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        '{"_id":"q1","text":"alpha?"}\n'
        '{"_id":"q2","text":"gamma?"}\n'
        '{"_id":"q3","text":"orphan without qrels"}\n',
        encoding="utf-8",
    )
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t1\n"
        "q2\td2\t2\n"
        "q2\td1\t0\n",  # score 0 — не релевантно
        encoding="utf-8",
    )


def test_read_corpus(tmp_path):
    _write_beir(tmp_path)
    docs = list(read_corpus(tmp_path))
    assert [d.doc_id for d in docs] == ["d1", "d2"]
    assert docs[0].title == "T1"
    assert docs[0].text == "alpha beta"


def test_read_queries_joins_qrels_and_filters_score(tmp_path):
    _write_beir(tmp_path)
    qs = read_queries(tmp_path, "test")
    # q3 без qrels исключён; порядок по qid
    assert [q.qid for q in qs] == ["q1", "q2"]
    assert qs[0].qrels == {"d1": 1}
    assert qs[1].qrels == {"d2": 2}  # d1 со score 0 отфильтрован, grade сохранён


def test_read_qrels_trec_four_column(tmp_path):
    (tmp_path / "qrels").mkdir(parents=True)
    (tmp_path / "corpus.jsonl").write_text('{"_id":"d1","text":"x"}\n', encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"x?"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text("q1 0 d1 1\n", encoding="utf-8")  # TREC
    qs = read_queries(tmp_path, "test")
    assert qs[0].qrels == {"d1": 1}


def test_read_qrels_duplicate_pair_keeps_max_grade(tmp_path):
    (tmp_path / "qrels").mkdir(parents=True)
    (tmp_path / "corpus.jsonl").write_text('{"_id":"d1","text":"x"}\n', encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"x?"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td1\t1\n", encoding="utf-8"
    )
    assert read_queries(tmp_path, "test")[0].qrels == {"d1": 2}


def test_import_directory_roundtrip(tmp_path):
    corpus = tmp_path / "src" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "a.md").write_text("content A", encoding="utf-8")
    (corpus / "b.md").write_text("content B", encoding="utf-8")
    manifest = tmp_path / "q.jsonl"
    manifest.write_text(
        '{"id":"q1","question":"about A?","relevant":["a.md#0"]}\n'
        '{"id":"q2","question":"about B?","relevant":["b.md#2","b.md#0"]}\n',
        encoding="utf-8",
    )
    out = tmp_path / "beir"

    counts = import_directory(corpus, out, manifest, split="test")

    assert counts["documents"] == 2
    assert {d.doc_id for d in read_corpus(out)} == {"a", "b"}  # расширение отброшено

    qs = read_queries(out, "test")
    rel = {q.qid: sorted(q.qrels) for q in qs}
    assert rel["q1"] == ["a"]
    assert rel["q2"] == ["b"]  # b.md#2 и b.md#0 -> "b", дедуп
