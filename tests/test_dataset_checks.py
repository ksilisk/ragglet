from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from ragglet.bench.dataset import eval_items, eval_sha256
from ragglet.core.issues import Severity, ValidationError
from ragglet.datasets.beir import parse_dataset, read_corpus
from ragglet.datasets.checks import validate_dataset

DEMO = Path(__file__).resolve().parents[1] / "datasets" / "demo"


def _write(root: Path, corpus: str, queries: str, qrels: dict[str, str]) -> Path:
    (root / "qrels").mkdir(parents=True, exist_ok=True)
    (root / "corpus.jsonl").write_text(corpus, encoding="utf-8")
    (root / "queries.jsonl").write_text(queries, encoding="utf-8")
    for split, text in qrels.items():
        (root / "qrels" / f"{split}.tsv").write_text(text, encoding="utf-8")
    return root


def _codes(issues):
    return [(i.code, i.severity.value, i.where) for i in issues]


def test_demo_dataset_is_clean():
    ds, warnings = validate_dataset(DEMO, None, "local:datasets/demo")
    assert warnings == []
    assert len(ds.corpus) == 50


def test_demo_hashes_are_pinned():
    # меняется только при осознанной смене канонизации или правке demo
    ds = parse_dataset(DEMO, ["test"])
    assert ds.corpus_sha256 == "893aa67b4bb4c811afb0afc144ac6fc0b081ae724a39c3f6899b789225c71477"
    assert eval_sha256(eval_items(ds, "test", 0)) == (
        "b68bae536f369e98dedd1f9d9a16e2c90245d347f535c7515b27ccf7b480a376"
    )


def test_all_violations_reported_with_location(tmp_path):
    root = _write(
        tmp_path,
        corpus=(
            '{"_id":"d1","text":"alpha"}\n'
            '{"_id":"d1","text":"duplicate"}\n'  # 2
            '{"_id":"d2","title":"","text":"   "}\n'  # 3: пустой, на него есть ссылка
            '{"_id":"d3","text":""}\n'  # 4: пустой, без ссылок
            "not json\n"  # 5
            '{"title":"no id"}\n'  # 6
        ),
        queries=(
            '{"_id":"q1","text":"alpha?"}\n'
            '{"_id":"q1","text":"again"}\n'  # 2
            '{"_id":"q2","text":"  "}\n'  # 3
        ),
        qrels={
            "test": (
                "query-id\tcorpus-id\tscore\n"
                "q1\td1\t1\n"
                "q1\td1\t2\n"  # 3: повтор пары
                "q1\td9\t1\n"  # 4: нет документа
                "q1\td8\t0\n"  # 5: нет документа, но нерелевантен
                "q2\td2\t1\n"  # 6: пустой документ
                "q7\td1\t1\n"  # 7: нет запроса
                "q7\td1\t1\n"  # 8: тот же неизвестный запрос (+ повтор пары)
                "q1 d1\n"  # 9: две колонки
                "q1\td1\tbad\n"  # 10: нечисловой score
            ),
            "dev": "q1\td1\t0\n",  # нет релевантных
        },
    )

    with pytest.raises(ValidationError) as exc:
        validate_dataset(root, None, "local:broken")

    assert sorted(_codes(exc.value.issues)) == sorted(
        [
            ("json.invalid", "error", "corpus.jsonl:5"),
            ("record.no_id", "error", "corpus.jsonl:6"),
            ("qrels.malformed", "error", "qrels/test.tsv:9"),
            ("qrels.malformed", "error", "qrels/test.tsv:10"),
            ("corpus.duplicate_id", "error", "corpus.jsonl:2"),
            ("corpus.empty_doc", "warning", "corpus.jsonl:4"),
            ("queries.duplicate_id", "error", "queries.jsonl:2"),
            ("queries.empty_text", "error", "queries.jsonl:3"),
            ("qrels.duplicate_pair", "error", "qrels/test.tsv:3"),
            ("qrels.duplicate_pair", "error", "qrels/test.tsv:8"),
            ("qrels.unknown_query", "error", "qrels/test.tsv:7"),
            ("qrels.unknown_doc", "error", "qrels/test.tsv:4"),
            ("qrels.unknown_doc", "warning", "qrels/test.tsv:5"),
            ("qrels.unindexable_doc", "error", "qrels/test.tsv:6"),
            ("split.empty", "error", "qrels/dev.tsv"),
        ]
    )
    message = str(exc.value)
    assert message.startswith("Dataset validation failed: local:broken (13 errors, 2 warnings)")
    assert "qrels/test.tsv:4" in message and "q1 -> 'd9': no such document in corpus" in message


def test_warnings_do_not_fail(tmp_path):
    root = _write(
        tmp_path,
        corpus='{"_id":"d1","text":"alpha"}\n{"_id":"d2","text":""}\n',
        queries='{"_id":"q1","text":"alpha?"}\n',
        qrels={"test": "q1\td1\t1\nq1\tgone\t0\n"},
    )
    _, warnings = validate_dataset(root, ["test"], "local:x")
    assert [i.code for i in warnings] == ["corpus.empty_doc", "qrels.unknown_doc"]
    assert all(i.severity is Severity.WARNING for i in warnings)


def test_query_without_relevant_judgments_is_warned_and_excluded(tmp_path):
    root = _write(
        tmp_path,
        corpus='{"_id":"d1","text":"alpha"}\n{"_id":"d2","text":"beta"}\n',
        queries='{"_id":"q1","text":"alpha?"}\n{"_id":"q2","text":"gamma?"}\n',
        qrels={"test": "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td1\t0\nq2\td2\t-1\n"},
    )
    ds, warnings = validate_dataset(root, ["test"], "local:x")

    assert _codes(warnings) == [("qrels.no_relevant", "warning", "qrels/test.tsv:3")]
    assert [it.qid for it in eval_items(ds, "test", 0)] == ["q1"]


def test_missing_split_lists_available(tmp_path):
    root = _write(
        tmp_path,
        corpus='{"_id":"d1","text":"alpha"}\n',
        queries='{"_id":"q1","text":"alpha?"}\n',
        qrels={"test": "q1\td1\t1\n", "dev": "q1\td1\t1\n"},
    )
    with pytest.raises(ValidationError) as exc:
        validate_dataset(root, ["train"], "local:x")
    assert _codes(exc.value.issues) == [("file.missing", "error", "qrels/train.tsv")]
    assert "available: dev, test" in str(exc.value)


def test_read_corpus_rejects_invalid_record(tmp_path):
    (tmp_path / "corpus.jsonl").write_text('{"_id":"d1","text":"a"}\n{"text":"b"}\n', encoding="utf-8")
    with pytest.raises(ValidationError, match="record.no_id"):
        list(read_corpus(tmp_path))


def _corpus_sha(root: Path, corpus: str, gz: bool = False) -> str:
    root.mkdir(parents=True)
    if gz:
        with gzip.open(root / "corpus.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(corpus)
    else:
        (root / "corpus.jsonl").write_text(corpus, encoding="utf-8")
    return parse_dataset(root, []).corpus_sha256


def test_corpus_hash_depends_on_content_and_order_only(tmp_path):
    base = _corpus_sha(tmp_path / "a", '{"_id":"d1","title":"T","text":"x"}\n{"_id":"d2","text":"y"}\n')
    same = [
        # порядок ключей, пробелы, лишние поля, пустые строки
        '{"text": "x", "_id": "d1", "title": "T", "metadata": {"url": "u"}}\n\n{"_id": "d2", "text": "y"}\n',
        # явный пустой title == отсутствующий
        '{"_id":"d1","title":"T","text":"x"}\n{"_id":"d2","title":"","text":"y"}\n',
    ]
    for i, corpus in enumerate(same):
        assert _corpus_sha(tmp_path / f"same{i}", corpus) == base
    assert _corpus_sha(tmp_path / "gz", '{"_id":"d1","title":"T","text":"x"}\n{"_id":"d2","text":"y"}\n', gz=True) == base

    changed = [
        '{"_id":"d1","title":"T","text":"x!"}\n{"_id":"d2","text":"y"}\n',  # текст
        '{"_id":"d2","text":"y"}\n{"_id":"d1","title":"T","text":"x"}\n',  # порядок
        '{"_id":"d1","title":"T","text":"x"}\n',  # удалён документ
    ]
    for i, corpus in enumerate(changed):
        assert _corpus_sha(tmp_path / f"changed{i}", corpus) != base


def test_eval_hash_depends_on_split_limit_and_grades(tmp_path):
    root = _write(
        tmp_path,
        corpus='{"_id":"d1","text":"a"}\n{"_id":"d2","text":"b"}\n',
        queries='{"_id":"q1","text":"a?"}\n{"_id":"q2","text":"b?"}\n',
        qrels={
            "test": "q1\td1\t1\nq2\td2\t1\n",
            "shuffled": "q2\td2\t1\nq1\td1\t1\n",
            "graded": "q1\td1\t2\nq2\td2\t1\n",
        },
    )
    ds = parse_dataset(root, None)
    base = eval_sha256(eval_items(ds, "test", 0))
    assert eval_sha256(eval_items(ds, "shuffled", 0)) == base  # порядок строк qrels не важен
    assert eval_sha256(eval_items(ds, "graded", 0)) != base
    assert eval_sha256(eval_items(ds, "test", 1)) != base
