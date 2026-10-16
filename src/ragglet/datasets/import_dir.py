from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path, PurePosixPath


def _iter_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p


def _doc_id(raw: str) -> str:
    """doc_id из относительного пути/ссылки: отрезаем chunk-суффикс `#idx` и расширение."""
    base = raw.rsplit("#", 1)[0]
    return PurePosixPath(base).with_suffix("").as_posix()


def import_directory(
    corpus_dir: Path,
    out_dir: Path,
    queries_jsonl: Path | None = None,
    split: str = "test",
) -> dict[str, int]:
    """
    Конвертирует директорию с текстовым корпусом (+ опциональный legacy-манифест запросов)
    в BEIR-датасет (corpus.jsonl / queries.jsonl / qrels/<split>.tsv).

    Authoring-инструмент: собрать BEIR-датасет из своих .md/.txt. Не является runtime-форматом.

    - corpus_dir: .md/.txt файлы; doc_id = относительный путь без расширения.
    - queries_jsonl: legacy-манифест {"id","question","relevant":["doc#idx" | "doc"]};
      relevant приводится к doc-level id (chunk-суффикс и расширение отбрасываются).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids: set[str] = set()
    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as fc:
        for path in _iter_files(corpus_dir):
            doc_id = _doc_id(path.relative_to(corpus_dir).as_posix())
            text = path.read_text(encoding="utf-8", errors="ignore")
            doc_ids.add(doc_id)
            fc.write(json.dumps({"_id": doc_id, "title": "", "text": text}, ensure_ascii=False) + "\n")

    counts = {"documents": len(doc_ids), "queries": 0, "qrels": 0}
    if queries_jsonl is None:
        return counts

    (out_dir / "qrels").mkdir(exist_ok=True)
    with (
        queries_jsonl.open("r", encoding="utf-8") as fin,
        (out_dir / "queries.jsonl").open("w", encoding="utf-8") as fq,
        (out_dir / "qrels" / f"{split}.tsv").open("w", encoding="utf-8") as fr,
    ):
        fr.write("query-id\tcorpus-id\tscore\n")
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("id", i))
            question = str(obj.get("question", obj.get("text", "")))
            fq.write(json.dumps({"_id": qid, "text": question}, ensure_ascii=False) + "\n")
            counts["queries"] += 1

            relevant = obj.get("relevant", [])
            if isinstance(relevant, str):
                relevant = [relevant]
            written: set[str] = set()
            for r in relevant:
                doc_id = _doc_id(str(r))
                if doc_id in written:
                    continue
                written.add(doc_id)
                fr.write(f"{qid}\t{doc_id}\t1\n")
                counts["qrels"] += 1

    return counts
