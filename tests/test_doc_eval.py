from __future__ import annotations

from ragglet.bench.metrics import mrr_at_k, project_to_docs, recall_at_k


def test_project_dedup_keeps_best_rank():
    chunk_ids = ["d1#3", "d2#0", "d1#0", "d3#1"]
    mapping = {"d1#3": "d1", "d2#0": "d2", "d1#0": "d1", "d3#1": "d3"}
    assert project_to_docs(chunk_ids, mapping) == ["d1", "d2", "d3"]


def test_doc_level_recall_and_mrr_use_deduped_sequence():
    # два чанка d2 идут первыми, затем d1; без дедупа ранги были бы искажены
    chunk_ids = ["d2#0", "d2#1", "d1#0"]
    mapping = {"d2#0": "d2", "d2#1": "d2", "d1#0": "d1"}
    docs = project_to_docs(chunk_ids, mapping)
    assert docs == ["d2", "d1"]

    truth = {"d1"}
    assert recall_at_k(docs, truth, 1) == 0.0
    assert recall_at_k(docs, truth, 2) == 1.0
    assert mrr_at_k(docs, truth, 10) == 0.5
