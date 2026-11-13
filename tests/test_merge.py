from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from ragglet.retrieval.merge import RetrievedItem, interleave_merge, rrf_merge, weighted_sum_merge


def item(cid, score=1.0, source="a", doc=None):
    return RetrievedItem(f"{source}:{cid}", cid, doc or cid, score, source, {"text": cid})


def ids(items):
    return [i.chunk_id for i in items]


def merge(method, ranked, weights=None, **kwargs):
    kwargs.setdefault("top_k", 10)
    if method == "rrf":
        return rrf_merge(ranked, weights or {}, rrf_k=60, deduplicate=True, **kwargs)
    if method == "sum":
        return weighted_sum_merge(ranked, weights or {}, **kwargs)
    return interleave_merge(ranked, weights or {}, weighted=method == "weighted", **kwargs)


METHODS = ["rrf", "sum", "interleave", "weighted"]


def test_shards_example_and_separate_score_contract():
    ranked = {"a": [item("a", .90), item("b", .89)], "b": [item("c", .20, "b"), item("d", .10, "b")]}
    before = deepcopy(ranked)
    rrf = merge("rrf", ranked)
    assert ids(rrf) == ["a", "c", "b", "d"]
    assert [i.score_fused for i in rrf] == pytest.approx([1/61, 1/61, 1/62, 1/62])
    normalized = merge("sum", ranked)
    assert ids(normalized) == ["a", "c", "b", "d"]
    assert [i.score_fused for i in normalized] == [1, 1, 0, 0]
    raw = merge("sum", ranked, normalize="none")
    assert ids(raw) == ["a", "b", "c", "d"]
    assert [i.score_raw for i in raw] == [.90, .89, .20, .10]
    assert [i.score_fused for i in raw] == [.90, .89, .20, .10]
    assert ranked == before


@pytest.mark.parametrize("method", METHODS)
def test_local_duplicates_do_not_change_result_or_consume_slots(method):
    a, b = item("x", 3), item("y", 2)
    baseline = {"a": [a, b], "b": [item("y", 20, "b"), item("z", 10, "b")]}
    duplicated = {**baseline, "a": [a, replace(a, score_raw=1000), b]}
    assert merge(method, duplicated) == merge(method, baseline)
    assert len(merge(method, baseline, top_k=3)) == 3


def test_rrf_and_sum_shared_chunk_weights_and_representative():
    ranked = {"a": [item("x", 5), item("y", 3)], "b": [item("y", 200, "b"), item("z", 100, "b")]}
    rrf = merge("rrf", ranked, {"a": 2, "b": 1})
    assert ids(rrf) == ["y", "x", "z"]
    assert rrf[0].score_fused == pytest.approx(2/62 + 1/61)
    assert rrf[0].source == "b"  # better local rank, not largest raw score
    summed = merge("sum", ranked, {"a": 2, "b": 1})
    assert ids(summed) == ["x", "y", "z"]
    assert [i.score_fused for i in summed] == [2, 1, 0]
    scaled = merge("rrf", ranked, {"a": 20, "b": 10})
    assert ids(scaled) == ids(rrf)
    assert [i.score_fused for i in scaled] == pytest.approx([i.score_fused * 10 for i in rrf])
    assert ids(rrf_merge(ranked, {}, 60, 10, False)) == ids(rrf_merge(ranked, {}, 60, 10, True))
    assert weighted_sum_merge(ranked, {}, 10, False) == weighted_sum_merge(ranked, {}, 10, True)


def test_rrf_k_changes_balance_of_consensus_and_top_rank():
    ranked = {"a": [item("a"), item("x")], "b": [item("b"), item("x", source="b")]}
    for k in [1, 10, 100]:
        result = rrf_merge(ranked, {}, k, 3, True)
        assert result[0].chunk_id == "x"
        assert result[0].score_fused == pytest.approx(2 / (k + 2))


@pytest.mark.parametrize("scores", [[3], [3, 3], [3, 3 + 5e-10]])
def test_minmax_degenerate_lists_keep_candidates_in_input_order(scores):
    ranked = {"a": [item(str(i), s) for i, s in enumerate(scores)], "b": []}
    result = merge("sum", ranked)
    assert ids(result) == [str(i) for i in range(len(scores))]
    assert all(i.score_fused == 0 for i in result)
    assert all(i.ranking_score == 0 for i in result)


def test_minmax_above_epsilon_and_negative_raw_scores():
    result = merge("sum", {"a": [item("x", -1), item("y", -1 - 2e-9)]})
    assert [i.score_fused for i in result] == [1, 0]
    ranked = {"a": [item("x", -.1), item("y", -.2)], "b": [item("x", -.4, "b")]}
    result = merge("sum", ranked, normalize="none")
    assert ids(result) == ["y", "x"]  # absent contribution is zero, not a penalty
    assert [i.score_fused for i in result] == pytest.approx([-.2, -.5])


@pytest.mark.parametrize("method", METHODS)
def test_zero_weight_empty_lists_and_document_chunks(method):
    ranked = {"zero": [item("ignored")], "a": [item("x#0", doc="x"), item("x#1", doc="x")], "b": []}
    assert ids(merge(method, ranked, {"zero": 0})) == ["x#0", "x#1"]
    assert merge(method, {}) == []
    assert merge(method, {"a": []}) == []
    assert len(merge(method, ranked, {"zero": 0}, top_k=1)) == 1


def test_weighted_interleave_distribution_and_exhaustion():
    ranked = {s: [item(f"{s}{i}", source=s) for i in range(8)] for s in ["a", "b"]}
    result = merge("weighted", ranked, {"a": 2, "b": 1}, top_k=6)
    assert [i.source for i in result] == ["a", "b", "a", "a", "b", "a"]
    assert all(i.score_fused is None for i in result)
    assert len(merge("weighted", ranked, top_k=5)) == 5  # old quotas returned 4
    ranked["a"] = ranked["a"][:1]
    assert len(merge("weighted", ranked, {"a": 9, "b": 1}, top_k=7)) == 7


def test_round_robin_preserves_cursor_when_middle_source_exhausts():
    ranked = {"a": [item("x"), item("a")], "b": [item("x", source="b")],
              "c": [item("c0", source="c"), item("c1", source="c")]}
    assert ids(merge("interleave", ranked)) == ["x", "c0", "a", "c1"]


@pytest.mark.parametrize("weighted", [False, True])
def test_interleave_dedup_flag_only_controls_cross_source_occurrences(weighted):
    ranked = {"a": [item("x"), item("x"), item("a")], "b": [item("x", source="b"), item("b", source="b")]}
    assert len(interleave_merge(ranked, {}, 10, weighted=weighted)) == 3
    assert ids(interleave_merge(ranked, {}, 10, deduplicate=False, weighted=weighted)) == ["x", "x", "a", "b"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("weights", [{"a": -1}, {"a": float("nan")}, {"a": float("inf")}, {"a": 0}])
def test_invalid_weights(method, weights):
    with pytest.raises(ValueError):
        merge(method, {"a": [item("x")]}, weights)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", [item(""), item("x", float("nan")), item("x", float("inf")), replace(item("x"), doc_id=" ")])
def test_invalid_candidates(method, bad):
    with pytest.raises(ValueError):
        merge(method, {"a": [bad]})


@pytest.mark.parametrize("method", METHODS)
def test_conflicting_doc_identity_is_rejected(method):
    with pytest.raises(ValueError, match="conflicting doc_id"):
        merge(method, {"a": [item("x")], "b": [item("x", doc="other")]})


def test_finite_inputs_cannot_produce_infinite_fused_score():
    with pytest.raises(ValueError, match="overflow"):
        merge("sum", {"a": [item("x", 1e308)]}, {"a": 10}, normalize="none")
    with pytest.raises(ValueError, match="overflow"):
        merge("sum", {"a": [item("x", 1e308), item("y", -1e308)]})
    with pytest.raises(ValueError, match="overflow"):
        merge("sum", {"a": [item("x", 1e308)], "b": [item("x", 1e308, "b")]}, normalize="none")


def test_invalid_algorithm_parameters():
    with pytest.raises(ValueError, match="rrf_k"):
        rrf_merge({}, {}, 0, 1, True)
    with pytest.raises(ValueError, match="normalization"):
        weighted_sum_merge({}, {}, 1, normalize="zscore")
    with pytest.raises(ValueError, match="eps"):
        weighted_sum_merge({}, {}, 1, eps=float("nan"))
    for method in METHODS:
        with pytest.raises(ValueError, match="top_k"):
            merge(method, {}, top_k=0)
