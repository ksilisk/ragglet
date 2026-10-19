from __future__ import annotations

import copy
import json

import pytest

from ragglet.bench.compare import Difference, compare_runs

SNAPSHOT = {
    "name": "s0",
    "description": None,
    "tags": [],
    "run": {"repeats": 1, "artifacts_dir": "results/runs"},
    "rerank": {"enabled": False, "model": None},
}
MANIFEST = {
    "dataset": {"revision": None, "corpus_sha256": "a" * 64, "eval_sha256": "b" * 64},
    "index": {
        "profile_id": "demo",
        "chunking": {"method": "recursive", "chunk_size": 800, "chunk_overlap": 120},
        "build_env": {"commit": "c" * 40, "compute": {"device": "cuda:0"}},
    },
    "rerank": None,
    "env": {
        "commit": "c" * 40,
        "compute": {"device": "mps"},
        "services": {"storage": {"default": {"type": "qdrant", "version": "1.17.0"}}},
    },
}


def _run(tmp_path, name, edit=None):
    snapshot, manifest = copy.deepcopy(SNAPSHOT), copy.deepcopy(MANIFEST)
    if edit:
        edit(snapshot, manifest)
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "scenario.snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_same_setup_has_no_differences_even_with_other_name_and_location(tmp_path):
    def rename(snapshot, manifest):
        snapshot.update(name="s0-again", description="retry", tags=["x"])
        snapshot["run"]["artifacts_dir"] = "elsewhere"

    assert compare_runs(_run(tmp_path, "a"), _run(tmp_path, "b", rename)) == []


def _edit_one(snapshot, manifest):
    manifest["dataset"]["eval_sha256"] = "e" * 64
    manifest["index"]["chunking"]["chunk_size"] = 400
    manifest["index"]["build_env"]["commit"] = "d" * 40
    manifest["env"]["compute"]["device"] = "cpu"
    manifest["env"]["services"]["cache"] = {"embedding": {"type": "redis", "version": "7.4.1"}}
    manifest["rerank"] = {"revision": "f" * 40}
    snapshot["rerank"]["enabled"] = True


def test_each_difference_is_classified_by_its_section(tmp_path):
    diffs = compare_runs(_run(tmp_path, "a"), _run(tmp_path, "b", _edit_one))

    assert diffs == [
        Difference("data", "dataset.eval_sha256", "b" * 64, "e" * 64),
        Difference("env", "env.compute.device", "mps", "cpu"),
        Difference("env", "env.services.cache", None, {"embedding": {"type": "redis", "version": "7.4.1"}}),
        Difference("env", "index.build_env.commit", "c" * 40, "d" * 40),
        Difference("config", "index.chunking.chunk_size", 800, 400),
        Difference("config", "rerank", None, {"revision": "f" * 40}),
        Difference("config", "scenario.rerank.enabled", False, True),
    ]


def test_unclassified_manifest_section_is_rejected(tmp_path):
    def add_section(snapshot, manifest):
        manifest["timings"] = {"wall_s": 12.5}

    with pytest.raises(ValueError, match="'timings' is not in any comparison section"):
        compare_runs(_run(tmp_path, "a"), _run(tmp_path, "b", add_section))
