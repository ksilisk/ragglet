"""Анализатор серии статьи (scripts/analyze_article_runs.py): прогоны серии выбираются по тегу через API."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyze_article_runs", ROOT / "scripts" / "analyze_article_runs.py")
analyzer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyzer)


def test_series_runs_by_tag(monkeypatch, tmp_path):
    items = [
        {"id": "article_b0_cache_disabled_1", "scenario": "article_b0_cache_disabled", "run_dir": "results/runs/b0_1"},
        {"id": "article_b0_cache_disabled_2", "scenario": "article_b0_cache_disabled", "run_dir": str(tmp_path)},
        {"id": "article_smoke_top_n_1", "scenario": "article_smoke_top_n", "run_dir": "results/runs/s_1"},
    ]
    asked = []

    def fake_urlopen(url, timeout):
        asked.append(url)
        return io.BytesIO(json.dumps({"k": 10, "total": len(items), "items": items}).encode())

    monkeypatch.setattr(analyzer, "urlopen", fake_urlopen)

    runs = analyzer.tagged_run_dirs("http://stand:8000/", "article-2026-11", ["article_smoke_top_n_1"])

    assert runs == {"article_b0_cache_disabled": [ROOT / "results/runs/b0_1", tmp_path]}
    (url,) = asked
    assert url.startswith("http://stand:8000/v1/runs?")
    assert parse_qs(urlsplit(url).query) == {"tag": ["article-2026-11"], "state": ["completed"],
                                             "sort": ["enqueued_at"], "limit": ["500"]}


def test_analyze_needs_a_tag():
    parse = analyzer.build_parser().parse_args
    args = parse(["analyze", "--tag", "s1", "--exclude", "a", "--exclude", "b"])
    assert (args.tag, args.exclude) == ("s1", ["a", "b"])
