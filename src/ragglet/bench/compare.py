from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Kind = Literal["data", "config", "env"]

# класс различия задаёт секция; самый длинный префикс побеждает
SECTIONS: dict[str, Kind] = {
    "scenario": "config",
    "dataset": "data",
    "index": "config",
    "index.build_env": "env",
    "rerank": "config",
    "env": "env",
}

# описание и место хранения прогона, а не его настройка
IGNORED = {"scenario.name", "scenario.description", "scenario.tags", "scenario.run.artifacts_dir"}


@dataclass(frozen=True)
class Difference:
    kind: Kind
    path: str  # "env.compute.device", "scenario.rerank.enabled"
    a: Any
    b: Any


def load_run(run_dir: Path) -> dict[str, Any]:
    """scenario.snapshot и manifest одного прогона как одно дерево: {scenario, dataset, index, ...}."""
    snapshot = json.loads((run_dir / "scenario.snapshot.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return {"scenario": snapshot, **manifest}


def section_kind(path: str) -> Kind:
    prefixes = [p for p in SECTIONS if path == p or path.startswith(p + ".")]
    if not prefixes:
        raise ValueError(f"{path!r} is not in any comparison section; add it to SECTIONS")
    return SECTIONS[max(prefixes, key=len)]


def _diff(a: Any, b: Any, path: str) -> Iterator[tuple[str, Any, Any]]:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            yield from _diff(a.get(key), b.get(key), f"{path}.{key}" if path else key)
    elif a != b:
        yield path, a, b


def compare_runs(a: Path, b: Path) -> list[Difference]:
    """
    Различия двух прогонов с классом: data — quality несравнимы; env — часть разницы
    может быть эффектом окружения; config — настройка RAG, то, что сравниваем.
    """
    return [
        Difference(section_kind(path), path, va, vb)
        for path, va, vb in _diff(load_run(a), load_run(b), "")
        if path not in IGNORED
    ]
