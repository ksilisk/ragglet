from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
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


def _spread(values: list[Any], path: str) -> Iterator[tuple[str, tuple[Any, ...]]]:
    """Спускаемся в ключ, только если у всех прогонов там словарь; иначе одно различие на этом пути."""
    if all(isinstance(v, dict) for v in values):
        for key in sorted(set().union(*values)):
            yield from _spread([v.get(key) for v in values], f"{path}.{key}" if path else key)
    elif any(v != values[0] for v in values[1:]):
        yield path, tuple(values)


@dataclass(frozen=True)
class Spread:
    """Различие N прогонов: значение каждого по порядку; None — ключа нет."""

    kind: Kind
    path: str
    values: tuple[Any, ...]


def compare_trees(trees: Sequence[dict[str, Any]]) -> list[Spread]:
    """Различия N деревьев {scenario, dataset, index, ...} — то же правило, что у compare_runs."""
    return [
        Spread(section_kind(path), path, values)
        for path, values in _spread(list(trees), "")
        if path not in IGNORED
    ]


def compare_runs(a: Path, b: Path) -> list[Difference]:
    """
    Различия двух прогонов с классом: data — quality несравнимы; env — часть разницы
    может быть эффектом окружения; config — настройка RAG, то, что сравниваем.
    """
    return [Difference(s.kind, s.path, *s.values) for s in compare_trees([load_run(a), load_run(b)])]


# окружение, от которого зависят задержки: машина, устройство, версии и состояние сервисов (ragglet-23)
LATENCY_ENV = ("env.system", "env.compute", "env.services")


@dataclass(frozen=True)
class Verdict:
    quality_comparable: bool  # нет различий data
    latency_comparable: bool  # нет различий env.system / env.compute / env.services
    confounders: list[str]  # прочие различия env: commit, python, сборка индекса


def verdict(spreads: Sequence[Spread]) -> Verdict:
    """Правило README: разница метрик — эффект настройки RAG, если нет различий data и env."""
    env = [s.path for s in spreads if s.kind == "env"]
    # «env» целиком (у одного прогона секции нет) тоже накрывает system / compute / services
    latency = [p for p in env if any(p == x or p.startswith(x + ".") or x.startswith(p + ".") for x in LATENCY_ENV)]
    return Verdict(
        quality_comparable=not any(s.kind == "data" for s in spreads),
        latency_comparable=not latency,
        confounders=[p for p in env if p not in latency],
    )
