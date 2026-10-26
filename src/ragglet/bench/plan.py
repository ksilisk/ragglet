from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragglet.config.scenario import ScenarioConfig


@dataclass(frozen=True)
class RunPlan:
    repeats: int
    queries_per_repeat: int
    total: int  # запросов за прогон: total у progress("queries", ...)
    load: dict[str, Any]  # run.load: type и параметры модели нагрузки


def run_plan(cfg: ScenarioConfig, n_queries: int) -> RunPlan:
    return RunPlan(
        repeats=cfg.run.repeats,
        queries_per_repeat=n_queries,
        total=n_queries * cfg.run.repeats,
        load={"type": cfg.run.load.type, **cfg.run.load.params},
    )
