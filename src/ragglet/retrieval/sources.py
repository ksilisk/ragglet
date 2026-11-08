from __future__ import annotations

from ragglet.config.scenario import RetrievalConfig
from ragglet.retrieval.merge import SourceSpec


def active_sources(cfg: RetrievalConfig) -> list[SourceSpec]:
    """Порядок snapshot задаёт индекс метрики; single использует только первый источник."""
    sources = cfg.sources[:1] if cfg.strategy == "single" else cfg.sources
    if cfg.strategy == "single" and not sources:
        return [SourceSpec(name="vec", kind="vector", weight=1.0, params={})]
    return [SourceSpec(name=s.name, kind=s.kind, weight=s.weight, params=s.params) for s in sources]
