from __future__ import annotations

from functools import cache
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ragglet.api.schemas import DatasetSourceMeta, IndexProfileMeta, ScenarioMeta
from ragglet.bench.load import LOAD_MODELS
from ragglet.bench.metrics import metrics_for_level
from ragglet.bench.query_loop import LATENCY_STEPS
from ragglet.config.index_profile import IndexProfileConfig
from ragglet.config.scenario import ScenarioConfig
from ragglet.datasets.sources import SOURCES
from ragglet.retrieval.engine import IMPLEMENTED_STRATEGIES
from ragglet.storage.layout import IMPLEMENTED_BACKENDS

router = APIRouter(tags=["meta"])


def model_defaults(model: type[BaseModel]) -> dict[str, Any]:
    """Умолчания всех полей, включая default_factory: pydantic не вызывает фабрики при генерации схемы."""
    out: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        key = field.alias or name
        if not field.is_required():
            value = field.get_default(call_default_factory=True)
            out[key] = value.model_dump(by_alias=True) if isinstance(value, BaseModel) else value
        elif isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
            out[key] = model_defaults(field.annotation)  # обязательная секция: её необязательные поля
    return out


def _dataset_sources() -> list[DatasetSourceMeta]:
    return [DatasetSourceMeta(id=name, supports_revision=src.supports_revision) for name, src in SOURCES.items()]


@cache
def _scenario_meta() -> ScenarioMeta:
    return ScenarioMeta(
        json_schema=ScenarioConfig.model_json_schema(),
        defaults=model_defaults(ScenarioConfig),
        load_models={name: model.Params.model_json_schema() for name, model in LOAD_MODELS.items()},
        dataset_sources=_dataset_sources(),
        storage_backends=sorted(IMPLEMENTED_BACKENDS),
        retrieval_strategies=list(IMPLEMENTED_STRATEGIES),
        metrics={"doc": metrics_for_level("doc"), "chunk": metrics_for_level("chunk")},
        latency_steps=list(LATENCY_STEPS),
    )


@cache
def _index_profile_meta() -> IndexProfileMeta:
    return IndexProfileMeta(
        json_schema=IndexProfileConfig.model_json_schema(),
        defaults=model_defaults(IndexProfileConfig),
        dataset_sources=_dataset_sources(),
        storage_backends=sorted(IMPLEMENTED_BACKENDS),
    )


@router.get("/meta/scenario")
def scenario_meta() -> ScenarioMeta:
    """JSON Schema сценария и то, чего в ней нет: модели нагрузки, источники, что реализовано."""
    return _scenario_meta()


@router.get("/meta/index-profile")
def index_profile_meta() -> IndexProfileMeta:
    return _index_profile_meta()
