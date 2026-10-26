from __future__ import annotations

import pytest
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from ragglet.config import index_profile, scenario
from ragglet.config.errors import FieldError

SCENARIO = {
    "name": "s",
    "dataset": {"id": "datasets/demo"},
    "storage": {"backend": {"type": "qdrant", "endpoint": "http://q:6333", "collection": "c"}},
}
PROFILE = {
    "id": "p",
    "dataset": {"id": "datasets/demo"},
    "embedding": {"model": "all-MiniLM-L6-v2"},
    "storage": {"endpoint": "http://q:6333", "collection": "c"},
}

# model-валидатор -> [(корневая модель, переопределения, ожидаемый путь поля)]
CASES = {
    "scenario.ScenarioConfig._generation_requires_context": [
        (scenario.ScenarioConfig, {"generation": {"enabled": True, "model": "m"}}, "generation.enabled"),
    ],
    "scenario.ScenarioConfig._shared_cache_single_repeat": [
        (
            scenario.ScenarioConfig,
            {
                "run": {"repeats": 2},
                "cache": {"enabled": True, "embedding": {"endpoint": "redis://r", "namespace": "n"}},
            },
            "cache.embedding.namespace",
        ),
    ],
    "scenario.ScenarioConfig._quality_k_within_final_depth": [
        (scenario.ScenarioConfig, {"metrics": {"quality": {"k": [50]}}}, "metrics.quality.k"),
    ],
    "scenario.RerankConfig._model_required_if_enabled": [
        (scenario.ScenarioConfig, {"rerank": {"enabled": True}}, "rerank.model"),
    ],
    "scenario.GenerationConfig._model_required_if_enabled": [
        (scenario.ScenarioConfig, {"generation": {"enabled": True}}, "generation.model"),
    ],
    "scenario.CacheConfig._at_least_one_layer_if_enabled": [
        (scenario.ScenarioConfig, {"cache": {"enabled": True}}, "cache.enabled"),
    ],
    "scenario.RetrievalConfig._single_keyword_not_supported": [
        (scenario.ScenarioConfig, {"retrieval": {"sources": [{"name": "k", "kind": "keyword"}]}}, "retrieval.sources"),
    ],
    "scenario.StorageConfig._validate_storage_shape": [
        (scenario.ScenarioConfig, {"storage": {**SCENARIO["storage"], "mode": "sharded"}}, "storage.backends"),
    ],
    "scenario.DatasetConfig._known_source": [
        (scenario.ScenarioConfig, {"dataset": {"source": "s3", "id": "x"}}, "dataset.source"),
        (scenario.ScenarioConfig, {"dataset": {"id": "x", "revision": "v1"}}, "dataset.revision"),
    ],
    "scenario.LoadConfig._validate_params": [
        (scenario.ScenarioConfig, {"run": {"load": {"type": "open"}}}, "run.load.type"),
    ],
    "index_profile.ChunkingConfig._validate_sizes": [
        (index_profile.IndexProfileConfig, {"chunking": {"method": "none", "chunk_size": 100}}, "chunking.chunk_size"),
        (index_profile.IndexProfileConfig, {"chunking": {"chunk_size": 100, "chunk_overlap": 100}},
         "chunking.chunk_overlap"),
    ],
    "index_profile.StorageConfig._validate_storage_shape": [
        (index_profile.IndexProfileConfig, {"storage": {"collection": "c"}}, "storage.endpoint"),
        (index_profile.IndexProfileConfig, {"storage": {**PROFILE["storage"], "mode": "sharded"}}, "storage.backends"),
    ],
    "index_profile.DatasetSourceConfig._known_source": [
        (index_profile.IndexProfileConfig, {"dataset": {"source": "s3", "id": "x"}}, "dataset.source"),
    ],
}


def _model_validators() -> set[str]:
    found = set()
    for module in (scenario, index_profile):
        short = module.__name__.rsplit(".", 1)[1]
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == module.__name__:
                found |= {f"{short}.{obj.__name__}.{name}" for name in obj.__pydantic_decorators__.model_validators}
    return found


def test_every_model_validator_has_a_case():
    # новый model-валидатор должен бросать FieldError и попасть сюда
    assert set(CASES) == _model_validators()


@pytest.mark.parametrize(
    "model, overrides, path",
    [case for cases in CASES.values() for case in cases],
    ids=[f"{name}:{path}" for name, cases in CASES.items() for _, _, path in cases],
)
def test_model_validator_names_its_field(model, overrides, path):
    base = SCENARIO if model is scenario.ScenarioConfig else PROFILE
    with pytest.raises(PydanticValidationError) as exc:
        model.model_validate({**base, **overrides})

    [err] = exc.value.errors()
    cause = err["ctx"]["error"]
    assert isinstance(cause, FieldError)
    assert ".".join([*map(str, err["loc"]), cause.path]) == path
