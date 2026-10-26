from __future__ import annotations

from collections import defaultdict

import yaml

from ragglet.api.deps import Services
from ragglet.api.library import ConfigLibrary
from ragglet.api.probes import normalize_endpoint
from ragglet.api.schemas import CollectionRef
from ragglet.config.scenario import ScenarioConfig
from ragglet.core.issues import ValidationError
from ragglet.storage.layout import StorageTarget, build_storage_layout

CollectionKey = tuple[str, str]  # (нормализованный endpoint, коллекция)


def targets_of(config: object) -> list[StorageTarget]:
    try:
        return build_storage_layout(config.storage).targets  # type: ignore[attr-defined]
    except ValidationError:  # форма storage неверна: целей нет
        return []


def collection_refs(config: object) -> list[CollectionRef]:
    return [CollectionRef(endpoint=t.endpoint, collection=t.collection) for t in targets_of(config)]


def users(library: ConfigLibrary) -> dict[CollectionKey, list[str]]:
    """Коллекция -> id сценариев (профилей) библиотеки, у которых она среди целей storage."""
    found: dict[CollectionKey, list[str]] = defaultdict(list)
    for entry in library.list():
        if entry.config is not None:
            for target in targets_of(entry.config):
                found[(normalize_endpoint(target.endpoint), target.collection)].append(entry.id)
    return found


def library_users(svc: Services) -> tuple[dict[CollectionKey, list[str]], dict[CollectionKey, list[str]]]:
    return users(svc.scenarios), users(svc.profiles)


def snapshot_yaml(snapshot: dict, **changes: object) -> str:
    """Snapshot прогона как YAML сценария: с алиасами (async, а не async_enabled), как пишут руками."""
    data = ScenarioConfig.model_validate(snapshot).model_dump(by_alias=True)
    data.update(changes)
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
