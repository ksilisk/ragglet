from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Request

from ragglet.api.library import ConfigLibrary
from ragglet.api.settings import ServerSettings
from ragglet.jobs.protocol import JobManager
from ragglet.registry.protocol import Registry


@dataclass
class Services:
    """Всё, что живёт в процессе сервера; один на приложение."""

    settings: ServerSettings
    scenarios: ConfigLibrary
    profiles: ConfigLibrary
    jobs: JobManager
    registry: Registry
    compute: dict[str, Any] | None = None  # predicted_compute(): считается один раз, лениво


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


Svc = Annotated[Services, Depends(get_services)]
