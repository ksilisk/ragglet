from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from ragglet.api.deps import Services
from ragglet.api.library import ConfigLibrary
from ragglet.api.problems import install_handlers
from ragglet.api.routers import (
    builds,
    datasets,
    indexes,
    library,
    meta,
    results,
    runs,
    stand,
    tasks,
)
from ragglet.api.schemas import RootInfo
from ragglet.api.settings import ServerSettings
from ragglet.config.index_profile import IndexProfileConfig
from ragglet.config.scenario import ScenarioConfig
from ragglet.jobs.memory import InMemoryJobManager
from ragglet.jobs.protocol import JobManager
from ragglet.registry.files import FileRegistry
from ragglet.registry.protocol import Registry


class SpaStaticFiles(StaticFiles):
    """Файл UI как есть, любой другой путь вне /v1 — index.html: прямые ссылки вроде /runs/<id>."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        if path == "v1" or path.startswith("v1/"):
            raise StarletteHTTPException(404, "Not Found")  # под /v1 фолбэка нет
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as e:
            if e.status_code != 404:
                raise
            return await super().get_response("index.html", scope)


def create_app(
        settings: ServerSettings | None = None,
        *,
        jobs: JobManager | None = None,
        registry: Registry | None = None,
) -> FastAPI:
    """Приложение API; без registry — реестр на файлах, без jobs — заглушка менеджера (её зовут тесты)."""
    settings = settings or ServerSettings()
    registry = registry or FileRegistry(settings.artifacts_dir, settings.builds_dir)
    jobs = jobs or InMemoryJobManager(registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # менеджер задач (ragglet-29) живёт вместе с сервером, при остановке гасит процессы задач
        async with AsyncExitStack() as stack:
            if isinstance(jobs, AbstractAsyncContextManager):
                await stack.enter_async_context(jobs)
            await registry.recover(jobs)  # ragglet-30: после падения сервера — осиротевшие queued/running
            yield

    app = FastAPI(
        title="Ragglet",
        summary="Single entry point to the benchmark stand: scenarios, indexes, runs.",
        version="1",
        openapi_url="/v1/openapi.json",
        docs_url="/v1/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.services = Services(
        settings=settings,
        scenarios=ConfigLibrary(settings.scenarios_dir, ScenarioConfig, "name", "scenario",
                                "Scenario config is invalid"),
        profiles=ConfigLibrary(settings.index_profiles_dir, IndexProfileConfig, "id", "index profile",
                               "Index profile config is invalid"),
        jobs=jobs,
        registry=registry,
    )
    install_handlers(app)
    for module in (meta, stand, library, indexes, datasets, runs, results, builds, tasks):
        app.include_router(module.router, prefix="/v1")

    if settings.ui_dir is not None:
        app.mount("/", SpaStaticFiles(directory=Path(settings.ui_dir), html=True), name="ui")
    else:
        @app.get("/", include_in_schema=False)
        def root() -> RootInfo:
            return RootInfo()

    return app
