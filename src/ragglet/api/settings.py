from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ServerSettings(BaseSettings):
    """
    Настройки `ragglet serve`: переменные RAGGLET_* и .env в рабочем каталоге. Относительные
    пути — от рабочего каталога сервера, умолчания работают из корня репозитория.
    """

    model_config = SettingsConfigDict(env_prefix="RAGGLET_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"  # авторизации нет: наружу — только явно
    port: int = 8000
    qdrant_endpoints: Annotated[list[str], NoDecode] = ["http://localhost:6333"]
    redis_url: str | None = "redis://localhost:6379/0"
    scenarios_dir: Path = Path("configs/scenarios")
    index_profiles_dir: Path = Path("configs/indexes")
    artifacts_dir: Path = Path("results/runs")
    builds_dir: Path = Path("results/builds")  # каталоги сборок: <builds_dir>/<id>/
    dataset_roots: Annotated[list[Path], NoDecode] = [Path("datasets")]
    ui_dir: Path | None = None  # собранный UI (ragglet-31); None — UI не раздаётся
    probe_timeout_s: float = 2.0

    @field_validator("qdrant_endpoints", "dataset_roots", mode="before")
    @classmethod
    def _comma_separated(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("redis_url", "ui_dir", mode="before")
    @classmethod
    def _empty_is_none(cls, value: Any) -> Any:
        return None if value == "" else value

    def describe(self) -> str:
        paths = {
            "scenarios": self.scenarios_dir,
            "index profiles": self.index_profiles_dir,
            "artifacts": self.artifacts_dir,
            "builds": self.builds_dir,
            "dataset roots": ", ".join(str(p.resolve()) for p in self.dataset_roots),
            "ui": self.ui_dir.resolve() if self.ui_dir else "not served",
        }
        lines = [
            f"listening: http://{self.host}:{self.port} (API /v1, docs /v1/docs)",
            f"qdrant: {', '.join(self.qdrant_endpoints)}",
            f"redis: {self.redis_url or 'not set'}",
        ]
        lines += [f"{name}: {value.resolve() if isinstance(value, Path) else value}" for name, value in paths.items()]
        return "\n".join(lines)
