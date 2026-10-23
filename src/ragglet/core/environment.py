from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def git_commit(path: Path = PACKAGE_DIR) -> str | None:
    """HEAD репозитория, из которого запущен ragglet; None вне git-checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _cpu_model() -> str | None:
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip() or None
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def system_info() -> dict[str, Any]:
    # sched_getaffinity учитывает cpuset контейнера; на macOS его нет
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cpus": cpus,
        "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
    }


ENV_FACTS: dict[str, Callable[[], Any]] = {
    "commit": git_commit,
    "python": platform.python_version,
    "system": system_info,
}


def capture_env(compute: dict[str, Any]) -> dict[str, Any]:
    """Окружение процесса; `compute` — где загруженная модель считает эмбеддинги."""
    env = {name: fact() for name, fact in ENV_FACTS.items()}
    env["compute"] = compute
    return env


class Service(Protocol):
    async def describe(self) -> dict[str, Any]:
        """Минимум `type` и `version`; остальные поля — на усмотрение типа сервиса."""
        ...


async def describe_services(services: Mapping[str, Mapping[str, Service]]) -> dict[str, dict[str, Any]]:
    """Роль (storage, cache, ...) -> имя -> описание, которое сервис отдаёт о себе."""
    return {
        role: {name: await service.describe() for name, service in by_name.items()}
        for role, by_name in services.items()
    }
