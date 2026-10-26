from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ragglet.config.scenario import ScenarioConfig

SNAPSHOT_FILE = "scenario.snapshot.json"


@dataclass(frozen=True)
class ArtifactPaths:
    base_dir: Path
    run_dir: Path
    timestamp: str

def make_artifact_paths(artifacts_dir: str, scenario_name: str, run_id: str | None = None) -> ArtifactPaths:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_dir = Path(artifacts_dir)
    run_dir = base_dir / (run_id or f"{scenario_name}_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    # каталог с заданным id реестр создаёт при постановке (в нём run.json и файлы менеджера);
    # snapshot в нём — значит, там уже был прогон: ошибка, а не смесь артефактов двух прогонов
    if run_id is not None and (run_dir / SNAPSHOT_FILE).exists():
        raise FileExistsError(f"run directory already holds a run: {run_dir}")
    return ArtifactPaths(base_dir=base_dir, run_dir=run_dir, timestamp=ts)

def save_config_snapshot(run_dir: Path, cfg: ScenarioConfig) -> None:
    # сохраняем нормализованный дамп (без алиасов) — удобно для воспроизводимости
    payload = cfg.model_dump()
    (run_dir / SNAPSHOT_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    (run_dir / "scenario.snapshot.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), "utf-8")


def save_manifest(run_dir: Path, manifest: dict) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
