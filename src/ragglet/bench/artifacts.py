from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import yaml

from ragglet.config.scenario import ScenarioConfig


@dataclass(frozen=True)
class ArtifactPaths:
    base_dir: Path
    run_dir: Path
    timestamp: str

def make_artifact_paths(artifacts_dir: str, scenario_name: str, run_id: str | None = None) -> ArtifactPaths:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_dir = Path(artifacts_dir)
    run_dir = base_dir / (run_id or f"{scenario_name}_{ts}")
    # каталог с заданным id создаёт только этот прогон: совпадение — ошибка, а не смесь артефактов
    run_dir.mkdir(parents=True, exist_ok=run_id is None)
    return ArtifactPaths(base_dir=base_dir, run_dir=run_dir, timestamp=ts)

def save_config_snapshot(run_dir: Path, cfg: ScenarioConfig) -> None:
    # сохраняем нормализованный дамп (без алиасов) — удобно для воспроизводимости
    payload = cfg.model_dump()
    (run_dir / "scenario.snapshot.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    (run_dir / "scenario.snapshot.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), "utf-8")


def save_manifest(run_dir: Path, manifest: dict) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
