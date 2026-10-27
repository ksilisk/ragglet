"""
Файлы задачи в её каталоге: через них общаются менеджер (процесс сервера) и воркер.
Ядро здесь не импортируется: модуль нужен обеим сторонам, а сервер не должен тянуть torch.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_core import to_jsonable_python

from ragglet.core.issues import Issue, Severity
from ragglet.jobs.protocol import ProgressInfo, TaskFailure, TaskKind, TaskRef, TaskSpec, TaskState

TASK = "task.json"  # что выполнять: пишет менеджер при постановке
LOG = "task.log"  # stdout и stderr воркера
PROGRESS = "progress.json"  # пишет воркер не чаще раза в секунду
RESULT = "result.json"  # итог: пишет воркер перед выходом


@dataclass(frozen=True)
class Outcome:
    state: TaskState  # completed | failed | cancelled
    failure: TaskFailure | None = None
    final_progress: ProgressInfo | None = None
    result: dict[str, Any] | None = None  # у сборки: documents и записанные метаданные


def write_json(path: Path, data: Any) -> None:
    """Атомарно: временный .<имя>.tmp рядом и os.replace, читатель не видит половину файла."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=to_jsonable_python), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any | None:
    """None — файла нет или он не JSON (запись атомарна, так бывает только после ручной правки)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def write_task(spec: TaskSpec) -> None:
    assert spec.run_dir is not None
    write_json(spec.run_dir / TASK, {
        "kind": spec.ref.kind, "id": spec.ref.id, "name": spec.name, "config": spec.config, "dir": str(spec.run_dir),
    })


def read_task(task_dir: Path) -> TaskSpec:
    data = read_json(task_dir / TASK)
    if data is None:
        raise FileNotFoundError(f"no readable {TASK} in {task_dir}")
    return TaskSpec(ref=TaskRef(TaskKind(data["kind"]), data["id"]), name=data["name"], config=data["config"],
                    run_dir=Path(data["dir"]))


def progress_to_json(progress: ProgressInfo | None) -> dict[str, Any] | None:
    return None if progress is None else asdict(progress)


def progress_from_json(data: dict[str, Any] | None) -> ProgressInfo | None:
    if data is None:
        return None
    return ProgressInfo(stage=data["stage"], done=data["done"], total=data["total"],
                        first_at=datetime.fromisoformat(data["first_at"]),
                        updated_at=datetime.fromisoformat(data["updated_at"]))


def failure_to_json(failure: TaskFailure | None) -> dict[str, Any] | None:
    return None if failure is None else asdict(failure)


def failure_from_json(data: dict[str, Any] | None) -> TaskFailure | None:
    if data is None:
        return None
    issues = tuple(Issue(**{**issue, "severity": Severity(issue["severity"])}) for issue in data.get("issues", ()))
    return TaskFailure(type=data["type"], message=data["message"], qid=data.get("qid"), title=data.get("title"),
                       hint=data.get("hint"), issues=issues)


def read_progress(task_dir: Path) -> ProgressInfo | None:
    return progress_from_json(read_json(task_dir / PROGRESS))


def write_outcome(task_dir: Path, outcome: Outcome) -> None:
    write_json(task_dir / RESULT, {
        "state": outcome.state,
        "failure": failure_to_json(outcome.failure),
        "final_progress": progress_to_json(outcome.final_progress),
        "result": outcome.result,
    })


def read_outcome(task_dir: Path) -> Outcome | None:
    data = read_json(task_dir / RESULT)
    if data is None:
        return None
    return Outcome(state=TaskState(data["state"]), failure=failure_from_json(data.get("failure")),
                   final_progress=progress_from_json(data.get("final_progress")), result=data.get("result"))
