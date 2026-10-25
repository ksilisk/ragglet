from __future__ import annotations

from fastapi import APIRouter

from ragglet.api.deps import Svc
from ragglet.api.problems import problem_error
from ragglet.api.schemas import LogChunkOut, TaskItem
from ragglet.jobs.protocol import JobManager, TaskFinished, TaskRef, TaskState

router = APIRouter(tags=["tasks"])

LOG_LIMIT = 65536


@router.get("/tasks")
async def list_tasks(svc: Svc) -> list[TaskItem]:
    """Текущая задача стенда, затем очередь по порядку. Завершённые — это прогоны и сборки в реестре."""
    return [TaskItem.build(status) for status in await svc.jobs.queue()]


def parse_include(include: str | None, allowed: set[str]) -> set[str]:
    parts = {part.strip() for part in (include or "").split(",") if part.strip()}
    unknown = parts - allowed
    if unknown:
        raise problem_error(422, "request.include", "Request is invalid",
                            f"unknown include {sorted(unknown)}; allowed: {sorted(allowed)}", path="include")
    return parts


async def cancel(jobs: JobManager, ref: TaskRef, state: TaskState) -> None:
    try:
        known = await jobs.status(ref)
        if known is None:  # менеджер задачу не знает: она завершилась (или до рестарта сервера)
            raise TaskFinished(f"{ref.kind} {ref.id} is not active ({state})")
        await jobs.cancel(ref)
    except TaskFinished as e:
        raise problem_error(409, "task.finished", "Task is not active", str(e), where=f"{ref.kind}s/{ref.id}") from e


async def read_log(jobs: JobManager, ref: TaskRef, offset: int) -> LogChunkOut:
    chunk = await jobs.log(ref, offset=offset, limit=LOG_LIMIT)
    return LogChunkOut(text=chunk.text, offset=offset, next_offset=chunk.next_offset, complete=chunk.complete)

