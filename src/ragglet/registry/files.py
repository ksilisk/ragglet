"""
Реестр на файлах. Запись прогона — run.json в его каталоге, рядом с артефактами ядра;
запись сборки — build.json в results/builds/<id>/. Индекс живёт в памяти и строится
сканом. Каталог прогона без run.json (CLI до ragglet-37, старые прогоны) — импортированная
запись: она выводится из артефактов при каждом скане и никуда не пишется.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import anyio
from pydantic import TypeAdapter

from ragglet.bench.artifacts import SNAPSHOT_FILE
from ragglet.datasets.sources import describe_dataset
from ragglet.jobs.protocol import JobManager, LogChunk, TaskFailure, TaskKind, TaskRef, TaskState
from ragglet.registry import per_query
from ragglet.registry.listing import run_metrics, select_builds, select_runs
from ragglet.registry.protocol import (
    Artifact,
    BuildQuery,
    BuildRecord,
    NewBuild,
    NewRun,
    Page,
    QueryFilter,
    QueryPage,
    RegistryError,
    RunMetrics,
    RunQuery,
    RunRecord,
    RunSummary,
    TaskChange,
)

log = logging.getLogger(__name__)

RUN_FILE = "run.json"
BUILD_FILE = "build.json"
FORMAT = 1  # версия файла записи
DONE_MARKER = "summary.aggregate.csv"  # ядро пишет его последним: прогон завершён
AGGREGATE_FILE = "summary.aggregate.json"
TASK_LOG = "task.log"  # вывод воркера: менеджер задач (ragglet-29) пишет его в каталог задачи
SERVICE_FILES = frozenset({"task.json", "progress.json", "result.json"})  # служебные файлы менеджера
RUN_DIR_NAME = re.compile(r"^(?P<name>.+)_(?P<at>\d{8}T\d{6}Z)(?:-\d+)?$")
ACTIVE = (TaskState.QUEUED, TaskState.RUNNING)

_RUNS = TypeAdapter(RunRecord)
_BUILDS = TypeAdapter(BuildRecord)
_RUN_DERIVED = {"run_dir", "origin", "metrics"}  # в run.json не пишутся


def clean_json(value: Any) -> Any:
    """NaN и ±inf -> None: JSONResponse их не сериализует."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def read_json(path: Path) -> Any:
    return clean_json(json.loads(path.read_text(encoding="utf-8")))


def write_json_atomic(path: Path, data: Any) -> None:
    """Временный файл в том же каталоге и os.replace: читатель видит старый или новый файл целиком."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_run(record: RunRecord) -> None:
    data = _RUNS.dump_python(record, mode="json", exclude=_RUN_DERIVED)
    write_json_atomic(record.run_dir / RUN_FILE, {"format": FORMAT, **data})


def write_build(record: BuildRecord) -> None:
    data = _BUILDS.dump_python(record, mode="json", exclude={"build_dir"})
    write_json_atomic(record.build_dir / BUILD_FILE, {"format": FORMAT, **data})


def _read_record[T](path: Path, adapter: TypeAdapter[T], **derived: Any) -> T:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.pop("format", None) != FORMAT:
        raise ValueError(f"{path}: unknown record format")
    return adapter.validate_python({**data, **derived})


def load_metrics(run_dir: Path) -> RunMetrics | None:
    path = run_dir / AGGREGATE_FILE
    if not path.is_file():
        return None
    try:
        aggregate = read_json(path)
    except (OSError, ValueError) as e:
        log.warning("cannot read %s: %s", path, e)
        return None
    return run_metrics(aggregate) if isinstance(aggregate, dict) else None


def import_run(run_dir: Path) -> RunRecord | None:
    """
    Запись из каталога без run.json. Состояние — по summary.aggregate.csv; точных времён в
    артефактах нет, поэтому started_at и finished_at пусты, а enqueued_at — из имени каталога.
    """
    if not (run_dir / SNAPSHOT_FILE).is_file():
        return None  # не прогон
    match = RUN_DIR_NAME.match(run_dir.name)
    if match is None:
        log.warning("not importing %s: name is not <name>_<YYYYMMDDTHHMMSSZ>[-N]", run_dir)
        return None
    try:
        snapshot = read_json(run_dir / SNAPSHOT_FILE)
        manifest = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").is_file() else None
    except (OSError, ValueError) as e:
        log.warning("not importing %s: %s", run_dir, e)
        return None
    if not isinstance(snapshot, dict):
        return None

    # snapshot — dict без валидации: старые поколения текущую схему не проходят
    dataset = snapshot.get("dataset") or {}
    revision = ((manifest or {}).get("dataset") or {}).get("revision") or dataset.get("revision")
    storage = snapshot.get("storage") or {}
    collection = (storage.get("backend") or {}).get("collection") or storage.get("collection") or ""
    done = (run_dir / DONE_MARKER).is_file()
    return RunRecord(
        id=run_dir.name,
        run_dir=run_dir,
        name=str(snapshot.get("name") or match["name"]),
        scenario_id=None,  # со сценарием библиотеки по имени не связываем: ссылка могла бы быть ложной
        scenario=snapshot,
        etag=None,
        tags=[],
        dataset=describe_dataset(str(dataset.get("source") or "local"), str(dataset.get("id", "")), revision),
        collection=str(collection),
        plan=None,
        warnings=[],
        state=TaskState.COMPLETED if done else TaskState.FAILED,
        enqueued_at=datetime.strptime(match["at"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC),
        failure=None if done else TaskFailure(
            type="RunIncomplete",
            message=f"no {DONE_MARKER}: the run crashed, was killed or is still running from the CLI",
        ),
        origin="import",
        metrics=load_metrics(run_dir) if done else None,
    )


def load_run_dir(run_dir: Path) -> RunRecord | None:
    """Запись из run.json, а если его нет или он не читается — импорт по артефактам."""
    path = run_dir / RUN_FILE
    if path.is_file():
        try:
            record = _read_record(path, _RUNS, run_dir=run_dir)
        except (OSError, ValueError) as e:  # pydantic.ValidationError — тоже ValueError
            log.warning("%s is unreadable, importing the directory instead: %s", path, e)
        else:
            if record.state is TaskState.COMPLETED:
                record = replace(record, metrics=load_metrics(run_dir))
            return record
    return import_run(run_dir)


def subdirs(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {e.name for e in os.scandir(root) if e.is_dir() and not e.name.startswith(".")}


def reserve_dir(root: Path, base: str, taken: Callable[[str], bool]) -> Path:
    """base, а если занят — base-2, base-3: mkdir без exist_ok атомарен и против чужих процессов."""
    root.mkdir(parents=True, exist_ok=True)
    candidate, n = base, 2
    while True:
        if not taken(candidate):
            try:
                (root / candidate).mkdir()
                return root / candidate
            except FileExistsError:
                pass
        candidate, n = f"{base}-{n}", n + 1


def list_artifacts(run_dir: Path) -> list[Artifact]:
    """Файлы каталога как есть, кроме временных и служебных файлов менеджера задач."""
    if not run_dir.is_dir():
        return []
    out = []
    for entry in os.scandir(run_dir):
        if not entry.is_file() or entry.name.startswith(".") or entry.name in SERVICE_FILES:
            continue
        stat = entry.stat()
        out.append(Artifact(name=entry.name, size=stat.st_size,
                            modified_at=datetime.fromtimestamp(stat.st_mtime, UTC)))
    return sorted(out, key=lambda a: a.name)


def zip_artifacts(run_dir: Path, names: list[str]) -> IO[bytes]:
    """Архив `<id>/<файл>` во временном файле; закрыть его — дело вызывающего."""
    out = tempfile.TemporaryFile()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.write(run_dir / name, f"{run_dir.name}/{name}")
    out.seek(0)
    return out


def _utf8_complete(data: bytes) -> int:
    """Длина data без последнего символа UTF-8, если он обрезан: его отдаст следующий кусок."""
    for back in range(1, min(4, len(data)) + 1):
        byte = data[-back]
        if byte & 0b1100_0000 == 0b1000_0000:
            continue  # байт продолжения: ищем начало символа
        size = 1 if byte < 0x80 else 2 if byte >> 5 == 0b110 else 3 if byte >> 4 == 0b1110 else 4
        return len(data) if size <= back else len(data) - back
    return len(data)


def read_task_log(path: Path, offset: int, limit: int) -> LogChunk:
    """
    Лог задачи из каталога, когда менеджер её не знает (после рестарта): те же правила, что у
    менеджера — смещение в байтах, кусок не рвёт символ UTF-8, битые байты — U+FFFD.
    """
    if not path.is_file():
        return LogChunk(text="", next_offset=offset, complete=True)
    with path.open("rb") as f:
        size = os.fstat(f.fileno()).st_size
        f.seek(offset)
        data = f.read(limit)
    cut = _utf8_complete(data) or len(data)  # кусок из одного обломка символа отдаём как есть: иначе застрянем
    next_offset = offset + cut
    return LogChunk(text=data[:cut].decode("utf-8", errors="replace"), next_offset=next_offset,
                    complete=next_offset >= size)


class FileRegistry:
    """Реестр прогонов и сборок на файлах (ragglet-30). Пишет только процесс сервера."""

    def __init__(
            self,
            artifacts_dir: Path,
            builds_dir: Path,
            now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.artifacts_dir = artifacts_dir
        self.builds_dir = builds_dir
        self._now = now
        self._runs: dict[str, RunRecord] = {}
        self._builds: dict[str, BuildRecord] = {}
        self._loaded = False
        self._lock = anyio.Lock()  # записи и применение скана; update приходит конкурентно

    # --- индекс

    def _scan_runs(self, known: dict[str, RunRecord]) -> tuple[dict[str, RunRecord | None], set[str]]:
        """Новые каталоги, перепроверенные неполные импорты и исчезнувшие каталоги."""
        names = subdirs(self.artifacts_dir)
        changed: dict[str, RunRecord | None] = {}
        for name in names:
            record = known.get(name)
            stale = (record is not None and record.origin == "import" and record.state is TaskState.FAILED
                     and (record.run_dir / DONE_MARKER).is_file())  # CLI-прогон с тех пор закончился
            if record is None or stale:
                changed[name] = load_run_dir(self.artifacts_dir / name)
        return changed, set(known) - names

    def _scan_builds(self) -> dict[str, BuildRecord]:
        builds = {}
        for name in subdirs(self.builds_dir):
            path = self.builds_dir / name / BUILD_FILE
            if not path.is_file():
                continue
            try:
                builds[name] = _read_record(path, _BUILDS, build_dir=path.parent)
            except (OSError, ValueError) as e:
                log.warning("skipping unreadable %s: %s", path, e)
        return builds

    def _apply(self, changed: dict[str, RunRecord | None], removed: set[str]) -> None:
        for name in removed:
            self._runs.pop(name, None)  # каталог удалили руками
        for name, record in changed.items():
            if record is None:
                self._runs.pop(name, None)
            else:
                self._runs[name] = record

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            changed, removed = await anyio.to_thread.run_sync(self._scan_runs, {})
            self._apply(changed, removed)
            self._builds = await anyio.to_thread.run_sync(self._scan_builds)
            self._loaded = True

    async def _refresh(self) -> None:
        await self._ensure_loaded()
        async with self._lock:
            self._apply(*await anyio.to_thread.run_sync(self._scan_runs, dict(self._runs)))

    # --- запись

    async def create_run(self, new: NewRun) -> RunRecord:
        await self._ensure_loaded()
        async with self._lock:
            now = self._now()
            # каталог резервирует реестр: в нём живёт запись queued, ядро потом кладёт рядом артефакты
            run_dir = reserve_dir(self.artifacts_dir, f"{new.name}_{now:%Y%m%dT%H%M%SZ}", self._runs.__contains__)
            record = RunRecord(
                id=run_dir.name, run_dir=run_dir, state=TaskState.QUEUED, enqueued_at=now,
                **{f.name: getattr(new, f.name) for f in fields(NewRun)},
            )
            await anyio.to_thread.run_sync(write_run, record)
            self._runs[record.id] = record
            return record

    async def create_build(self, new: NewBuild) -> BuildRecord:
        await self._ensure_loaded()
        async with self._lock:
            now = self._now()
            build_dir = reserve_dir(self.builds_dir, f"{new.name}_{now:%Y%m%dT%H%M%SZ}", self._builds.__contains__)
            record = BuildRecord(
                id=build_dir.name, build_dir=build_dir, state=TaskState.QUEUED, enqueued_at=now,
                **{f.name: getattr(new, f.name) for f in fields(NewBuild)},
            )
            await anyio.to_thread.run_sync(write_build, record)
            self._builds[record.id] = record
            return record

    async def update(self, ref: TaskRef, change: TaskChange) -> None:
        await self._ensure_loaded()
        changes = {f.name: getattr(change, f.name) for f in fields(TaskChange) if getattr(change, f.name) is not None}
        async with self._lock:
            if ref.kind is TaskKind.RUN:
                run = self._runs.get(ref.id)
                if run is None:
                    log.warning("update of unknown run %s (directory removed?): %s", ref.id, change)
                    return
                changes.pop("result", None)
                run = replace(run, **changes)
                if run.state is TaskState.COMPLETED and run.metrics is None:
                    run = replace(run, metrics=await anyio.to_thread.run_sync(load_metrics, run.run_dir))
                await anyio.to_thread.run_sync(write_run, run)
                self._runs[ref.id] = run
            else:
                build = self._builds.get(ref.id)
                if build is None:
                    log.warning("update of unknown build %s: %s", ref.id, change)
                    return
                build = replace(build, **changes)
                await anyio.to_thread.run_sync(write_build, build)
                self._builds[ref.id] = build

    async def recover(self, jobs: JobManager) -> None:
        """
        Аварийный путь после падения сервера: штатную остановку помечает сам менеджер. Задача,
        которую менеджер не знает, уже не выполняется; если прогон успел записать summary, он
        завершён, иначе — сбой.
        """
        await self._ensure_loaded()
        now = self._now()
        active = [(TaskKind.RUN, r.id, r.state, r.run_dir) for r in self._runs.values() if r.state in ACTIVE]
        active += [(TaskKind.BUILD, b.id, b.state, None) for b in self._builds.values() if b.state in ACTIVE]
        for kind, task_id, state, run_dir in active:
            ref = TaskRef(kind, task_id)
            if await jobs.status(ref) is not None:
                continue
            if run_dir is not None and (run_dir / DONE_MARKER).is_file():
                change = TaskChange(state=TaskState.COMPLETED, finished_at=now)
            else:
                change = TaskChange(state=TaskState.FAILED, finished_at=now, failure=_restart_failure(kind, state))
            log.warning("recovering %s %s after a server restart: %s", kind, task_id, change.state)
            await self.update(ref, change)

    # --- чтение

    async def get_run(self, run_id: str) -> RunRecord | None:
        await self._ensure_loaded()
        record = self._runs.get(run_id)
        if record is not None:
            return record
        async with self._lock:
            # каталог мог появиться после скана (CLI); путь — только из листинга, не из id запроса
            if run_id not in await anyio.to_thread.run_sync(subdirs, self.artifacts_dir):
                return None
            record = await anyio.to_thread.run_sync(load_run_dir, self.artifacts_dir / run_id)
            if record is not None:
                self._runs[run_id] = record
            return record

    async def get_build(self, build_id: str) -> BuildRecord | None:
        await self._ensure_loaded()
        return self._builds.get(build_id)

    async def list_runs(self, query: RunQuery) -> Page[RunRecord]:
        await self._refresh()
        return select_runs(list(self._runs.values()), query)

    async def list_builds(self, query: BuildQuery) -> Page[BuildRecord]:
        await self._ensure_loaded()
        return select_builds(list(self._builds.values()), query)

    async def summary(self, run_id: str) -> RunSummary | None:
        record = await self.get_run(run_id)
        if record is None:
            return None
        return await anyio.to_thread.run_sync(_read_summary, record.run_dir)

    async def manifest(self, run_id: str) -> dict[str, Any] | None:
        record = await self.get_run(run_id)
        path = record.run_dir / "manifest.json" if record else None
        if path is None or not path.is_file():
            return None
        manifest = await anyio.to_thread.run_sync(read_json, path)
        return manifest if isinstance(manifest, dict) else None

    async def artifacts(self, run_id: str) -> list[Artifact]:
        record = await self.get_run(run_id)
        return await anyio.to_thread.run_sync(list_artifacts, record.run_dir) if record else []

    async def _run_dir(self, run_id: str) -> Path:
        record = await self.get_run(run_id)
        if record is None:
            raise RegistryError(404, "run.not_found", f"run {run_id!r} not found")
        return record.run_dir

    async def queries(self, run_id: str, query: QueryFilter) -> QueryPage:
        return await anyio.to_thread.run_sync(per_query.query_page, await self._run_dir(run_id), query)

    async def query_row(self, run_id: str, repeat: int, qid: str) -> dict[str, Any] | None:
        return await anyio.to_thread.run_sync(per_query.query_row, await self._run_dir(run_id), repeat, qid)


def _restart_failure(kind: TaskKind, state: TaskState) -> TaskFailure:
    if state is TaskState.QUEUED:
        message = f"the server restarted before the {kind} started; repeat it to run again"
    elif kind is TaskKind.RUN:
        message = "the server restarted while the run was running; artifacts of finished repeats are kept"
    else:
        message = "the server restarted while the build was running; the collection may be left without metadata: rebuild it"
    return TaskFailure(type="ServerRestart", message=message)


def _read_summary(run_dir: Path) -> RunSummary | None:
    path = run_dir / "summary.json"
    if not path.is_file():
        return None  # прогон в очереди, идёт или упал до конца
    repeats = read_json(path)
    aggregate_path = run_dir / AGGREGATE_FILE
    aggregate = read_json(aggregate_path) if aggregate_path.is_file() else {}
    return RunSummary(repeats=repeats if isinstance(repeats, list) else [],
                      aggregate=aggregate if isinstance(aggregate, dict) else {})
