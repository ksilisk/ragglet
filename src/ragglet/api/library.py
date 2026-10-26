from __future__ import annotations

import hashlib
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from ragglet.api.problems import (
    Problem,
    ProblemError,
    ProblemIssue,
    from_pydantic,
    load_mapping,
    problem_error,
)

ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")  # id — имя файла: name сценария, id профиля


def etag_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class LibraryEntry:
    id: str
    text: str
    updated_at: datetime
    etag: str
    config: Any | None  # провалидированный конфиг; None — черновик
    problem: Problem | None  # почему черновик


class ConfigLibrary:
    """
    Каталог YAML-файлов `<id>.yaml`, id = поле `id_field` внутри. Сохраняется всё, что
    разбирается как YAML-mapping с правильным id: невалидное по схеме — черновиком. Версия
    файла — etag содержимого: запись поверх чужой правки — 409.
    """

    def __init__(self, directory: Path, model: type[BaseModel], id_field: str, kind: str, title: str):
        self.directory = directory
        self.model = model
        self.id_field = id_field
        self.kind = kind  # "scenario" | "index profile": для сообщений
        self.title = title  # заголовок Problem для ошибок схемы
        self._lock = threading.Lock()

    def list(self) -> list[LibraryEntry]:
        if not self.directory.is_dir():
            return []
        return [self._read(path) for path in sorted(self.directory.glob("*.yaml"))]

    def find(self, entry_id: str) -> LibraryEntry | None:
        path = self._path(entry_id)
        return self._read(path) if path.is_file() else None

    def get(self, entry_id: str) -> LibraryEntry:
        entry = self.find(entry_id)
        if entry is None:
            raise problem_error(404, "library.not_found", f"No such {self.kind}", f"{self.kind} {entry_id!r} not found",
                                where=self._where(entry_id))
        return entry

    def create(self, text: str) -> LibraryEntry:
        entry_id = self._declared_id(load_mapping(text, self.title))
        if entry_id is None:
            raise problem_error(422, "library.invalid_id", self.title,
                                f"{self.id_field} must match [A-Za-z0-9_-]{{1,64}}: it is the file name",
                                path=self.id_field)
        entry, _ = self.write(entry_id, text, etag=None)
        return entry

    def write(self, entry_id: str, text: str, etag: str | None) -> tuple[LibraryEntry, bool]:
        """Пишет файл; (запись, создан ли). etag=None — файла быть не должно."""
        data = load_mapping(text, self.title)  # битый YAML не сохраняем
        if data.get(self.id_field) != entry_id:
            raise problem_error(
                422, "library.name_mismatch", self.title,
                f"{self.id_field} must be {entry_id!r}: it is the file name", path=self.id_field,
            )
        with self._lock:
            current = self.find(entry_id)
            if current is not None and current.etag != etag:
                hint = "exists" if etag is None else f"changed on disk: etag {current.etag}"
                raise problem_error(409, "library.conflict", f"{self.kind.capitalize()} was changed",
                                    f"{self.kind} {entry_id!r} {hint}, updated at {current.updated_at:%Y-%m-%dT%H:%M:%SZ}",
                                    where=self._where(entry_id))
            if current is None and etag is not None:
                raise problem_error(409, "library.conflict", f"{self.kind.capitalize()} was changed",
                                    f"{self.kind} {entry_id!r} was deleted", where=self._where(entry_id))
            self._atomic_write(self._path(entry_id), text)
        return self.get(entry_id), current is None

    def delete(self, entry_id: str, etag: str | None) -> None:
        with self._lock:
            current = self.get(entry_id)
            if etag is not None and current.etag != etag:
                raise problem_error(409, "library.conflict", f"{self.kind.capitalize()} was changed",
                                    f"{self.kind} {entry_id!r} changed on disk: etag {current.etag}",
                                    where=self._where(entry_id))
            self._path(entry_id).unlink()

    # ---

    def _path(self, entry_id: str) -> Path:
        return self.directory / f"{entry_id}.yaml"

    def _where(self, entry_id: str) -> str:
        return f"{self.directory.name}/{entry_id}.yaml"

    def _declared_id(self, data: dict[str, Any]) -> str | None:
        value = data.get(self.id_field)
        return value if isinstance(value, str) and ID_RE.fullmatch(value) else None

    def _read(self, path: Path) -> LibraryEntry:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        entry_id = path.stem
        config, problem = None, None
        try:
            data = load_mapping(text, self.title)
        except ProblemError as e:
            problem = e.problem
        else:
            issues: list[ProblemIssue] = []
            if data.get(self.id_field) != entry_id:  # файл переименовали в IDE
                issues.append(ProblemIssue(
                    code="library.name_mismatch", severity="error", path=self.id_field,
                    message=f"{self.id_field} must be {entry_id!r}: it is the file name",
                ))
            try:
                config = self.model.model_validate(data)
            except PydanticValidationError as e:
                issues += from_pydantic(e, self.title).issues
            if issues:
                config, problem = None, Problem(title=self.title, issues=issues)
        return LibraryEntry(
            id=entry_id,
            text=text,
            updated_at=datetime.fromtimestamp(path.stat().st_mtime, UTC),
            etag=etag_of(text),
            config=config,
            problem=problem,
        )

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(text.encode("utf-8"))
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
