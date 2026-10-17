from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    code: str
    severity: Severity
    where: str
    message: str


def error(code: str, where: str, message: str) -> Issue:
    return Issue(code, Severity.ERROR, where, message)


def warning(code: str, where: str, message: str) -> Issue:
    return Issue(code, Severity.WARNING, where, message)


def render_issues(title: str, issues: list[Issue], hint: str | None = None, limit: int = 20) -> str:
    n_errors = sum(1 for i in issues if i.severity is Severity.ERROR)
    n_warnings = len(issues) - n_errors
    counts = [f"{n_errors} error{'s' if n_errors != 1 else ''}"] if n_errors else []
    if n_warnings:
        counts.append(f"{n_warnings} warning{'s' if n_warnings != 1 else ''}")
    ordered = sorted(issues, key=lambda i: i.severity is not Severity.ERROR)
    code_w = max((len(i.code) for i in ordered[:limit]), default=0)
    where_w = max((len(i.where) for i in ordered[:limit]), default=0)
    lines = [f"{title} ({', '.join(counts)})"]
    lines += [f"  {i.code:<{code_w}}  {i.where:<{where_w}}  {i.message}" for i in ordered[:limit]]
    if len(ordered) > limit:
        lines.append(f"  ... and {len(ordered) - limit} more")
    if hint:
        lines.append(hint)
    return "\n".join(lines)


class ValidationError(Exception):
    def __init__(self, title: str, issues: list[Issue], hint: str | None = None):
        self.title = title
        self.issues = issues
        self.hint = hint
        super().__init__(render_issues(title, issues, hint))


def raise_on_errors(title: str, issues: Iterable[Issue], hint: str | None = None) -> list[Issue]:
    """Бросает ValidationError, если есть хотя бы одна ошибка; иначе возвращает предупреждения."""
    issues = list(issues)
    if any(i.severity is Severity.ERROR for i in issues):
        raise ValidationError(title, issues, hint)
    return issues
