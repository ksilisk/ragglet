"""
Фиктивный воркер для тестов менеджера задач: настоящий `worker.main`, но вместо ядра —
сценарий из `config["script"]`. torch не импортируется, процесс поднимается за ~0,1 с.

Шаги: ["progress", stage, done, total] — вызовы 1..done; ["sleep", s]; ["async_sleep", s] —
через `worker.cancellable`, как прогон; ["print", text]; ["eprint", text]; ["raise", message,
note?]; ["invalid"] — ValidationError; ["exit", code] — выход без result.json; ["kill"] —
SIGKILL себе; ["ignore_sigterm"]; ["mark", path, label] — дописать "label time" в файл;
["return", value]. Без сценария — прогресс repeats × 3 и успех.
"""
from __future__ import annotations

import os
import signal
import sys
import time
from functools import partial
from typing import Any

import anyio

from ragglet.core.issues import ValidationError, error, warning
from ragglet.jobs import worker
from ragglet.jobs.protocol import TaskKind, TaskSpec


def default_script(spec: TaskSpec) -> list[list[Any]]:
    total = spec.config.get("run", {}).get("repeats", 1) * 3
    stage = "queries" if spec.ref.kind is TaskKind.RUN else "corpus"
    result = {"documents": total} if spec.ref.kind is TaskKind.BUILD else None
    return [["progress", stage, total, total], ["return", result]]


def play(spec: TaskSpec, progress: worker.ThrottledProgress) -> Any:
    for step, *args in spec.config.get("script") or default_script(spec):
        match step:
            case "progress":
                stage, done, total = args
                for i in range(1, done + 1):
                    progress(stage, i, total)
            case "sleep":
                time.sleep(args[0])
            case "async_sleep":
                anyio.run(worker.cancellable, partial(anyio.sleep, args[0]))
            case "print":
                print(args[0], flush=True)
            case "eprint":
                print(args[0], file=sys.stderr, flush=True)
            case "raise":
                e = RuntimeError(args[0])
                for note in args[1:]:
                    e.add_note(note)
                raise e
            case "invalid":
                raise ValidationError("Scenario config is invalid", [
                    error("config.value_error", "metrics.quality.k", "k=30 exceeds depth 10",
                          path="metrics.quality.k"),
                    warning("index.not_green", "target default (chunks_v1)", "collection status is yellow"),
                ], hint="lower metrics.quality.k")
            case "exit":
                os._exit(args[0])
            case "kill":
                os.kill(os.getpid(), signal.SIGKILL)
            case "ignore_sigterm":
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            case "mark":
                with open(args[0], "a", encoding="utf-8") as f:
                    f.write(f"{args[1]} {time.time()}\n")
            case "return":
                return args[0]
            case _:
                raise ValueError(f"unknown step {step!r}")
    return None


if __name__ == "__main__":
    sys.exit(worker.main(operations={TaskKind.RUN: play, TaskKind.BUILD: play}))
