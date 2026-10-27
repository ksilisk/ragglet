"""
`python -m ragglet.server` — REST API стенда (ragglet-28). Его запускает `start` клиента в корне
репозитория (в фоне, интерпретатором .venv) и compose (ragglet-34). Остальные настройки —
переменные RAGGLET_* или .env в рабочем каталоге.
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ragglet.server", description=__doc__.strip().splitlines()[0])
    parser.add_argument("--host", help="bind address (default 127.0.0.1, or RAGGLET_HOST)")
    parser.add_argument("--port", type=int, help="port (default 8000, or RAGGLET_PORT)")
    args = parser.parse_args(argv)

    import uvicorn

    from ragglet.api.app import create_app
    from ragglet.api.settings import ServerSettings
    from ragglet.jobs.process import ProcessJobManager
    from ragglet.registry.files import FileRegistry

    settings = ServerSettings(**{k: v for k, v in {"host": args.host, "port": args.port}.items() if v is not None})
    print(settings.describe(), flush=True)
    registry = FileRegistry(settings.artifacts_dir, settings.builds_dir)
    jobs = ProcessJobManager(registry, cancel_grace_s=settings.cancel_grace_s, builds_dir=settings.builds_dir)
    # один процесс: очередь задач и блокировки библиотеки живут в его памяти
    uvicorn.run(create_app(settings, jobs=jobs, registry=registry), host=settings.host, port=settings.port,
                workers=1)


if __name__ == "__main__":
    main()
