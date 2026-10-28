"""
Клиент стенда в корне репозитория (ragglet-37) против настоящих роутеров: `create_app` с
заглушками менеджера задач и реестра под uvicorn в потоке, клиент ходит по HTTP своим urllib.
`sleep` клиента подменяется сценарием, который на каждом тике двигает заглушку менеджера.
"""
from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import json
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import anyio
import pytest
import uvicorn
import yaml
from api_support import DEMO_SHA, MODEL_SHA, Stand, make_stand, profile_yaml, scenario_yaml

from ragglet.core.issues import error, render_issues, warning
from ragglet.jobs.protocol import TaskFailure, TaskKind, TaskRef, TaskState
from ragglet.registry.protocol import RunQuery

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "ragglet"


def _load_client():
    loader = importlib.machinery.SourceFileLoader("ragglet_client", str(CLIENT))  # без .py: загрузчик явно
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("ragglet_client", loader))
    loader.exec_module(module)
    return module


cli = _load_client()


@dataclass
class Live:
    stand: Stand
    url: str
    capsys: pytest.CaptureFixture[str]

    def __call__(self, *argv: str) -> tuple[int, str, str]:
        code = cli.main(["--api", self.url, *argv])
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err

    def ref(self) -> TaskRef:
        """Единственная задача заглушки — та, которую поставил клиент."""
        (ref,) = self.stand.jobs._tasks
        return ref

    def run_dir(self) -> Path:
        return self.stand.settings.artifacts_dir / self.ref().id


@pytest.fixture
def live(tmp_path, monkeypatch, capsys) -> Iterator[Live]:
    stand = make_stand(tmp_path, monkeypatch, file_registry=True)
    server = uvicorn.Server(uvicorn.Config(stand.client.app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    monkeypatch.setattr(cli, "die_by_sigint", lambda: 130)  # выход сигналом убил бы pytest
    yield Live(stand, f"http://127.0.0.1:{port}", capsys)
    server.should_exit = True
    thread.join(10)


def ticks(monkeypatch, *actions: Callable[[], object]) -> list[Callable[[], object]]:
    """Подменяет sleep клиента: каждый тик выполняет следующее действие над заглушкой."""
    pending = list(actions)

    def fake_sleep(_seconds: float) -> None:
        assert pending, "the client kept waiting after the scenario ended"
        pending.pop(0)()

    monkeypatch.setattr(cli, "sleep", fake_sleep)
    return pending


def _file(tmp_path: Path, text: str, name: str = "scenario.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _finish(live: Live, state: TaskState, failure: TaskFailure | None = None, result: dict | None = None):
    return lambda: anyio.run(live.stand.jobs.finish, live.ref(), state, failure, result)


def _write_summary(live: Live) -> None:
    run_dir = live.run_dir()
    aggregate = {"scenario": "demo_run", "repeats": 2, "doc_ndcg@10_mean_mean_over_repeats": 0.315,
                 "lat_total_p50_mean_over_repeats": 152.5, "doc_ndcg@10_mean_std_over_repeats": 0.01}
    (run_dir / "summary.json").write_text(json.dumps([{"repeat_idx": 0}, {"repeat_idx": 1}]), encoding="utf-8")
    (run_dir / "summary.aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
    (run_dir / "summary.aggregate.csv").write_text("scenario,repeats\ndemo_run,2\n", encoding="utf-8")


# -----------------------
# Клиент — один файл на stdlib, Python 3.9
# -----------------------


def test_client_is_stdlib_only_and_parses_as_python39():
    tree = ast.parse(CLIENT.read_text(encoding="utf-8"), feature_version=(3, 9))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert imported - sys.stdlib_module_names == set()
    assert any(isinstance(n, ast.ImportFrom) and n.module == "__future__" for n in tree.body)


def test_render_issues_matches_the_core():
    issues = [error("index.no_metadata", "target default (chunks_v1)", "no metadata"),
              warning("index.not_green", "target default", "yellow"),
              error("config.value_error", "metrics.quality.k", "k too deep", path="metrics.quality.k")]
    as_json = [{"code": i.code, "severity": str(i.severity), "where": i.where, "path": i.path,
                "message": i.message} for i in issues]
    assert cli.render_issues("Index is not compatible", as_json, "rebuild") == render_issues(
        "Index is not compatible", issues, "rebuild")


def test_no_arguments_prints_help_with_groups(capsys):
    assert cli.main([]) == 0
    printed = capsys.readouterr().out

    assert "stand:\n  start " in printed and "results:\n  runs " in printed
    assert "scenarios  the scenario library on the server: list, show, edit, new, check, delete" in printed
    assert "start the stand" in printed and printed.count("start the stand") == 1  # без плоского списка argparse
    assert "\033[" not in printed  # не терминал — без цвета


def test_help_colors_keep_the_columns(monkeypatch):
    monkeypatch.setattr(cli, "color", lambda text, code, stream=None: f"<{code}>{text}</>")
    listing = cli.build_parser().description

    assert "<1>stand:</>" in listing
    assert "  <36>run</>        run a scenario" in listing  # ширина колонки — по имени без кода цвета


def test_every_command_is_in_exactly_one_help_group():
    commands = next(a for a in cli.build_parser()._actions if isinstance(a, cli.argparse._SubParsersAction))
    grouped = [name for names in cli.GROUPS.values() for name in names]
    assert sorted(grouped) == sorted(commands.choices)


# -----------------------
# run и build: dry-run, 422
# -----------------------


def test_dry_run_prints_preflight_report(live, tmp_path):
    code, out, _ = live("run", str(_file(tmp_path, scenario_yaml())), "--dry-run")

    assert code == 0, out
    assert "Preflight: OK" in out
    assert f"corpus sha256: {DEMO_SHA}" in out
    assert f"embedding: all-MiniLM-L6-v2@{MODEL_SHA}" in out
    assert "plan: 100 queries x 2 repeats, load closed" in out
    assert "target default: chunks_v1 at http://q:6333, green, 53 points" in out
    assert not live.stand.settings.artifacts_dir.exists() or not any(live.stand.settings.artifacts_dir.iterdir())


def test_dry_run_json_is_the_raw_response(live, tmp_path):
    code, out, _ = live("run", str(_file(tmp_path, scenario_yaml())), "--dry-run", "--json")

    assert code == 0
    assert json.loads(out)["dataset"]["corpus_sha256"] == DEMO_SHA


@pytest.mark.parametrize("change, code_name", [
    ({"retrieval": {"strategy": "cascade", "sources": [{"name": "v"}]}}, "retrieval.not_implemented"),
    ({"bogus": 1}, "config.extra_forbidden"),
])
def test_problems_exit_with_code_2(live, tmp_path, change, code_name):
    data = yaml.safe_load(scenario_yaml())
    data.update(change)
    code, out, err = live("run", str(_file(tmp_path, yaml.safe_dump(data))), "--dry-run")

    assert code == 2
    assert code_name in err and out == ""


def test_index_without_metadata_is_a_problem(live, tmp_path):
    live.stand.qdrant.collections.clear()
    code, _, err = live("run", str(_file(tmp_path, scenario_yaml())), "--dry-run")

    assert code == 2
    assert "index." in err


def test_broken_yaml_points_to_the_line(live, tmp_path):
    code, _, err = live("run", str(_file(tmp_path, "name: x\nstorage:\n  - [\n")), "--dry-run")

    assert code == 2
    assert "config.yaml" in err and "line " in err


def test_build_dry_run_lists_collections_to_recreate(live, tmp_path):
    code, out, _ = live("build", str(_file(tmp_path, profile_yaml(), "profile.yaml")), "--dry-run")

    assert code == 0, out
    assert "will recreate: chunks_v1 at http://q:6333 (target default); exists, 53 points" in out


def test_library_id_sends_the_reference(live, tmp_path):
    (live.stand.settings.scenarios_dir / "demo_run.yaml").write_text(scenario_yaml(), encoding="utf-8")
    code, out, err = live("run", "demo_run", "--no-wait")

    assert code == 0, err
    assert anyio.run(live.stand.registry.get_run, out.strip()).scenario_id == "demo_run"
    live("run", str(_file(tmp_path, scenario_yaml())), "--no-wait")  # из файла — без ссылки
    refs = {r.scenario_id for r in anyio.run(live.stand.registry.list_runs, RunQuery()).items}
    assert refs == {"demo_run", None}


def test_unknown_file_or_id(live):
    code, _, err = live("run", "no_such_scenario")

    assert code == 1
    assert "no such file and no such scenario" in err


# -----------------------
# Ожидание задачи
# -----------------------


def test_run_waits_through_phases_and_prints_summary(live, tmp_path, monkeypatch):
    jobs = live.stand.jobs
    ticks(monkeypatch,
          lambda: anyio.run(jobs.start, live.ref()),
          lambda: jobs.report(live.ref(), "queries", 3, 10),
          lambda: jobs.report(live.ref(), "queries", 10, 10),
          lambda: (_write_summary(live), _finish(live, TaskState.COMPLETED)()))

    code, out, err = live("run", str(_file(tmp_path, scenario_yaml())), "--tag", "night")

    assert code == 0, err
    run_id = live.ref().id
    lines = [line for line in err.splitlines() if line]
    assert lines == [f"Queued: {run_id} (position 1)", f"Started: {run_id}",
                     "Preparing (preflight, loading models)...", "Finishing (writing summary)..."]
    assert out.splitlines()[0] == "Done"
    assert "  doc_ndcg@10_mean  0.315" in out and "std" not in out
    assert out.splitlines()[-1] == f"Artifacts: {live.run_dir()} (on the server)"
    assert anyio.run(live.stand.registry.get_run, run_id).tags == ["night"]


def test_failed_run_prints_error_and_log_tail(live, tmp_path, monkeypatch):
    failure = TaskFailure(type="RuntimeError", message="qdrant went away", qid="PLAIN-7")
    ticks(monkeypatch,
          lambda: anyio.run(live.stand.jobs.start, live.ref()),
          lambda: (live.stand.jobs.append_log(live.ref(), "line 1\nTraceback: boom\n"),
                   _finish(live, TaskState.FAILED, failure)()))

    code, _, err = live("run", str(_file(tmp_path, scenario_yaml())))

    assert code == 1
    assert "Failed: RuntimeError: qdrant went away (qid=PLAIN-7)" in err
    assert "Traceback: boom" in err


def test_failed_preflight_at_start_is_code_2(live, tmp_path, monkeypatch):
    failure = TaskFailure(type="ValidationError", message="Index is not compatible", title="Index is not compatible",
                          issues=(error("index.corpus_mismatch", "target default", "rebuilt"),))
    ticks(monkeypatch, _finish(live, TaskState.FAILED, failure))

    code, _, err = live("run", str(_file(tmp_path, scenario_yaml())))

    assert code == 2
    assert "index.corpus_mismatch" in err


def test_cancelled_elsewhere_is_code_1(live, tmp_path, monkeypatch):
    ticks(monkeypatch, _finish(live, TaskState.CANCELLED))

    code, _, err = live("build", str(_file(tmp_path, profile_yaml(), "profile.yaml")))

    assert code == 1
    assert "Cancelled" in err


def test_ctrl_c_cancels_the_task(live, tmp_path, monkeypatch):
    def interrupt():
        raise KeyboardInterrupt

    ticks(monkeypatch,
          lambda: anyio.run(live.stand.jobs.start, live.ref()),
          interrupt,
          _finish(live, TaskState.CANCELLED))

    code, _, err = live("run", str(_file(tmp_path, scenario_yaml())))

    assert code == 130  # die_by_sigint подменён
    assert live.stand.jobs._tasks[live.ref()].cancel_requested
    assert "Cancelling" in err and "Cancelled" in err


def test_no_wait_prints_only_the_id(live, tmp_path):
    code, out, _ = live("run", str(_file(tmp_path, scenario_yaml())), "--no-wait")

    assert code == 0
    assert out.strip() == live.ref().id
    assert live.stand.jobs._tasks[live.ref()].status.state is TaskState.QUEUED


# -----------------------
# cancel, log, queue; адрес и недоступный сервер
# -----------------------


def test_cancel_log_and_queue(live, tmp_path, monkeypatch):
    first = live("run", str(_file(tmp_path, scenario_yaml())), "--no-wait")[1].strip()
    ref = TaskRef(TaskKind.RUN, first)
    anyio.run(live.stand.jobs.start, ref)
    live.stand.jobs.append_log(ref, "== run started ==\n")
    second = live("run", str(_file(tmp_path, scenario_yaml())), "--no-wait")[1].strip()

    code, out, _ = live("queue")
    assert code == 0
    assert first in out.splitlines()[1] and second in out.splitlines()[2]

    assert live("cancel", second)[0:2] == (0, "Cancelled\n")
    assert live("cancel", second)[0] == 1  # уже не активна

    ticks(monkeypatch, lambda: (live.stand.jobs.append_log(ref, "done\n"), anyio.run(
        live.stand.jobs.finish, ref, TaskState.COMPLETED)))
    code, out, _ = live("log", first, "--follow")
    assert (code, out) == (0, "== run started ==\ndone\n")


def test_unreachable_server_is_code_3(capsys):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # порт свободен после закрытия

    code = cli.main(["--api", f"http://127.0.0.1:{port}", "queue"])

    assert code == 3
    assert "start" in capsys.readouterr().err


def test_not_ragglet_is_code_3(monkeypatch, capsys):
    monkeypatch.setattr(cli, "transport", lambda *a: (404, b"<html>nginx</html>"))
    assert cli.main(["--api", "http://127.0.0.1:1", "queue"]) == 3
    assert "does not look like a ragglet server" in capsys.readouterr().err


def test_address_order(monkeypatch, tmp_path):
    state = tmp_path / "server.json"
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    monkeypatch.delenv("RAGGLET_API_URL", raising=False)
    parse = cli.build_parser().parse_args

    assert cli.api_url(parse(["queue"])) == "http://127.0.0.1:8000"
    state.write_text(json.dumps({"url": "http://127.0.0.1:8100"}), encoding="utf-8")
    assert cli.api_url(parse(["queue"])) == "http://127.0.0.1:8100"
    monkeypatch.setenv("RAGGLET_API_URL", "http://env:1")
    assert cli.api_url(parse(["queue"])) == "http://env:1"
    assert cli.api_url(parse(["queue", "--api", "http://flag:2"])) == "http://flag:2"


# -----------------------
# Стенд: start, stop, status — с крошечным stdlib-сервером вместо ragglet.server
# -----------------------

STUB = """
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

tasks_file = Path(sys.argv[2])
stand = {"qdrant": [], "redis": None, "compute": {}, "system": {}, "commit": None, "python": "3.x",
         "env": {}, "current": None, "queued": 0}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        bodies = {"/v1/stand": stand, "/v1/tasks": json.loads(tasks_file.read_text())}
        body = json.dumps(bodies.get(self.path.split("?")[0], {})).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


print("stub listening", flush=True)
HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def stub(tmp_path, monkeypatch):
    """start поднимает заглушку сервера; окружение и Docker не трогаются."""
    script = tmp_path / "stub_server.py"
    script.write_text(STUB, encoding="utf-8")
    tasks = tmp_path / "tasks.json"
    tasks.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(cli, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(cli, "prepare_env", lambda: sys.executable)
    monkeypatch.setattr(cli, "server_command", lambda python, host, port: [python, str(script), str(port), str(tasks)])
    monkeypatch.delenv("RAGGLET_API_URL", raising=False)
    yield tasks
    pid = cli.read_state().get("pid")
    if pid and cli.alive(pid):
        import os
        import signal

        os.kill(pid, signal.SIGKILL)


def test_start_status_stop(stub, monkeypatch, capsys):
    port = _free_port()

    assert cli.main(["start", "--port", str(port), "--no-services"]) == 0
    assert f"Ragglet is running: http://127.0.0.1:{port}" in capsys.readouterr().out
    pid = cli.read_state()["pid"]
    assert cli.alive(pid)

    assert cli.main(["start", "--port", str(port), "--no-services"]) == 0  # повторный — не падает
    assert "already running" in capsys.readouterr().out
    assert cli.main(["status"]) == 0  # адрес из server.json
    assert f"server: http://127.0.0.1:{port}, pid {pid}" in capsys.readouterr().out

    stub.write_text(json.dumps([{"kind": "run", "id": "demo_run_x", "state": "running"}]), encoding="utf-8")
    monkeypatch.setattr(cli, "confirm", lambda question: False)
    assert cli.main(["stop"]) == 1  # активная задача, не подтвердили
    assert cli.alive(pid)

    monkeypatch.setattr(cli, "confirm", lambda question: True)
    assert cli.main(["stop"]) == 0
    assert "Stopped the server" in capsys.readouterr().out
    assert not cli.alive(pid) and cli.read_state() == {}
    assert cli.main(["status"]) == 3


def test_start_reports_a_server_that_died(stub, monkeypatch, capsys, tmp_path):
    crash = tmp_path / "crash.py"
    crash.write_text("import sys; print('OSError: [Errno 48] Address already in use'); sys.exit(1)\n")
    monkeypatch.setattr(cli, "server_command", lambda python, host, port: [python, str(crash)])

    assert cli.main(["start", "--port", str(_free_port()), "--no-services"]) == 1
    assert "Address already in use" in capsys.readouterr().err
    assert cli.read_state() == {}


def test_start_without_docker_prints_the_compose_command(stub, monkeypatch, capsys):
    monkeypatch.setattr(cli, "port_open", lambda url, timeout=1.0: False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)

    assert cli.main(["start", "--port", str(_free_port())]) == 1
    assert "docker compose -f docker/docker-compose.yaml --profile qdrant up -d qdrant redis" in (
        capsys.readouterr().err)


# -----------------------
# Библиотека: scenarios, profiles
# -----------------------

COMMENTED = "# мой сценарий\n" + scenario_yaml()


def _editor(tmp_path: Path, monkeypatch, replace: tuple[str, str]) -> None:
    """$EDITOR — python-скрипт, который правит файл без участия человека."""
    script = tmp_path / "editor.py"
    script.write_text("import sys, pathlib\np = pathlib.Path(sys.argv[1])\n"
                      f"p.write_text(p.read_text().replace({replace[0]!r}, {replace[1]!r}))\n", encoding="utf-8")
    monkeypatch.setenv("EDITOR", f"{sys.executable} {script}")
    monkeypatch.delenv("VISUAL", raising=False)


def test_scenarios_list_show_and_edit_round_trip(live, tmp_path, monkeypatch):
    library = live.stand.settings.scenarios_dir / "demo_run.yaml"
    library.write_text(COMMENTED, encoding="utf-8")

    code, out, _ = live("scenarios", "list")
    assert code == 0 and "demo_run" in out and "baseline_qdrant" in out
    assert live("scenarios", "show", "demo_run")[1] == COMMENTED  # байт в байт, с комментариями

    _editor(tmp_path, monkeypatch, ("repeats: 2", "repeats: 3"))
    code, out, _ = live("scenarios", "edit", "demo_run")
    assert (code, out) == (0, "Saved demo_run\n")
    assert "repeats: 3" in library.read_text(encoding="utf-8") and "# мой сценарий" in library.read_text()


def test_edit_over_a_changed_file_is_409(live, tmp_path, monkeypatch):
    library = live.stand.settings.scenarios_dir / "demo_run.yaml"
    library.write_text(COMMENTED, encoding="utf-8")
    script = tmp_path / "editor.py"  # пока открыт редактор, файл правят в IDE
    script.write_text("import sys, pathlib\n"
                      f"pathlib.Path({str(library)!r}).write_text(pathlib.Path({str(library)!r}).read_text() + '#x\\n')\n"
                      "p = pathlib.Path(sys.argv[1]); p.write_text(p.read_text() + '# mine\\n')\n", encoding="utf-8")
    monkeypatch.setenv("EDITOR", f"{sys.executable} {script}")

    code, _, err = live("scenarios", "edit", "demo_run")

    assert code == 1
    assert "changed on the server" in err
    kept = Path(err.rsplit("kept in ", 1)[1].strip())
    assert kept.read_text(encoding="utf-8").endswith("# mine\n")


def test_new_from_scenario_and_from_run(live, monkeypatch):
    (live.stand.settings.scenarios_dir / "demo_run.yaml").write_text(COMMENTED, encoding="utf-8")

    assert live("scenarios", "new", "demo_copy", "--from", "demo_run")[0:2] == (0, "Created demo_copy\n")
    copy = (live.stand.settings.scenarios_dir / "demo_copy.yaml").read_text(encoding="utf-8")
    assert copy.startswith("# мой сценарий\n") and 'name: "demo_copy"' in copy

    run_id = live("run", "demo_run", "--no-wait")[1].strip()
    assert live("scenarios", "new", "from_run", "--from", run_id)[0] == 0
    assert (live.stand.settings.scenarios_dir / "from_run.yaml").is_file()


def test_check_and_delete(live, monkeypatch):
    (live.stand.settings.scenarios_dir / "demo_run.yaml").write_text(COMMENTED, encoding="utf-8")
    assert "Preflight: OK" in live("scenarios", "check", "demo_run")[1]

    monkeypatch.setattr(cli, "confirm", lambda question: False)
    assert live("scenarios", "delete", "demo_run")[0] == 1
    monkeypatch.setattr(cli, "confirm", lambda question: True)
    assert live("scenarios", "delete", "demo_run")[0:2] == (0, "Deleted demo_run\n")
    assert not (live.stand.settings.scenarios_dir / "demo_run.yaml").exists()


def test_profiles_list_and_check(live):
    (live.stand.settings.index_profiles_dir / "demo_build.yaml").write_text(profile_yaml(), encoding="utf-8")

    assert "demo_build" in live("profiles", "list")[1]
    assert "will recreate: chunks_v1" in live("profiles", "check", "demo_build")[1]


# -----------------------
# Результаты: runs, builds, indexes
# -----------------------


@pytest.fixture
def finished_run(live):
    from registry_support import RUN_ID, make_run_dir

    make_run_dir(live.stand.settings.artifacts_dir)
    return RUN_ID


def test_runs_list_and_show(live, finished_run):
    code, out, _ = live("runs", "list", "-k", "3")
    assert code == 0
    header, row = out.splitlines()[:2]
    assert "ndcg@3" in header and row.startswith(finished_run) and "completed" in row
    assert json.loads(live("runs", "list", "--json")[1])["items"][0]["id"] == finished_run

    code, out, _ = live("runs", "show", finished_run)
    assert code == 0
    assert f"artifacts: {live.stand.settings.artifacts_dir / finished_run} (on the server)" in out
    assert "summary:" in out and "doc_ndcg@3_mean" in out and "environment: commit eeeeeeeeeeee" in out


def test_runs_files_export_and_queries(live, finished_run, tmp_path):
    assert "summary.aggregate.json" in live("runs", "files", finished_run)[1]

    code, out, _ = live("runs", "export", finished_run, "-o", str(tmp_path / "x"))
    assert code == 0 and (Path(out.strip()) / "manifest.json").is_file()
    code, out, _ = live("runs", "export", finished_run, "--zip", "-o", str(tmp_path / "z"))
    assert code == 0 and Path(out.strip()).suffix == ".zip"

    code, out, _ = live("runs", "queries", finished_run, "-n", "2")
    assert code == 0
    assert out.splitlines()[0].split()[:2] == ["qid", "outcome"] and len(out.splitlines()) == 4


def test_builds_and_indexes(live, tmp_path, monkeypatch):
    ticks(monkeypatch, lambda: anyio.run(live.stand.jobs.start, live.ref()),
          _finish(live, TaskState.COMPLETED, result={"documents": 53}))
    assert live("build", str(_file(tmp_path, profile_yaml(), "profile.yaml")))[0] == 0

    build_id = live.ref().id
    assert build_id in live("builds", "list")[1]
    code, out, _ = live("builds", "show", build_id)
    assert code == 0 and "documents: 53" in out and "state: completed" in out

    code, out, _ = live("indexes")
    assert code == 0 and "chunks_v1" in out and "ready" in out
