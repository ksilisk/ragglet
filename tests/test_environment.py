from __future__ import annotations

import subprocess

from ragglet.core.environment import ENV_FACTS, capture_env, describe_services, git_commit

COMPUTE = {"device": "cpu", "accelerator": None, "torch_threads": 2}


def _git(cwd, *args) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def test_git_commit_is_head_of_enclosing_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")

    assert git_commit(tmp_path / "pkg") == _git(tmp_path, "rev-parse", "HEAD")


def test_git_commit_outside_repo_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_commit(tmp_path) is None


def test_capture_env_has_every_fact_and_compute():
    env = capture_env(COMPUTE)

    assert set(env) == {*ENV_FACTS, "compute"}
    assert env["compute"] == COMPUTE
    assert env["python"].count(".") == 2
    system = env["system"]
    assert system["platform"] and system["cpus"] >= 1 and system["ram_bytes"] > 0


class _Service:
    def __init__(self, kind):
        self.kind = kind

    async def describe(self):
        return {"type": self.kind, "version": "1.0"}


async def test_services_described_by_role_and_name():
    services = {"storage": {"shard_0": _Service("qdrant"), "shard_1": _Service("qdrant")}, "cache": {}}

    described = await describe_services(services)

    assert described == {
        "storage": {"shard_0": {"type": "qdrant", "version": "1.0"}, "shard_1": {"type": "qdrant", "version": "1.0"}},
        "cache": {},
    }
