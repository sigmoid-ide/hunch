import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hunch import cli
from hunch.store import Store, append_jsonl, summarize_metrics, write_json

TRAIN = """\
import argparse, sys, hunch
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--fail", action="store_true")
parser.add_argument("--crash", action="store_true")
args = parser.parse_args()
hunch.init(config=args)
for step in range(5):
    hunch.log({"loss": 1.0 / (step + 1), "val_loss": 2.0 / (step + 1)})
print("done training")
if args.crash:
    raise RuntimeError("boom")
if args.fail:
    sys.exit(3)
"""

BASE = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated_git(tmp_path_factory, monkeypatch):
    """Keeps the user's git config (signing, hooks, autocrlf) out of the tests."""
    config = tmp_path_factory.mktemp("gitconfig") / "config"
    config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "test@example.com")
    monkeypatch.delenv("HUNCH_RUN_DIR", raising=False)
    monkeypatch.delenv("RANK", raising=False)


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repository with a committed train.py, as the working directory."""
    path = tmp_path / "project"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    (path / "train.py").write_text(TRAIN)
    git(path, "add", "train.py")
    git(path, "commit", "-q", "-m", "init")
    monkeypatch.chdir(path)
    return path


def hunch(*args) -> int:
    return cli.main(list(args))


def wrap(script: str, *options) -> int:
    """`hunch run [options] -- python <script>` with the test's own Python."""
    return hunch("run", *options, "--", sys.executable, *script.split())


def make_run(store: Store, short: str, *, start: int, duration: float = 60, line="main", parent=None,
             status="finished", why=None, metrics=(), notes=()) -> Path:
    """Writes a run folder directly; `start` is in minutes after BASE."""
    started = BASE + timedelta(minutes=start)
    run_id = f"{started:%Y%m%d-%H%M}-{short}"
    folder = store.runs_dir / run_id
    folder.mkdir(parents=True)
    write_json(folder / "run.json", {
        "format": 1, "id": run_id, "started_at": started.isoformat(), "source": "test",
        "command": ["python", "train.py"], "cwd": ".", "why": why, "parent": parent,
        "parent_source": None, "line": line, "main_metric": None, "labels": {}, "git": None,
        "host": {"hostname": "test"},
    })
    if status is not None:
        last, keys, sparks = summarize_metrics(list(metrics))
        write_json(folder / "end.json", {
            "status": status, "exit_code": 0, "duration": duration,
            "ended_at": (started + timedelta(seconds=duration)).isoformat(),
            "metrics": last, "metric_keys": keys, "sparks": sparks,
        })
    for kind, value in notes:
        append_jsonl(folder / "notes.jsonl", {"at": started.isoformat(), "kind": kind, "value": value})
    return folder


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An empty store outside git, as the working directory."""
    root = tmp_path / "runs-only"
    root.mkdir()
    monkeypatch.chdir(root)
    result = Store(root)
    result.ensure()
    return result
