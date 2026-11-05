"""What a run records when it starts: code state, environment and host."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from . import _probe, snapshot
from .gitinfo import git_text, strip_credentials, toplevel
from .store import FORMAT_VERSION, Store, timestamp, write_json

ENV_VARS = ("CUDA_VISIBLE_DEVICES", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV")
ENV_PREFIXES = ("SLURM_",)

_PYTHON_NAME = re.compile(r"^(python[\d.]*|pythonw|py)(\.exe)?$", re.IGNORECASE)


def start_run(
    store: Store,
    *,
    command: list[str],
    cwd: Path,
    why: str | None,
    parent: str | None,
    source: str,
    python: str | None,
    main_metric: str | None = None,
    labels: dict | None = None,
    config: dict | None = None,
) -> Path:
    """Creates a run folder and records everything known at start.

    `python` is the interpreter to inspect from outside, or None to inspect the
    current process. run.json is written last: until it exists, the folder is
    not a run.
    """
    store.ensure()
    run_dir = store.new_run_dir()
    write_json(run_dir / "files.json", snapshot.take(store.root, store.dir / "objects"))
    code = capture_git(store.root)
    line = code.get("branch") if code else None
    if parent:
        parent_source = "explicit"
    else:
        parent = store.default_parent(line)
        parent_source = "default" if parent else None
    write_json(run_dir / "config.json", {"args": command, "values": config or {}})
    write_json(run_dir / "env.json", capture_env(python))
    (run_dir / "heartbeat").touch()
    write_json(
        run_dir / "run.json",
        {
            "format": FORMAT_VERSION,
            "id": run_dir.name,
            "started_at": timestamp(),
            "source": source,
            "command": command,
            "cwd": str(cwd),
            "why": why or None,
            "parent": parent,
            "parent_source": parent_source,
            "line": line,
            "main_metric": main_metric,
            "labels": labels or {},
            "git": code,
            "host": capture_host(),
        },
    )
    return run_dir


def capture_git(start: Path) -> dict | None:
    """Where the code sits in git, when the project uses it. The code itself is in
    the run's snapshot, so git is optional. Only reads; never touches the repository."""
    top = toplevel(start)
    if top is None:
        return None
    remote = git_text(["config", "--get", "remote.origin.url"], top)
    status = git_text(["status", "--porcelain"], top)
    return {
        "root": str(top),
        "commit": git_text(["rev-parse", "--verify", "-q", "HEAD"], top),
        "branch": git_text(["symbolic-ref", "--short", "-q", "HEAD"], top),
        "remote": strip_credentials(remote) if remote else None,
        "dirty": bool(status),
    }


def capture_env(python: str | None) -> dict:
    if python is None:
        env = _probe.collect()
        env["source"] = "process"
    else:
        env = probe_interpreter(python)
        env["source"] = "wrapper"
    env["gpus"] = capture_gpus()
    env["env_vars"] = {
        key: value
        for key, value in sorted(os.environ.items())
        if key in ENV_VARS or key.startswith(ENV_PREFIXES)
    }
    return env


def probe_interpreter(python: str) -> dict:
    """Runs the probe in another interpreter, from an empty folder so files in
    the project cannot shadow standard modules."""
    source = Path(_probe.__file__).read_text(encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory() as empty:
            proc = subprocess.run([python, "-c", source], cwd=empty, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"executable": python, "error": f"could not inspect this Python: {e}"}
    if proc.returncode != 0:
        return {"executable": python, "error": proc.stderr.decode("utf-8", "replace").strip()[-500:]}
    try:
        return json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"executable": python, "error": "could not read the probe's output"}


def guess_python(executable: str) -> str:
    """The Python the wrapped command will most likely run. The training process
    corrects this guess when it calls hunch.init()."""
    path = Path(executable)
    if _PYTHON_NAME.match(path.name):
        return str(path)
    for name in ("python.exe", "python3", "python"):
        sibling = path.parent / name
        if sibling.is_file():
            return str(sibling)
    return shutil.which("python3") or shutil.which("python") or sys.executable


def capture_gpus() -> list[dict] | None:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    gpus = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
    return gpus


def capture_host() -> dict:
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
