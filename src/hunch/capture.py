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

from . import _probe
from .gitinfo import git, git_text, strip_credentials, toplevel
from .store import FORMAT_VERSION, Store, timestamp, write_json

UNTRACKED_FILE_LIMIT = 100 * 1024
UNTRACKED_TOTAL_LIMIT = 1024 * 1024
UNTRACKED_LIST_LIMIT = 1000
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
    code = capture_git(cwd, run_dir)
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


def capture_git(start: Path, run_dir: Path) -> dict | None:
    """Records the commit, the uncommitted diff of tracked files, and copies of
    small untracked text files. Never touches the repository's index or files."""
    top = toplevel(start)
    if top is None:
        return None
    commit = git_text(["rev-parse", "--verify", "-q", "HEAD"], top)
    remote = git_text(["config", "--get", "remote.origin.url"], top)
    # Before the first commit, diff the working tree against the empty tree.
    base = commit or git_text(["hash-object", "-t", "tree", "--stdin"], top, input=b"")
    # The explicit options override diff settings in the user's git config
    # (noprefix, submodule, external tools) that would make the patch unappliable.
    proc = git(
        ["diff", base, "--binary", "--no-color", "--no-ext-diff", "--no-textconv",
         "--src-prefix=a/", "--dst-prefix=b/", "--submodule=short"],
        top,
    ) if base else None
    patch = proc.stdout if proc is not None and proc.returncode == 0 else b""
    if patch:
        (run_dir / "code.patch").write_bytes(patch)

    proc = git(["ls-files", "--others", "--exclude-standard", "-z"], top)
    names = [n for n in proc.stdout.decode("utf-8", "replace").split("\0") if n] if proc and proc.returncode == 0 else []
    included, skipped, total = [], [], 0
    for name in names:
        source = top / name
        try:
            size = source.lstat().st_size
        except OSError:
            continue
        # A symlink's target may be outside the repository, such as a credentials file.
        if source.is_symlink() or size > UNTRACKED_FILE_LIMIT or total + size > UNTRACKED_TOTAL_LIMIT or not _is_text(source):
            skipped.append(name)
            continue
        target = run_dir / "untracked" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        total += size
        included.append(name)

    return {
        "root": str(top),
        "commit": commit,
        "branch": git_text(["symbolic-ref", "--short", "-q", "HEAD"], top),
        "remote": strip_credentials(remote) if remote else None,
        "dirty": bool(patch or names),
        "patch_bytes": len(patch),
        "untracked": included,
        "untracked_skipped": skipped[:UNTRACKED_LIST_LIMIT],
        "untracked_skipped_count": len(skipped),
    }


def _is_text(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" not in f.read(8192)
    except OSError:
        return False


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
