"""Small wrappers around the git command line."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def git(args: list[str], cwd: Path, env: dict | None = None, input: bytes | None = None) -> subprocess.CompletedProcess | None:
    """Runs git and returns the finished process, or None when git is missing."""
    try:
        return subprocess.run(["git", *args], cwd=cwd, env=env, input=input, capture_output=True)
    except OSError:
        return None


def git_text(args: list[str], cwd: Path, env: dict | None = None, input: bytes | None = None) -> str | None:
    """Runs git and returns its stripped output, or None when it fails."""
    proc = git(args, cwd, env, input)
    if proc is None or proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace").strip()


def toplevel(path: Path) -> Path | None:
    out = git_text(["rev-parse", "--show-toplevel"], path)
    return Path(out) if out else None


def strip_credentials(url: str) -> str:
    """Removes a user or token from a remote URL such as https://token@host/repo."""
    return re.sub(r"(?<=://)[^/@]+@", "", url)
