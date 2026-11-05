"""Code snapshots: every run keeps the exact files it ran with, with or without git.

Each file is stored once in .hunch/objects as a git blob object (zlib-compressed
"blob <size>\\0<content>", named by its SHA-1), so identical files across runs
cost nothing and a later `hunch push` can hand them to git unchanged. Each run
lists its files in files.json.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import os
import threading
import zlib
from pathlib import Path

from .gitinfo import git

FILE_LIMIT = 1024 * 1024
TOTAL_LIMIT = 20 * 1024 * 1024
COUNT_LIMIT = 5000
SCAN_LIMIT = 20000  # without git, stop walking a folder tree after this many files
LIST_LIMIT = 1000
SKIP_DIRS = {
    ".git", ".hunch", "__pycache__", "node_modules", ".ipynb_checkpoints", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", "site-packages", "wandb", "mlruns", "lightning_logs",
}
ENV_MARKERS = ("pyvenv.cfg", "conda-meta")  # a folder with one of these is a Python environment
SECRET_FILES = (".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa", "id_rsa.*",
                "id_ed25519", "id_ed25519.*", ".netrc", ".pypirc")
IGNORE_FILE = ".hunchignore"


def take(root: Path, objects: Path) -> dict:
    """Saves the project's code files into `objects` and returns the manifest for
    files.json. Uses git's file list (so .gitignore applies) when `root` is in a
    repository, otherwise walks the folder."""
    patterns = _ignore_patterns(root)
    names = _git_files(root)
    truncated = False
    if names is None:
        names, truncated = _walk(root)
    env_dirs: dict[str, bool] = {}
    files, skipped, total = {}, [], 0
    for rel in sorted(set(names)):
        if _ignored(rel, patterns, root, env_dirs):
            continue
        path = root / rel
        try:
            info = path.lstat()
        except OSError:
            continue  # listed by git but deleted from disk
        # A symlink's target may be outside the project, such as a credentials file.
        if path.is_symlink() or info.st_size > FILE_LIMIT or total + info.st_size > TOTAL_LIMIT or len(files) >= COUNT_LIMIT:
            skipped.append(rel)
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:  # binary
            skipped.append(rel)
            continue
        files[rel] = {"sha": store_object(objects, data), "size": len(data)}
        total += len(data)
    return {
        "files": files,
        "bytes": total,
        "skipped": skipped[:LIST_LIMIT],
        "skipped_count": len(skipped),
        "scan_truncated": truncated,
    }


def store_object(objects: Path, data: bytes) -> str:
    raw = b"blob %d\0" % len(data) + data
    sha = hashlib.sha1(raw, usedforsecurity=False).hexdigest()
    path = objects / sha[:2] / sha[2:]
    if path.exists():
        return sha
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{sha[2:]}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(zlib.compress(raw))
    try:
        os.replace(tmp, path)
    except PermissionError:  # Windows: another run is writing the same object right now
        tmp.unlink(missing_ok=True)
        if not path.exists():
            raise
    return sha


def read_object(objects: Path, sha: str) -> bytes | None:
    try:
        raw = zlib.decompress((objects / sha[:2] / sha[2:]).read_bytes())
    except (OSError, zlib.error):
        return None
    return raw[raw.index(b"\0") + 1 :]


def changes(objects: Path, before: dict, after: dict) -> list[tuple[str, int, int, list[str]]]:
    """Each changed file as (path, lines added, lines removed, unified diff lines)."""
    out = []
    for path in sorted(set(before) | set(after)):
        old, new = before.get(path), after.get(path)
        if old and new and old["sha"] == new["sha"]:
            continue
        old_lines = _lines(objects, old)
        new_lines = _lines(objects, new)
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{path}" if old else "/dev/null",
            tofile=f"b/{path}" if new else "/dev/null",
            lineterm="",
        ))
        added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
        out.append((path, added, removed, diff))
    return out


def _lines(objects: Path, entry: dict | None) -> list[str]:
    if entry is None:
        return []
    data = read_object(objects, entry["sha"])
    if data is None:
        return ["(missing from .hunch/objects)"]
    return data.decode("utf-8", "replace").splitlines()


def _git_files(root: Path) -> list[str] | None:
    proc = git(["ls-files", "-z", "--cached", "--others", "--exclude-standard"], root)
    if proc is None or proc.returncode != 0:
        return None
    return [name for name in proc.stdout.decode("utf-8", "replace").split("\0") if name]


def _walk(root: Path) -> tuple[list[str], bool]:
    names: list[str] = []
    for folder, dirs, filenames in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".") and not _is_env(Path(folder, d))]
        for name in filenames:
            names.append(Path(folder, name).relative_to(root).as_posix())
            if len(names) >= SCAN_LIMIT:
                return names, True
    return names, False


def _is_env(folder: Path) -> bool:
    return any((folder / marker).exists() for marker in ENV_MARKERS)


def _ignore_patterns(root: Path) -> list[str]:
    try:
        text = (root / IGNORE_FILE).read_text(encoding="utf-8")
    except OSError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def _ignored(rel: str, patterns: list[str], root: Path, env_dirs: dict[str, bool]) -> bool:
    """Folders Hunch always skips, secrets, .hunchignore patterns and Python environments."""
    parts = rel.split("/")
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return True
    if any(fnmatch.fnmatch(parts[-1], secret) for secret in SECRET_FILES):
        return True
    for pattern in patterns:
        if pattern.endswith("/"):
            if any(fnmatch.fnmatch(part, pattern[:-1]) for part in parts[:-1]):
                return True
        elif "/" in pattern:
            if fnmatch.fnmatch(rel, pattern.lstrip("/")):
                return True
        elif any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
    for depth in range(1, len(parts)):
        folder = "/".join(parts[:depth])
        if folder not in env_dirs:
            env_dirs[folder] = _is_env(root / folder)
        if env_dirs[folder]:
            return True
    return False
