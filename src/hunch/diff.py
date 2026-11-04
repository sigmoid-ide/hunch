"""`hunch diff`: what changed between two runs, and how the metrics moved."""

from __future__ import annotations

import math
import os
import shlex
import tempfile
from pathlib import Path

from .gitinfo import git, git_text, toplevel
from .render import View, flatten, format_value, truncate
from .store import RunRecord, read_json

PACKAGE_LIMIT = 10


def diff_lines(a: RunRecord, b: RunRecord, view: View, root: Path, full_code: bool) -> list[str]:
    lines = []
    for run in (a, b):
        lines.append(f"{view.short[run.id]}  {run.started:%b %d %H:%M}  {truncate(run.why or '(no why)', 60)}")
    lines.append("")
    sections = [
        ("code", code_changes(a, b, root, full_code)),
        ("config", config_changes(a, b)),
        ("env", env_changes(a, b)),
        ("metrics", metric_changes(a, b, view)),
    ]
    for label, changes in sections:
        changes = changes or ["no changes"]
        lines.append(f"{label:<9}{changes[0]}")
        lines.extend(change if label == "code" and full_code and i else f"{'':<9}{change}" for i, change in enumerate(changes[1:], 1))
    return lines


def code_changes(a: RunRecord, b: RunRecord, root: Path, full: bool) -> list[str]:
    git_a, git_b = a.meta.get("git"), b.meta.get("git")
    if not git_a or not git_b:
        return ["not comparable: at least one run was not in a git repository"]
    if _same_code(a, b):
        return []
    repo = toplevel(root)
    if repo is None:
        return ["not comparable: this folder is not in a git repository"]
    built = [code_tree(repo, run) for run in (a, b)]
    problems = [f"{run.id[-4:]}: {problem}" for run, (tree, problem) in zip((a, b), built) if tree is None]
    if problems:
        return ["files not comparable here", *problems]
    trees = [tree for tree, _ in built]
    if full:
        proc = git(["diff", "--no-color", "--no-ext-diff", trees[0], trees[1]], repo)
        text = proc.stdout.decode("utf-8", "replace") if proc and proc.returncode == 0 else ""
        text_lines = text.rstrip("\n").splitlines()
        return ["full diff:", *text_lines] if text_lines else []
    out = git_text(["diff", "--numstat", trees[0], trees[1]], repo) or ""
    changes = []
    for line in out.splitlines():
        added, deleted, path = line.split("\t", 2)
        changes.append(f"{path}  binary" if added == "-" else f"{path}  +{added} -{deleted}")
    if changes:
        changes[-1] += "   (--code for the full diff)"
    return changes


def _same_code(a: RunRecord, b: RunRecord) -> bool:
    if a.meta["git"].get("commit") != b.meta["git"].get("commit"):
        return False
    if _read_bytes(a.dir / "code.patch") != _read_bytes(b.dir / "code.patch"):
        return False
    return _untracked(a) == _untracked(b)


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


def _untracked(run: RunRecord) -> dict[str, bytes]:
    folder = run.dir / "untracked"
    if not folder.is_dir():
        return {}
    return {f.relative_to(folder).as_posix(): f.read_bytes() for f in folder.rglob("*") if f.is_file()}


def code_tree(repo: Path, run: RunRecord) -> tuple[str | None, str | None]:
    """Rebuilds a run's exact code as a git tree, using a temporary index, so the
    repository's own index and files are never touched. Returns the tree, or
    None and the reason it could not be rebuilt."""
    commit = run.meta["git"].get("commit")
    with tempfile.TemporaryDirectory() as temp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temp) / "index")}
        if commit:
            if git_text(["cat-file", "-e", f"{commit}^{{commit}}"], repo) is None:
                return None, f"commit {commit[:9]} is not in this repository"
            if git_text(["read-tree", commit], repo, env) is None:
                return None, f"could not read commit {commit[:9]}"
        elif git_text(["read-tree", "--empty"], repo, env) is None:
            return None, "could not start from an empty tree"
        patch = run.dir / "code.patch"
        if patch.is_file() and patch.stat().st_size:
            if git_text(["apply", "--cached", "--whitespace=nowarn", str(patch)], repo, env) is None:
                return None, "its uncommitted changes (code.patch) did not apply"
        folder = run.dir / "untracked"
        if folder.is_dir():
            for file in sorted(folder.rglob("*")):
                if not file.is_file():
                    continue
                name = file.relative_to(folder).as_posix()
                blob = git_text(["hash-object", "-w", f"--path={name}", str(file)], repo)
                if blob is None or git_text(["update-index", "--add", "--cacheinfo", f"100644,{blob},{name}"], repo, env) is None:
                    return None, f"could not add the untracked file {name}"
        tree = git_text(["write-tree"], repo, env)
        return (tree, None) if tree else (None, "could not write the rebuilt tree")


def config_changes(a: RunRecord, b: RunRecord) -> list[str]:
    config_a = read_json(a.dir / "config.json", {})
    config_b = read_json(b.dir / "config.json", {})
    changes = []
    values_a, values_b = flatten(config_a.get("values") or {}), flatten(config_b.get("values") or {})
    for key in sorted(set(values_a) | set(values_b)):
        if values_a.get(key, _UNSET) != values_b.get(key, _UNSET):
            changes.append(f"{key}: {_show(values_a.get(key, _UNSET))} -> {_show(values_b.get(key, _UNSET))}")
    if (config_a.get("args") or [])[1:] != (config_b.get("args") or [])[1:]:
        changes.append(f"command: {shlex.join(config_a.get('args') or [])}")
        changes.append(f"      -> {shlex.join(config_b.get('args') or [])}")
    return changes


_UNSET = object()


def _show(value) -> str:
    return "(unset)" if value is _UNSET else format_value(value)


def env_changes(a: RunRecord, b: RunRecord) -> list[str]:
    env_a = read_json(a.dir / "env.json", {})
    env_b = read_json(b.dir / "env.json", {})
    changes = []

    def compare(label, value_a, value_b):
        if value_a != value_b:
            changes.append(f"{label}: {value_a or '(none)'} -> {value_b or '(none)'}")

    compare("python", env_a.get("python"), env_b.get("python"))
    torch_a, torch_b = env_a.get("torch") or {}, env_b.get("torch") or {}
    compare("torch", torch_a.get("version"), torch_b.get("version"))
    compare("cuda", torch_a.get("cuda"), torch_b.get("cuda"))
    gpus_a = ", ".join(g["name"] for g in env_a.get("gpus") or [])
    gpus_b = ", ".join(g["name"] for g in env_b.get("gpus") or [])
    compare("gpus", gpus_a, gpus_b)
    compare("driver", ((env_a.get("gpus") or [{}])[0]).get("driver"), ((env_b.get("gpus") or [{}])[0]).get("driver"))
    compare("host", (a.meta.get("host") or {}).get("hostname"), (b.meta.get("host") or {}).get("hostname"))

    packages_a, packages_b = env_a.get("packages") or {}, env_b.get("packages") or {}
    package_changes = []
    for name in sorted(set(packages_a) | set(packages_b), key=str.lower):
        if name in ("torch",):
            continue
        version_a, version_b = packages_a.get(name), packages_b.get(name)
        if version_a != version_b:
            package_changes.append(f"{name} {version_a or '(none)'} -> {version_b or '(none)'}")
    changes += package_changes[:PACKAGE_LIMIT]
    if len(package_changes) > PACKAGE_LIMIT:
        changes.append(f"(+{len(package_changes) - PACKAGE_LIMIT} more package changes)")
    return changes


def metric_changes(a: RunRecord, b: RunRecord, view: View) -> list[str]:
    metrics_a, metrics_b = a.final_metrics(), b.final_metrics()
    key = b.main_metric(view.main_metric) or a.main_metric(view.main_metric)
    keys = [k for k in b.metric_keys() + a.metric_keys() if k in metrics_a and k in metrics_b]
    keys = list(dict.fromkeys(([key] if key in keys else []) + keys))
    changes = []
    for k in keys:
        value_a, value_b = metrics_a[k], metrics_b[k]
        line = f"{k} {format_value(value_a)} -> {format_value(value_b)}"
        if _number(value_a) and _number(value_b) and math.isfinite(value_a) and math.isfinite(value_b):
            line += f" ({value_b - value_a:+.4g})"
        changes.append(line)
    only = sorted(set(metrics_a) ^ set(metrics_b))
    if only:
        changes.append(f"in one run only: {', '.join(only[:8])}" + (" ..." if len(only) > 8 else ""))
    return changes


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
