"""`hunch diff`: what changed between two runs, and how the metrics moved."""

from __future__ import annotations

import math
import shlex
from pathlib import Path

from . import snapshot
from .render import View, flatten, format_value, truncate
from .store import RunRecord, read_json

PACKAGE_LIMIT = 10


def diff_lines(a: RunRecord, b: RunRecord, view: View, objects: Path, full_code: bool) -> list[str]:
    lines = []
    for run in (a, b):
        lines.append(f"{view.short[run.id]}  {run.started:%b %d %H:%M}  {truncate(run.why or '(no why)', 60)}")
    lines.append("")
    sections = [
        ("code", code_changes(a, b, objects, full_code)),
        ("config", config_changes(a, b)),
        ("env", env_changes(a, b)),
        ("metrics", metric_changes(a, b, view)),
    ]
    for label, changes in sections:
        changes = changes or ["no changes"]
        lines.append(f"{label:<9}{changes[0]}")
        lines.extend(change if label == "code" and full_code and i else f"{'':<9}{change}" for i, change in enumerate(changes[1:], 1))
    return lines


def code_changes(a: RunRecord, b: RunRecord, objects: Path, full: bool) -> list[str]:
    """Compares the two runs' snapshots. Needs no git."""
    manifests = [read_json(run.dir / "files.json") for run in (a, b)]
    if None in manifests:
        return ["not comparable: a run was recorded before Hunch saved code snapshots (0.1)"]
    changed = snapshot.changes(objects, manifests[0]["files"], manifests[1]["files"])
    if full:
        return ["full diff:", *(line for *_, diff in changed for line in diff)] if changed else []
    lines = [f"{path}  +{added} -{removed}" for path, added, removed, _ in changed]
    if lines:
        lines[-1] += "   (--code for the full diff)"
    return lines


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
