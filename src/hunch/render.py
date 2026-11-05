"""Turning runs into text: the `ls` list, the `log` tree and the `show` page."""

from __future__ import annotations

import math
import shlex
from collections import Counter
from pathlib import Path

from .runner import format_duration
from .store import RunRecord, downsample, read_json

RAMP = "▁▂▃▄▅▆▇█"
RAMP_ASCII = "_.-:=+*#"
MARKERS = {"finished": "*", "crashed": "!", "killed": "!", "lost": "?", "running": ">"}
WHY_WIDTH = 36
NOTE_WIDTH = 40


class View:
    """Settings shared by everything that prints runs."""

    def __init__(self, runs: list[RunRecord], main_metric: str | None = None, unicode: bool = True):
        self.by_id = {run.id: run for run in runs}
        self.short = short_ids([run.id for run in runs])
        self.main_metric = main_metric
        self.unicode = unicode


def can_encode(stream, text: str) -> bool:
    try:
        text.encode(getattr(stream, "encoding", None) or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def short_ids(ids: list[str]) -> dict[str, str]:
    """The shortest unique ending of each id: 4 characters, then longer."""
    lengths = (4, 6, 11)
    counts = {n: Counter(i[-n:] for i in ids) for n in lengths}
    out = {}
    for run_id in ids:
        out[run_id] = next((run_id[-n:] for n in lengths if counts[n][run_id[-n:]] == 1), run_id)
    return out


def truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return truncate(str(value), 24)


def sparkline(values: list[float], unicode: bool = True, width: int = 8) -> str:
    points = downsample([float(v) for v in values], width)
    finite = [v for v in points if math.isfinite(v)]
    if not finite:
        return ""
    ramp = RAMP if unicode else RAMP_ASCII
    low, high = min(finite), max(finite)
    chars = []
    for value in points:
        if not math.isfinite(value):
            chars.append("!")
        elif high == low:
            chars.append(ramp[len(ramp) // 2])
        else:
            chars.append(ramp[round((value - low) / (high - low) * (len(ramp) - 1))])
    return "".join(chars)


def marker(run: RunRecord) -> str:
    return "x" if run.verdict == "drop" else MARKERS.get(run.status, "*")


def visible(runs: list[RunRecord], show_all: bool) -> list[RunRecord]:
    """Hides trivial runs, except those that other shown runs came from."""
    if show_all:
        return runs
    by_id = {run.id: run for run in runs}
    keep = {run.id for run in runs if not run.trivial}
    pending = list(keep)
    while pending:
        parent = by_id[pending.pop()].parent
        if parent in by_id and parent not in keep:
            keep.add(parent)
            pending.append(parent)
    return [run for run in runs if run.id in keep]


def _cells(run: RunRecord, view: View) -> list[str]:
    """id, why, result, sparkline, verdict and conclusion."""
    key = run.main_metric(view.main_metric)
    value = run.final_metrics().get(key) if key else None
    metric = f"{key} {format_value(value)}" if key and value is not None else ""
    status = run.status
    if status == "finished":
        result = metric
    else:
        if status == "crashed":
            code = (run.end or {}).get("exit_code")
            state = f"crashed, exit {code}" if code is not None else "crashed"
        elif status == "running":
            state = f"running {format_duration(run.duration)}"
        else:
            state = status
        result = f"{state}; {metric}" if metric else state
    tail = []
    if run.verdict:
        tail.append(run.verdict)
    if run.conclusion:
        tail.append(f'"{truncate(run.conclusion, NOTE_WIDTH)}"')
    elif status == "finished" and not run.verdict:
        tail.append("(no conclusion yet)")
    spark = sparkline(run.spark(key), view.unicode) if key else ""
    why = truncate(run.why, WHY_WIDTH) if run.why else "(no why)"
    return [view.short[run.id], why, result, spark, "  ".join(tail)]


def _table(rows: list[list[str]]) -> list[str]:
    """Aligns columns; the last column is not padded."""
    if not rows:
        return []
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    return ["  ".join(cell.ljust(widths[i]) if i < len(widths) else cell for i, cell in enumerate(row)).rstrip() for row in rows]


def ls_lines(runs: list[RunRecord], view: View) -> list[str]:
    rows = []
    for run in runs:
        sid, why, result, spark, tail = _cells(run, view)
        rows.append([marker(run), sid, f"{run.started:%b %d %H:%M}", why, result, spark, tail])
    return _table(rows)


def graph_order(runs: list[RunRecord]) -> tuple[list[RunRecord], dict[str, str | None]]:
    """Orders runs like `git log --graph --topo-order`: each run's descendants
    come right above it, newest branch first. Also returns each run's parent
    among the shown runs (None for a root)."""
    by_id = {run.id: run for run in runs}
    parent_of: dict[str, str | None] = {}
    children: dict[str, list[RunRecord]] = {run.id: [] for run in runs}
    roots = []
    for run in runs:
        parent = run.parent
        if parent in by_id and by_id[parent].started < run.started:
            parent_of[run.id] = parent
            children[parent].append(run)
        else:
            parent_of[run.id] = None
            roots.append(run)

    latest = {}  # newest start time in each run's subtree
    for run in sorted(runs, key=lambda r: r.started, reverse=True):
        latest[run.id] = max([run.started, *(latest[c.id] for c in children[run.id])])

    order = []
    stack = [(run, False) for run in sorted(roots, key=lambda r: latest[r.id])]
    while stack:
        run, expanded = stack.pop()
        if expanded:
            order.append(run)
            continue
        stack.append((run, True))
        stack.extend((child, False) for child in sorted(children[run.id], key=lambda r: latest[r.id]))
    return order, parent_of


def graph_rows(order: list[RunRecord], parent_of: dict[str, str | None]) -> list[tuple[str, RunRecord | None]]:
    """The graph column: lanes of `|`, run markers, and `/` where branches join."""
    lanes: list[str] = []
    rows: list[tuple[str, RunRecord | None]] = []
    for run in order:
        waiting = [i for i, wanted in enumerate(lanes) if wanted == run.id]
        if waiting:
            col = waiting[0]
            for lane in reversed(waiting[1:]):
                rows.append((" ".join("|" * lane) + "/", None))
                lanes.pop(lane)
        else:
            lanes.append(run.id)
            col = len(lanes) - 1
        rows.append((" ".join(marker(run) if i == col else "|" for i in range(len(lanes))), run))
        parent = parent_of[run.id]
        if parent:
            lanes[col] = parent
        else:
            lanes.pop(col)
            if not lanes and run is not order[-1]:
                rows.append(("", None))  # a blank line between unrelated trees
    return rows


def log_lines(runs: list[RunRecord], view: View) -> list[str]:
    order, parent_of = graph_order(runs)
    rows = graph_rows(order, parent_of)
    cells = [[f"{run.started:%b %d}", *_cells(run, view)] for _, run in rows if run is not None]
    table = iter(_table([[c[1], c[0], *c[2:]] for c in cells]))
    width = max((len(graph) for graph, _ in rows), default=0)
    return [graph.ljust(width) + "  " + next(table) if run else graph for graph, run in rows]


def run_counts(runs: list[RunRecord]) -> str:
    """"9 runs (2 kept, 3 dropped, 1 crashed)", leaving out quick crashes."""
    shown = [run for run in runs if not run.trivial]
    counts = [f"{n} {label}" for n, label in (
        (sum(r.verdict == "keep" for r in shown), "kept"),
        (sum(r.verdict == "drop" for r in shown), "dropped"),
        (sum(r.status in ("crashed", "killed") for r in shown), "crashed"),
    ) if n]
    return plural(len(shown), "run") + (f" ({', '.join(counts)})" if counts else "")


def milestone_lines(milestones: list, numbers: dict[str, str], runs: list[RunRecord], view: View) -> list[str]:
    """`hunch log` once there are milestones: newest first, each with its runs folded."""
    by_id = {run.id: run for run in runs}
    covered = {run_id for m in milestones for run_id in m.runs}
    pending = [run for run in runs if run.id not in covered and not run.trivial]
    diamond, bar, fold = ("◆", "│", "▸") if view.unicode else ("#", "|", ">")
    lines = []
    if pending:
        lines.append(f"  since the last milestone: {plural(len(pending), 'run')}   (hunch log --runs to see them)")
    newest_first = list(reversed(milestones))
    title_width = max((len(truncate(m.title, 48)) for m in newest_first), default=0)
    for index, m in enumerate(newest_first):
        best = by_id.get(m.best) if m.best else None
        best_text = ""
        if best is not None and m.metric:
            best_text = f"best {view.short.get(best.id, best.id[-4:])}  {m.metric} {format_value(best.final_metrics().get(m.metric))}"
        lines.append(f"{diamond} {numbers[m.id]:<4} {m.created:%b %d}  {truncate(m.title, 48).ljust(title_width)}  {best_text}".rstrip())
        side = " " if index == len(newest_first) - 1 else bar
        if m.description:
            lines.append(f'{side}       "{truncate(m.description, 70)}"')
        members = [by_id[run_id] for run_id in m.runs if run_id in by_id]
        lines.append(f"{side}     {fold} {run_counts(members)}   (hunch log {numbers[m.id]})")
    return lines


def milestone_show_lines(m, number: str, runs: list[RunRecord], view: View, root: Path) -> list[str]:
    by_id = {run.id: run for run in runs}
    lines: list[str] = []

    def add(label: str, text: str) -> None:
        lines.append(f"{label:<11}{text}")

    add("milestone", f"{number}  {m.id}")
    add("title", m.title)
    add("saved", f"{m.created:%Y-%m-%d %H:%M}" + (f" on {m.line}" if m.line else ""))
    if m.description:
        add("note", m.description)
    add("summary", m.summary)
    best = by_id.get(m.best) if m.best else None
    if best is not None:
        value = f"  {m.metric} {format_value(best.final_metrics().get(m.metric))}" if m.metric else ""
        add("best", f"{view.short.get(best.id, best.id)}  {best.why or '(no hypothesis)'}{value}")
    members = [by_id[run_id] for run_id in m.runs if run_id in by_id]
    for index, run in enumerate(members):
        add("runs" if index == 0 else "", f"{view.short.get(run.id, run.id)}  {truncate(run.why or '(no hypothesis)', 50)}")
    if not members:
        add("runs", "none")
    files = (read_json(m.dir / "files.json", {}) or {}).get("files") or {}
    add("code", f"{plural(len(files), 'file')} saved")
    code = m.meta.get("git") or {}
    if code.get("commit"):
        add("git", f"{code['commit'][:9]} on {code.get('branch') or 'detached HEAD'}")
    try:
        add("folder", str(m.dir.relative_to(root)))
    except ValueError:
        add("folder", str(m.dir))
    return lines


def show_lines(run: RunRecord, view: View, root: Path) -> list[str]:
    meta = run.meta
    code = meta.get("git") or {}
    env = read_json(run.dir / "env.json", {})
    config = read_json(run.dir / "config.json", {})
    end = run.end or {}
    lines: list[str] = []

    def add(label: str, text: str) -> None:
        lines.append(f"{label:<11}{text}")

    add("run", f"{run.id}  ({view.short.get(run.id, run.id)})")
    status = run.status
    duration = format_duration(run.duration)
    if status == "finished":
        add("status", f"finished in {duration}")
    elif status == "running":
        add("status", f"running for {duration}")
    elif status == "lost":
        add("status", "lost: no end was recorded and the heartbeat stopped")
    else:
        code_text = f", exit {end.get('exit_code')}" if end.get("exit_code") is not None else ""
        add("status", f"{status} after {duration}{code_text}")
    host = meta.get("host") or {}
    where = host.get("hostname", "?") + (f", SLURM job {host['slurm_job_id']}" if host.get("slurm_job_id") else "")
    add("started", f"{run.started:%Y-%m-%d %H:%M} on {where}")
    add("why", run.why or "(none)")
    if run.verdict or run.conclusion:
        add("verdict", "  ".join(filter(None, [run.verdict, f'"{run.conclusion}"' if run.conclusion else None])))
    if run.parent:
        parent = view.by_id.get(run.parent)
        label = f"{view.short.get(run.parent, run.parent)}  {parent.why or ''}".rstrip() if parent else run.parent
        how = " (set by hand)" if any(e.get("kind") == "parent" for e in run.notes) else (
            " (default: last run on this line)" if meta.get("parent_source") == "default" else "")
        add("parent", label + how)
    add("line", run.line or "(none)")
    add("command", shlex.join(meta.get("command") or []))

    manifest = read_json(run.dir / "files.json")
    if manifest is not None:
        text = f"{plural(len(manifest['files']), 'file')} saved ({format_bytes(manifest['bytes'])})"
        if manifest.get("skipped_count"):
            text += f"; {manifest['skipped_count']} too big or binary, not saved"
        add("code", text)
    else:
        add("code", "not saved (recorded before Hunch 0.2)")
    if code:
        commit = (code.get("commit") or "")[:9] or "no commits yet"
        dirty = ", with uncommitted changes" if code.get("dirty") else ""
        add("git", f"{commit} on {code.get('branch') or 'detached HEAD'}{dirty}")

    values = flatten(config.get("values") or {})
    if values:
        add("config", truncate(", ".join(f"{k}={format_value(v)}" for k, v in values.items()), 200))

    if env.get("error"):
        add("env", f"{env.get('executable', '?')}: {env['error']}")
    else:
        text = f"Python {env.get('python', '?')} ({env.get('executable', '?')})"
        torch = env.get("torch") or {}
        if torch.get("version"):
            text += f"; torch {torch['version']}" + (f", CUDA {torch['cuda']}" if torch.get("cuda") else "")
        if env.get("packages"):
            text += f"; {plural(len(env['packages']), 'package')}"
        add("env", text)
    if env.get("gpus"):
        names = Counter(gpu["name"] for gpu in env["gpus"])
        driver = env["gpus"][0].get("driver")
        add("gpu", ", ".join(f"{n} x {name}" if n > 1 else name for name, n in names.items()) + f"; driver {driver}")

    metrics = run.final_metrics()
    if metrics:
        key = run.main_metric(view.main_metric)
        ordered = ([key] if key in metrics else []) + [k for k in run.metric_keys() if k != key and k in metrics]
        parts = []
        for k in ordered[:8]:
            spark = sparkline(run.spark(k), view.unicode)
            parts.append(f"{k} {format_value(metrics[k])}" + (f" {spark}" if spark else ""))
        if len(ordered) > 8:
            parts.append(f"(+{len(ordered) - 8} more)")
        add("metrics", parts[0])
        for part in parts[1:]:
            add("", part)

    for event in run.notes:
        value = event.get("value")
        shown = f'"{value}"' if isinstance(value, str) and event.get("kind") != "verdict" else str(value)
        add("history", f"{event.get('at', '')[:16].replace('T', ' ')}  {event.get('kind')}: {shown}")
    try:
        folder = run.dir.relative_to(root)
    except ValueError:
        folder = run.dir
    add("folder", str(folder))
    return lines


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def format_bytes(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / 1024 / 1024:.1f} MB"


def flatten(values: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out
