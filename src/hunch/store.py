"""Runs on disk: where they live, and how they are written, read and annotated.

The layout is specified in docs/format.md. A run folder is written once, except
for the streams appended while the run is in progress. After end.json exists,
only notes.jsonl changes, and only by appending.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path

from .gitinfo import toplevel

FORMAT_VERSION = 2
STORE_DIR = ".hunch"
TRIVIAL_SECONDS = 30
HEARTBEAT_SECONDS = 60
REMOTE_FILE = "remote.json"  # written by Ferry next to a run that is copied home from another machine
LOST_AFTER_SECONDS = 300
SPARK_POINTS = 32

_MISSING = object()


class HunchError(Exception):
    """An error shown to the user as a one-line message."""


def now() -> datetime:
    return datetime.now().astimezone()


def timestamp(when: datetime | None = None) -> str:
    # Milliseconds, so runs started or ended within the same second still sort correctly.
    return (when or now()).isoformat(timespec="milliseconds")


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _dumps(data, indent: int | None = None) -> str:
    text = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:  # lone surrogates, such as a non-UTF-8 file name on Linux
        text = json.dumps(data, indent=indent, ensure_ascii=True, default=str)
    return text


def write_json(path: Path, data) -> None:
    """Writes atomically, so a reader never sees a half-written file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(_dumps(data, indent=2) + "\n", encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows, while another process has the file open
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def read_jsonl(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    rows = []
    for line in text.split("\n"):  # not splitlines(): U+2028 and friends may appear inside values
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a last line cut short by a crash
    return rows


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(_dumps(record) + "\n")


def downsample(values: list[float], points: int) -> list[float]:
    """Averages values into at most `points` buckets, skipping NaN and infinity."""
    if len(values) <= points:
        return list(values)
    out = []
    for i in range(points):
        chunk = values[i * len(values) // points : (i + 1) * len(values) // points]
        finite = [v for v in chunk if math.isfinite(v)]
        out.append(sum(finite) / len(finite) if finite else chunk[-1])
    return out


def summarize_metrics(rows: list[dict]) -> tuple[dict, list[str], dict[str, list[float]]]:
    """Returns the last value of each metric, the metric names in the order they
    first appeared, and a short series of each numeric metric for sparklines."""
    last: dict = {}
    keys: list[str] = []
    series: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key.startswith("_"):
                continue
            if key not in last:
                keys.append(key)
            last[key] = value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                series.setdefault(key, []).append(float(value))
    sparks = {key: downsample(values, SPARK_POINTS) for key, values in series.items()}
    return last, keys, sparks


def finish_run(run_dir: Path, *, status: str, exit_code: int | None, duration: float) -> None:
    rows = read_jsonl(run_dir / "metrics.jsonl")
    last, keys, sparks = summarize_metrics(rows)
    summary = read_json(run_dir / "summary.json", {})
    last.update(summary)
    keys += [key for key in summary if key not in keys]
    write_json(
        run_dir / "end.json",
        {
            "status": status,
            "exit_code": exit_code,
            "ended_at": timestamp(),
            "duration": round(duration, 3),
            "steps": rows[-1].get("_step") if rows else None,
            "metrics": last,
            "metric_keys": keys,
            "sparks": sparks,
        },
    )


class Heartbeat:
    """Touches a file every minute, so readers can tell a running run from a lost one."""

    def __init__(self, path: Path, interval: float = HEARTBEAT_SECONDS):
        self.path = path
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, name="hunch-heartbeat", daemon=True)

    def start(self) -> None:
        self.path.touch()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _beat(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.path.touch()
            except OSError:
                pass


class RunRecord:
    """One run folder, read back."""

    def __init__(self, path: Path, meta: dict, end: dict | None, notes: list[dict]):
        self.dir = path
        self.id = path.name
        self.meta = meta
        self.end = end
        self.notes = notes
        self._rows: list[dict] | None = None

    @classmethod
    def load(cls, path: Path) -> RunRecord | None:
        meta = read_json(path / "run.json")
        if not isinstance(meta, dict) or "started_at" not in meta:
            return None  # not a run yet, or not a run at all
        return cls(path, meta, read_json(path / "end.json"), read_jsonl(path / "notes.jsonl"))

    @property
    def started(self) -> datetime:
        return datetime.fromisoformat(self.meta["started_at"])

    @property
    def ended(self) -> datetime | None:
        return datetime.fromisoformat(self.end["ended_at"]) if self.end else None

    @property
    def duration(self) -> float | None:
        if self.end:
            return self.end.get("duration")
        return (now() - self.started).total_seconds()

    @property
    def status(self) -> str:
        if self.end:
            return self.end.get("status", "finished")
        if (self.dir / REMOTE_FILE).exists():
            # Copied from another machine (Ferry): this computer may have slept, so a quiet
            # heartbeat here says nothing about the run there.
            return "running"
        try:
            seen = (self.dir / "heartbeat").stat().st_mtime
        except FileNotFoundError:
            seen = self.started.timestamp()
        return "running" if time.time() - seen < LOST_AFTER_SECONDS else "lost"

    @property
    def trivial(self) -> bool:
        """A quick crash or interrupt, such as a typo. Quick runs that finished count:
        study scripts often take seconds."""
        if self.end is None or self.end.get("status", "finished") == "finished":
            return False
        return (self.end.get("duration") or 0) < TRIVIAL_SECONDS

    def _latest_note(self, kind: str):
        for event in reversed(self.notes):
            if event.get("kind") == kind:
                return event.get("value")
        return _MISSING

    @property
    def why(self) -> str | None:
        value = self._latest_note("why")
        return self.meta.get("why") if value is _MISSING else value

    @property
    def conclusion(self) -> str | None:
        value = self._latest_note("note")
        return None if value is _MISSING else value

    @property
    def verdict(self) -> str | None:
        value = self._latest_note("verdict")
        return None if value is _MISSING else value

    @property
    def parent(self) -> str | None:
        value = self._latest_note("parent")
        return self.meta.get("parent") if value is _MISSING else value

    @property
    def line(self) -> str | None:
        return self.meta.get("line")

    def rows(self) -> list[dict]:
        if self._rows is None:
            self._rows = read_jsonl(self.dir / "metrics.jsonl")
        return self._rows

    def _summary(self) -> tuple[dict, list[str], dict]:
        if self.end:
            return self.end.get("metrics", {}), self.end.get("metric_keys", []), self.end.get("sparks", {})
        last, keys, sparks = summarize_metrics(self.rows())
        return last, keys, sparks

    def final_metrics(self) -> dict:
        return self._summary()[0]

    def metric_keys(self) -> list[str]:
        return self._summary()[1]

    def spark(self, key: str) -> list[float]:
        return self._summary()[2].get(key, [])

    def main_metric(self, override: str | None = None) -> str | None:
        """The project setting, then the run's own setting, then the first metric
        with "val" in its name, then the last metric logged."""
        if override:
            return override
        if self.meta.get("main_metric"):
            return self.meta["main_metric"]
        keys = self.metric_keys()
        for key in keys:
            if "val" in key.lower():
                return key
        return keys[-1] if keys else None


class Store:
    """The .hunch folder of one project."""

    def __init__(self, root: Path):
        self.root = root
        self.dir = root / STORE_DIR
        self.runs_dir = self.dir / "runs"

    @classmethod
    def find(cls, start: Path | None = None) -> Store:
        """In a git repository: the nearest folder with a .hunch store between
        `start` and the repository root, else the root. Outside git: `start`.
        The search never leaves the repository, so unrelated projects never mix."""
        start = Path(start or os.getcwd()).resolve()
        top = toplevel(start)
        if top is None:
            return cls(start)
        top = top.resolve()
        for folder in (start, *start.parents):
            if (folder / STORE_DIR / "runs").is_dir():
                return cls(folder)
            if folder == top:
                break
        return cls(top)

    def ensure(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        ignore = self.dir / ".gitignore"
        if not ignore.exists():
            ignore.write_text("*\n", encoding="utf-8")

    def config(self) -> dict:
        return read_json(self.dir / "config.json", {})

    def set_config(self, key: str, value) -> None:
        self.ensure()
        config = self.config()
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
        write_json(self.dir / "config.json", config)

    def runs(self) -> list[RunRecord]:
        if not self.runs_dir.is_dir():
            return []
        records = [RunRecord.load(path) for path in self.runs_dir.iterdir() if path.is_dir()]
        return sorted((r for r in records if r is not None), key=lambda r: (r.started, r.id))

    def new_run_dir(self) -> Path:
        stamp = now().strftime("%Y%m%d-%H%M")
        for _ in range(50):
            path = self.runs_dir / f"{stamp}-{secrets.token_hex(3)}"
            try:
                path.mkdir()
                return path
            except FileExistsError:
                continue
        raise HunchError("could not pick a unique run id")

    def resolve(self, ref: str, runs: list[RunRecord] | None = None) -> RunRecord:
        """Finds a run by its full id, the end of its id, or any part of it."""
        runs = self.runs() if runs is None else runs
        for match in (lambda r: r.id == ref, lambda r: r.id.endswith(ref), lambda r: ref in r.id):
            found = [r for r in runs if match(r)]
            if len(found) == 1:
                return found[0]
            if len(found) > 1:
                names = ", ".join(r.id for r in found[:5])
                raise HunchError(f"'{ref}' matches {len(found)} runs ({names}); use more characters")
        raise HunchError(f"no run matches '{ref}'")

    def default_parent(self, line: str | None) -> str | None:
        """The run that ended most recently on the same line, ignoring trivial
        and dropped runs and runs still in progress."""
        best = None
        for run in self.runs():
            if run.line != line or run.end is None or run.trivial or run.verdict == "drop":
                continue
            if best is None or run.ended > best.ended:
                best = run
        return best.id if best else None

    def annotate(self, run: RunRecord, kind: str, value) -> None:
        append_jsonl(run.dir / "notes.jsonl", {"at": timestamp(), "kind": kind, "value": value})
