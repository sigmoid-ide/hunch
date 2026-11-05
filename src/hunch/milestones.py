"""Milestones: entries you save yourself, each covering the runs since the previous one.

Runs are recorded automatically; a milestone is the moment you decide something is
worth remembering. It keeps a title, your optional description, a summary Hunch drafts
from the runs it covers, the best of those runs, and the code at that moment.

On disk, .hunch/milestones/<id>/ holds files.json (the code snapshot) and milestone.json,
written last and never changed after.
"""

from __future__ import annotations

import re
import secrets
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import snapshot
from .capture import capture_git
from .store import FORMAT_VERSION, HunchError, RunRecord, Store, now, read_json, timestamp, write_json

# Words in a metric's name that say which direction is better.
LOWER_IS_BETTER = ("loss", "error", "err", "perplexity", "ppl", "wer", "cer", "mse", "mae", "rmse", "nll", "fid")
HIGHER_IS_BETTER = ("acc", "f1", "auc", "bleu", "rouge", "precision", "recall", "map", "iou", "dice", "reward", "score")
VARIED_KEYS = 5
VARIED_VALUES = 4


class Milestone:
    def __init__(self, path: Path, meta: dict):
        self.dir = path
        self.id = path.name
        self.meta = meta

    @classmethod
    def load(cls, path: Path) -> Milestone | None:
        meta = read_json(path / "milestone.json")
        if not isinstance(meta, dict) or "created_at" not in meta:
            return None
        return cls(path, meta)

    @property
    def created(self) -> datetime:
        return datetime.fromisoformat(self.meta["created_at"])

    @property
    def title(self) -> str:
        return self.meta.get("title") or ""

    @property
    def description(self) -> str | None:
        return self.meta.get("description")

    @property
    def summary(self) -> str:
        return self.meta.get("summary") or ""

    @property
    def runs(self) -> list[str]:
        return list(self.meta.get("runs") or [])

    @property
    def best(self) -> str | None:
        return self.meta.get("best")

    @property
    def metric(self) -> str | None:
        return self.meta.get("metric")

    @property
    def line(self) -> str | None:
        return self.meta.get("line")


def milestones(store: Store) -> list[Milestone]:
    folder = store.dir / "milestones"
    if not folder.is_dir():
        return []
    loaded = [Milestone.load(path) for path in folder.iterdir() if path.is_dir()]
    return sorted((m for m in loaded if m is not None), key=lambda m: (m.created, m.id))


def numbers(all_milestones: list[Milestone]) -> dict[str, str]:
    """M1, M2, ... in the order milestones were saved."""
    return {m.id: f"M{index}" for index, m in enumerate(all_milestones, 1)}


def resolve(store: Store, ref: str, all_milestones: list[Milestone] | None = None) -> Milestone:
    """Finds a milestone by its number (M3) or by the end of its id."""
    all_milestones = milestones(store) if all_milestones is None else all_milestones
    match = re.fullmatch(r"[Mm](\d+)", ref)
    if match:
        index = int(match.group(1))
        if 1 <= index <= len(all_milestones):
            return all_milestones[index - 1]
        raise HunchError(f"there is no milestone {ref.upper()}")
    found = [m for m in all_milestones if m.id.endswith(ref)]
    if len(found) == 1:
        return found[0]
    raise HunchError(f"no milestone matches '{ref}'" if not found else f"'{ref}' matches {len(found)} milestones")


def is_milestone_ref(ref: str) -> bool:
    return re.fullmatch(r"[Mm]\d+", ref) is not None


def lower_is_better(metric: str | None) -> bool | None:
    if not metric:
        return None
    name = metric.lower()
    if any(word in name for word in LOWER_IS_BETTER):
        return True
    if any(word in name for word in HIGHER_IS_BETTER):
        return False
    return None


def pick_best(runs: list[RunRecord], metric: str | None) -> RunRecord | None:
    """The finished run with the best final value of `metric`, when its name says which way is better."""
    lower = lower_is_better(metric)
    if lower is None:
        return None
    scored = []
    for run in runs:
        value = run.final_metrics().get(metric)
        if run.status == "finished" and isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
            scored.append((value, run))
    if not scored:
        return None
    return (min if lower else max)(scored, key=lambda pair: pair[0])[1]


def main_metric(store: Store, runs: list[RunRecord]) -> str | None:
    setting = store.config().get("main_metric")
    if setting:
        return setting
    counts = Counter(m for m in (run.main_metric() for run in runs) if m)
    return counts.most_common(1)[0][0] if counts else None


def draft(runs: list[RunRecord], metric: str | None, best: RunRecord | None) -> str:
    """A one-paragraph summary of the runs: how many and how they ended, what varied, the best one."""
    from .render import flatten, format_value, run_counts  # render is imported lazily to avoid a cycle

    shown = [run for run in runs if not run.trivial]
    if not shown:
        return "No runs since the previous milestone."
    sentences = [run_counts(shown)]

    values: dict[str, list[str]] = {}
    for run in shown:
        config = flatten((read_json(run.dir / "config.json", {}) or {}).get("values") or {})
        for key, value in config.items():
            text = format_value(value)
            if text not in values.setdefault(key, []):
                values[key].append(text)
    varied = [(key, texts) for key, texts in values.items() if len(texts) > 1][:VARIED_KEYS]
    if varied:
        sentences.append("Varied: " + "; ".join(
            f"{key} {', '.join(texts[:VARIED_VALUES])}" + (", ..." if len(texts) > VARIED_VALUES else "")
            for key, texts in varied
        ))
    if best is not None and metric:
        sentences.append(f"Best {metric} {format_value(best.final_metrics().get(metric))} ({best.id[-4:]})")
    return ". ".join(sentences) + "."


def create(store: Store, title: str, description: str | None = None, best_ref: str | None = None) -> Milestone:
    """Saves a milestone covering the ended runs on the current branch since the previous one."""
    title = title.strip()
    if not title:
        raise HunchError("a milestone needs a title")
    store.ensure()
    code = capture_git(store.root)
    line = code.get("branch") if code else None
    existing = milestones(store)
    covered = {run_id for milestone in existing for run_id in milestone.runs}
    all_runs = store.runs()
    runs = [r for r in all_runs if r.line == line and r.id not in covered and r.end is not None]
    metric = main_metric(store, runs)
    best = store.resolve(best_ref, all_runs) if best_ref else pick_best(runs, metric)
    previous = [m for m in existing if m.line == line]

    folder = _new_dir(store.dir / "milestones")
    write_json(folder / "files.json", snapshot.take(store.root, store.dir / "objects"))
    write_json(folder / "milestone.json", {
        "format": FORMAT_VERSION,
        "id": folder.name,
        "created_at": timestamp(),
        "title": title,
        "description": (description or "").strip() or None,
        "summary": draft(runs, metric, best),
        "line": line,
        "previous": previous[-1].id if previous else None,
        "runs": [run.id for run in runs],
        "best": best.id if best else None,
        "metric": metric,
        "git": code,
    })
    return Milestone.load(folder)


def _new_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = now().strftime("%Y%m%d-%H%M")
    for _ in range(50):
        path = parent / f"{stamp}-{secrets.token_hex(3)}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise HunchError("could not pick a unique milestone id")
