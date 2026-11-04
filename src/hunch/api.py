"""The in-code API. Its names follow W&B, so `import hunch as wandb` works for
the common calls: init, log, config, summary and finish."""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from .capture import capture_env, start_run
from .runner import RUN_DIR_ENV
from .store import Heartbeat, Store, append_jsonl, finish_run, read_json, write_json

_active: Run | None = None
_exit_status: str | None = None
_hooks_installed = False

# Set by distributed launchers: torchrun and friends, then MPI.
RANK_VARS = ("RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK")
# srun sets this for every task, including independent sweep workers, so it only
# decides who records when the tasks share one `hunch run`.
JOINED_RANK_VARS = (*RANK_VARS, "SLURM_PROCID")


class Config(dict):
    """A dict that also allows attribute access and saves itself when it changes."""

    def __init__(self, data=None, on_change=None):
        super().__init__(data or {})
        object.__setattr__(self, "_on_change", on_change)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._changed()

    def update(self, other=(), /, allow_val_change=None, **kwargs):
        super().update(_as_dict(other), **kwargs)
        self._changed()

    def _changed(self):
        callback = object.__getattribute__(self, "_on_change")
        if callback is not None:
            callback(dict(self))

    def __reduce__(self):
        # Pickles and deep-copies as a plain dict, for example inside torch.save checkpoints.
        return (dict, (dict(self),))


class Run:
    """A run being recorded by this process. A run without a folder records
    nothing, for example in processes other than rank 0."""

    def __init__(self, run_dir: Path | None, owned: bool, config: dict | None = None):
        self.dir = run_dir
        self.id = run_dir.name if run_dir else None
        self._owned = owned
        self._finished = False
        self._step = 0
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._heartbeat: Heartbeat | None = None
        self.config = Config(config, self._save_config if run_dir else None)
        self.summary = Config(None, self._save_summary if run_dir else None)

    def log(self, data: Mapping, step: int | None = None, commit=None) -> None:
        if self.dir is None:
            return
        if self._finished:
            raise RuntimeError("this hunch run has finished; call hunch.init() to start a new one")
        if not isinstance(data, Mapping):
            raise TypeError("hunch.log() takes a dict of metric names to values")
        with self._lock:
            step = self._step if step is None else int(step)
            self._step = step + 1
            row = {"_step": step, "_time": round(time.time(), 3)}
            row.update((str(key), _scalar(value)) for key, value in data.items())
            append_jsonl(self.dir / "metrics.jsonl", row)

    def finish(self, exit_code: int | None = None, quiet=None) -> None:
        global _active
        if self._finished:
            return
        self._finished = True
        if self._heartbeat is not None:
            self._heartbeat.stop()
        if self._owned and self.dir is not None:
            finish_run(
                self.dir,
                status="finished" if not exit_code else "crashed",
                exit_code=exit_code or 0,
                duration=time.monotonic() - self._started,
            )
        if _active is self:
            _active = None

    def _save_config(self, values: dict) -> None:
        self._check_open()
        config = read_json(self.dir / "config.json", {"args": [], "values": {}})
        config["values"] = values
        write_json(self.dir / "config.json", config)

    def _save_summary(self, values: dict) -> None:
        self._check_open()
        write_json(self.dir / "summary.json", {key: _scalar(value) for key, value in values.items()})

    def _check_open(self) -> None:
        if self._finished:  # after the end, a run's record never changes
            raise RuntimeError("this hunch run has finished; its config and summary can no longer change")


def init(
    config=None,
    why: str | None = None,
    main_metric: str | None = None,
    *,
    notes: str | None = None,
    project: str | None = None,
    name: str | None = None,
    tags=None,
    **_ignored,
) -> Run:
    """Starts recording a run, or joins the run of a surrounding `hunch run`.

    `notes` is W&B's name for the why. Other W&B arguments are accepted and ignored.
    """
    global _active
    values = _as_dict(config) if config is not None else {}
    joined = os.environ.get(RUN_DIR_ENV)
    joinable = bool(joined) and (Path(joined) / "run.json").is_file() and not (Path(joined) / "end.json").exists()
    if _rank(JOINED_RANK_VARS if joinable else RANK_VARS) != "0":
        _active = Run(None, owned=False, config=values)  # in distributed training only rank 0 records
        return _active
    if _active is not None and not _active._finished:
        if not _active._owned:
            return _active
        _active.finish()

    why = why or notes
    labels = {key: value for key, value in (("project", project), ("name", name), ("tags", list(tags or []))) if value}
    if joinable:
        run = _join(Path(joined), why, main_metric, labels, values)
    else:
        run_dir = start_run(
            Store.find(),
            command=[sys.executable, *sys.argv],
            cwd=Path.cwd(),
            why=why,
            parent=None,
            source="init",
            python=None,
            main_metric=main_metric,
            labels=labels,
            config=values,
        )
        run = Run(run_dir, owned=True, config=values)
        run._heartbeat = Heartbeat(run_dir / "heartbeat")
        run._heartbeat.start()
        _install_hooks()
    _active = run
    return run


def _join(run_dir: Path, why, main_metric, labels, values) -> Run:
    """Adds what only the training process knows to the wrapper's run."""
    meta = read_json(run_dir / "run.json", {})
    changed = False
    if why and not meta.get("why"):
        meta["why"] = why
        changed = True
    if main_metric:
        meta["main_metric"] = main_metric
        changed = True
    if labels:
        meta["labels"] = {**meta.get("labels", {}), **labels}
        changed = True
    if changed:
        write_json(run_dir / "run.json", meta)
    write_json(run_dir / "env.json", capture_env(None))
    existing = read_json(run_dir / "config.json", {}).get("values", {})
    run = Run(run_dir, owned=False, config={**existing, **values})
    if values:
        run._save_config(dict(run.config))
    return run


def log(data: Mapping, step: int | None = None, commit=None) -> None:
    if _active is None:
        raise RuntimeError("call hunch.init() before hunch.log()")
    _active.log(data, step=step, commit=commit)


def finish(exit_code: int | None = None, quiet=None) -> None:
    if _active is not None:
        _active.finish(exit_code=exit_code)


def _install_hooks() -> None:
    """Records how the process ended when the script never calls finish()."""
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True
    previous = sys.excepthook

    def excepthook(kind, value, traceback):
        global _exit_status
        _exit_status = "killed" if issubclass(kind, KeyboardInterrupt) else "crashed"
        previous(kind, value, traceback)

    sys.excepthook = excepthook
    atexit.register(_at_exit)


def _at_exit() -> None:
    run = _active
    if run is None or run._finished or not run._owned or run.dir is None:
        return
    run._finished = True
    if run._heartbeat is not None:
        run._heartbeat.stop()
    status = _exit_status or "finished"
    finish_run(
        run.dir,
        status=status,
        exit_code=0 if status == "finished" else None,
        duration=time.monotonic() - run._started,
    )


def _rank(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return "0"


def _as_dict(config) -> dict:
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):  # argparse.Namespace and similar
        return dict(vars(config))
    return dict(config)


def _scalar(value):
    """Turns tensors and NumPy scalars into plain numbers that JSON can hold."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError, RuntimeError):
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
