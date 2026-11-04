"""Hunch: a zero-config experiment journal for ML research.

Every run keeps its code, config, environment, metrics and the reason you ran
it, as plain files in the project's .hunch folder.

    import hunch
    run = hunch.init(config={"lr": 3e-4}, why="warmup to fix loss spikes")
    hunch.log({"val_loss": 2.31}, step=1000)
"""

from importlib import metadata as _metadata

from .api import Run, finish, init, log

try:
    __version__ = _metadata.version("sigmoid-hunch")
except _metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["Run", "init", "log", "finish", "config", "summary", "run"]


def __getattr__(name):
    # hunch.config, hunch.summary and hunch.run follow the active run, as in W&B.
    if name in ("config", "summary", "run"):
        from . import api

        if name == "run":
            return api._active
        if api._active is None:
            raise AttributeError(f"hunch.{name} is available after hunch.init()")
        return getattr(api._active, name)
    raise AttributeError(f"module 'hunch' has no attribute '{name}'")
