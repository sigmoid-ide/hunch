"""Jupyter notebooks: which notebook a run came from, the cells that actually ran,
and notebook files without their outputs.

A notebook's state comes from every cell the kernel has run, in the order it ran
them, including cells edited or run out of order since. So a run started in a
notebook appends each cell to cells.jsonl as it runs, starting with the cells run
before hunch.init().
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .store import append_jsonl, timestamp

CELLS_FILE = "cells.jsonl"
STRIP_LIMIT = 50 * 1024 * 1024  # notebooks bigger than this are not read at all


def shell():
    """The IPython shell this code runs in, or None. Never imports IPython itself."""
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return None
    try:
        return ipython.get_ipython()
    except Exception:
        return None


def command(ipython, cwd: Path) -> list[str] | None:
    """The notebook as the run's command, when this is a notebook. VS Code puts the
    notebook's path in the kernel's namespace and Jupyter Server in an environment
    variable. A kernel that tells neither is still a notebook, just unnamed."""
    path = ipython.user_ns.get("__vsc_ipynb_file__") or os.environ.get("JPY_SESSION_NAME")
    if path:
        try:
            return [Path(path).resolve().relative_to(cwd.resolve()).as_posix()]
        except ValueError:
            return [str(path)]
    if getattr(ipython, "kernel", None) is not None:
        return ["notebook"]
    return None  # the IPython terminal: its own command line says enough


class CellLog:
    """Appends every cell the kernel runs to the run's cells.jsonl."""

    def __init__(self, ipython, run_dir: Path):
        self.ipython = ipython
        self.path = run_dir / CELLS_FILE
        # The cells run before this one made the kernel's state, so they belong to the run.
        # The last cell in the history is the one running now; it is written when it ends.
        cells = self._history()
        for number, source in cells[:-1]:
            append_jsonl(self.path, {"n": number, "source": source})
        self._written = cells[-1][0] - 1 if cells else 0
        ipython.events.register("post_run_cell", self._ran)

    def _history(self) -> list[tuple[int, str]]:
        history = getattr(self.ipython, "history_manager", None)
        if history is None:
            return []
        return [(number, source) for _session, number, source in history.get_range(session=0, start=1, stop=None, raw=True)]

    def _ran(self, result) -> None:
        if result is None or not result.execution_count or result.execution_count <= self._written:
            return  # run without history, such as an editor's own queries
        row = {"n": result.execution_count, "source": result.info.raw_cell, "at": timestamp()}
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            row["error"] = f"{type(error).__name__}: {error}"[:300]
        append_jsonl(self.path, row)
        self._written = result.execution_count

    def stop(self) -> None:
        """Called when the run finishes, which happens inside a cell: that cell is part
        of the run too, though it has not ended yet."""
        try:
            self.ipython.events.unregister("post_run_cell", self._ran)
        except ValueError:
            pass
        cells = self._history()
        if cells and cells[-1][0] > self._written:
            number, source = cells[-1]
            append_jsonl(self.path, {"n": number, "source": source, "at": timestamp()})
            self._written = number


def strip_outputs(data: bytes) -> bytes:
    """A notebook without outputs, execution counts or saved widget state: the code
    and text only. Anything that is not a notebook comes back unchanged."""
    try:
        nb = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return data
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        return data
    for cell in nb["cells"]:
        if isinstance(cell, dict) and cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    if isinstance(nb.get("metadata"), dict):
        nb["metadata"].pop("widgets", None)
    return (json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def cell_lines(data: bytes) -> list[str] | None:
    """A notebook as its cells' sources, each under a "# %%" line, so a diff reads
    like code. None when the data is not a notebook."""
    try:
        nb = json.loads(data)
        cells = nb["cells"]
    except (ValueError, UnicodeDecodeError, KeyError, TypeError):
        return None
    lines: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        kind = cell.get("cell_type", "code")
        lines.append("# %%" if kind == "code" else f"# %% [{kind}]")
        source = cell.get("source", "")
        lines.extend(("".join(source) if isinstance(source, list) else str(source)).splitlines())
        lines.append("")
    return lines
