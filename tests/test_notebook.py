"""Jupyter notebooks: the run names its notebook, keeps the cells that actually ran,
and notebook files are saved and diffed without their outputs."""

import json

import pytest
from conftest import hunch, wrap
from test_run import saved_files

from hunch import api
from hunch.store import Store, read_jsonl

InteractiveShell = pytest.importorskip("IPython.core.interactiveshell").InteractiveShell


@pytest.fixture
def ipython(repo, tmp_path, monkeypatch):
    """An IPython shell, like the one inside a Jupyter kernel, in the repo."""
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    monkeypatch.delenv("JPY_SESSION_NAME", raising=False)
    from traitlets.config import Config

    config = Config()
    config.HistoryManager.hist_file = ":memory:"
    shell = InteractiveShell.instance(config=config)
    yield shell
    if api._active is not None:
        api._active.finish()
    api._active = None
    InteractiveShell.clear_instance()


def run_cells(shell, *cells):
    for cell in cells:
        shell.run_cell(cell, store_history=True)


def notebook(*sources, outputs=()) -> str:
    cells = [{"cell_type": "code", "execution_count": 1, "metadata": {}, "source": source,
              "outputs": list(outputs)} for source in sources]
    return json.dumps({"cells": cells, "metadata": {"widgets": {"state": "x" * 100}}, "nbformat": 4, "nbformat_minor": 5})


def test_a_run_names_its_notebook_and_keeps_every_cell_that_ran(ipython, repo):
    ipython.user_ns["__vsc_ipynb_file__"] = str(repo / "train.ipynb")  # as VS Code sets it
    run_cells(
        ipython,
        "lr = 0.1",
        "import hunch\nhunch.init(config={'lr': lr}, why='from a notebook')",
        "hunch.log({'val_loss': 1.0})",
        "1/0",
        "hunch.finish()",
    )
    (run,) = Store(repo).runs()
    assert run.meta["command"] == ["train.ipynb"]
    assert run.status == "finished"
    cells = read_jsonl(run.dir / "cells.jsonl")
    assert [cell["n"] for cell in cells] == [1, 2, 3, 4, 5]
    assert cells[0] == {"n": 1, "source": "lr = 0.1"}  # ran before hunch.init(), so it has no time
    assert cells[3]["error"] == "ZeroDivisionError: division by zero"
    assert all("error" not in cell for cell in cells[:3] + cells[4:])


def test_a_run_inside_one_cell_keeps_that_cell(ipython, repo, monkeypatch):
    monkeypatch.setenv("JPY_SESSION_NAME", str(repo / "nb" / "explore.ipynb"))  # as Jupyter Server sets it
    run_cells(ipython, "import hunch\nhunch.init()\nhunch.log({'loss': 1})\nhunch.finish()", "x = 1")
    (run,) = Store(repo).runs()
    assert run.meta["command"] == ["nb/explore.ipynb"]
    assert [cell["n"] for cell in read_jsonl(run.dir / "cells.jsonl")] == [1]


def test_a_second_run_keeps_the_whole_session(ipython, repo):
    run_cells(
        ipython,
        "import hunch\nhunch.init(why='first')",
        "hunch.log({'loss': 1})",
        "lr = 0.2",
        "hunch.init(why='second')",  # finishes the first run
        "hunch.finish()",
    )
    first, second = Store(repo).runs()
    assert [cell["n"] for cell in read_jsonl(first.dir / "cells.jsonl")] == [1, 2, 3, 4]
    assert [cell["n"] for cell in read_jsonl(second.dir / "cells.jsonl")] == [1, 2, 3, 4, 5]


def test_the_ipython_terminal_keeps_its_command(ipython, repo):
    run_cells(ipython, "import hunch\nhunch.init()", "hunch.finish()")
    (run,) = Store(repo).runs()
    assert run.meta["command"][0].endswith(("python", "python.exe", "python3"))
    assert len(read_jsonl(run.dir / "cells.jsonl")) == 2


def test_show_lists_the_cells_and_diff_compares_them(ipython, repo, capsys):
    run_cells(ipython, "import hunch\nhunch.init()", "lr = 0.1", "hunch.finish()")
    run_cells(ipython, "hunch.init()", "lr = 0.3", "hunch.finish()")
    first, second = Store(repo).runs()
    capsys.readouterr()
    assert hunch("show", second.id) == 0
    assert "6 cells ran in the notebook" in capsys.readouterr().out
    assert hunch("show", first.id, "--cells") == 0
    assert capsys.readouterr().out.startswith("[1]\nimport hunch\nhunch.init()\n\n[2]\nlr = 0.1\n")
    assert hunch("diff", first.id, second.id) == 0
    assert "3 -> 6 cells ran  +6 -0" in capsys.readouterr().out
    assert hunch("diff", first.id, second.id, "--code") == 0
    assert "+lr = 0.3" in capsys.readouterr().out


def test_notebooks_are_saved_without_outputs(repo):
    image = {"output_type": "display_data", "data": {"image/png": "A" * 2_000_000}, "metadata": {}}
    (repo / "explore.ipynb").write_text(notebook("print(1)", outputs=[image]))
    assert wrap("train.py") == 0
    (run,) = Store(repo).runs()
    saved = json.loads(saved_files(run)["explore.ipynb"])  # over 1 MB with its output, yet saved
    assert saved["cells"][0]["source"] == "print(1)"
    assert saved["cells"][0]["outputs"] == [] and saved["cells"][0]["execution_count"] is None
    assert "widgets" not in saved["metadata"]


def test_diff_shows_a_notebook_as_its_cells(repo, capsys):
    (repo / "explore.ipynb").write_text(notebook("lr = 0.1", "train(lr)"))
    assert wrap("train.py") == 0
    (repo / "explore.ipynb").write_text(notebook("lr = 0.3", "train(lr)"))
    assert wrap("train.py") == 0
    first, second = Store(repo).runs()
    capsys.readouterr()
    assert hunch("diff", first.id, second.id, "--code") == 0
    out = capsys.readouterr().out
    assert "-lr = 0.1\n+lr = 0.3" in out
    assert '"source"' not in out
