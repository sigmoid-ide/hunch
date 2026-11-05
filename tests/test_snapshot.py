"""Code snapshots: the code of every run is saved without git, once per file."""

import json
import subprocess
import sys

from conftest import TRAIN, git, hunch, wrap
from test_run import only_run, saved_files

from hunch.snapshot import take
from hunch.store import Store


def test_a_project_without_git_is_saved_and_diffed(tmp_path, monkeypatch, capsys):
    project = tmp_path / "no-git"
    project.mkdir()
    (project / "train.py").write_bytes(TRAIN.encode())
    monkeypatch.chdir(project)
    assert wrap("train.py --lr 0.1") == 0
    (project / "train.py").write_bytes(TRAIN.replace("2.0 /", "1.5 /").encode())
    assert wrap("train.py --lr 0.3") == 0
    first, second = Store(project).runs()
    assert first.meta["git"] is None
    assert saved_files(first)["train.py"] == TRAIN.encode()
    capsys.readouterr()
    assert hunch("diff", first.id, second.id) == 0
    out = capsys.readouterr().out
    assert "train.py  +1 -1" in out
    assert "lr: 0.1 -> 0.3" in out


def test_unchanged_files_are_stored_once(repo):
    assert wrap("train.py") == 0
    assert wrap("train.py") == 0
    objects = [p for p in (repo / ".hunch" / "objects").rglob("*") if p.is_file()]
    assert len(objects) == 1  # train.py, shared by both runs


def test_objects_are_git_blobs(repo):
    assert wrap("train.py") == 0
    sha = json.loads((only_run(repo).dir / "files.json").read_text())["files"]["train.py"]["sha"]
    expected = subprocess.run(["git", "hash-object", "--no-filters", "train.py"], cwd=repo,
                              capture_output=True, text=True, check=True).stdout.strip()
    assert sha == expected


def test_secrets_environments_caches_and_hunchignore_are_skipped(tmp_path):
    project = tmp_path / "project"
    for path, text in {
        "train.py": "print('hi')",
        ".env": "API_KEY=secret",
        "keys/server.pem": "-----BEGIN-----",
        "venv/pyvenv.cfg": "home = /usr/bin",
        "venv/lib/site.py": "x = 1",
        "src/__pycache__/train.cpython-311.pyc": "x",
        "data/big_table.csv": "a,b\n1,2",
        ".hunchignore": "# my data\ndata/\n",
    }.items():
        (project / path).parent.mkdir(parents=True, exist_ok=True)
        (project / path).write_text(text)
    manifest = take(project, tmp_path / "objects")
    assert set(manifest["files"]) == {"train.py", ".hunchignore"}


def test_gitignored_files_are_skipped(repo):
    (repo / ".gitignore").write_text("outputs/\n")
    (repo / "outputs").mkdir()
    (repo / "outputs" / "log.txt").write_text("ignored")
    assert wrap("train.py") == 0
    assert set(saved_files(only_run(repo))) == {"train.py", ".gitignore"}


def test_show_reports_the_snapshot(repo, capsys):
    assert wrap("train.py") == 0
    capsys.readouterr()
    assert hunch("show", only_run(repo).id) == 0
    out = capsys.readouterr().out
    assert "1 file saved" in out
    assert "on main" in out
