"""Edge cases found in review: distributed ranks, odd text, user git settings,
leftover background processes and other ways a run could be lost or wrong."""

import copy
import json
import os
import pickle
import subprocess
import sys
import time

import pytest
from conftest import git, hunch, make_run, wrap
from test_run import saved_files

from hunch import api, runner
from hunch.api import Run
from hunch.store import Store, read_json, write_json


def only_run(root):
    runs = Store(root).runs()
    assert len(runs) == 1
    return runs[0]


def test_other_ranks_keep_their_config_but_record_nothing(repo, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    try:
        run = api.init(config={"lr": 0.1})
        assert run.dir is None
        assert run.config.lr == 0.1
    finally:
        api._active = None
    assert Store(repo).runs() == []


def test_mpi_ranks_are_respected_but_independent_srun_tasks_record(repo):
    def train(**env):
        subprocess.run([sys.executable, "train.py"], cwd=repo, check=True, capture_output=True, env={**os.environ, **env})

    train(OMPI_COMM_WORLD_RANK="1")
    assert Store(repo).runs() == []
    train(SLURM_PROCID="2")  # a sweep worker started by srun, outside hunch run
    assert len(Store(repo).runs()) == 1


def test_inside_hunch_run_only_slurm_task_zero_joins(repo, monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "1")
    assert wrap("train.py") == 0
    assert only_run(repo).rows() == []


def test_a_note_with_a_unicode_line_separator_survives(store):
    make_run(store, "aaaa", start=0)
    assert hunch("note", "aaaa", "pasted from a pdf") == 0
    assert hunch("note", "aaaa", "second") == 0
    assert [e["value"] for e in store.resolve("aaaa").notes] == ["pasted from a pdf", "second"]


def test_a_note_can_come_before_or_after_the_verdict(store):
    # Python 3.11's argparse fills the optional text before it reaches --keep; later versions do not.
    make_run(store, "aaaa", start=0)
    assert hunch("note", "aaaa", "--keep", "after the flag") == 0
    assert hunch("note", "aaaa", "before the flag", "--drop") == 0
    assert hunch("note", "aaaa", "--keep") == 0
    run = store.resolve("aaaa")
    assert [(e["kind"], e["value"]) for e in run.notes] == [
        ("note", "after the flag"), ("verdict", "keep"), ("note", "before the flag"), ("verdict", "drop"), ("verdict", "keep")]
    with pytest.raises(SystemExit):
        hunch("note", "aaaa", "--keep", "one", "two")  # more than one text is still refused


def test_diff_ignores_the_users_git_settings(repo, capsys):
    git(repo, "config", "diff.noprefix", "true")
    assert wrap("train.py") == 0
    (repo / "train.py").write_text((repo / "train.py").read_text().replace("2.0 /", "1.5 /"))
    assert wrap("train.py") == 0
    first, second = Store(repo).runs()
    capsys.readouterr()
    assert hunch("diff", first.id, second.id) == 0
    assert "train.py  +1 -1" in capsys.readouterr().out


def test_before_the_first_commit_the_working_tree_is_recorded(tmp_path, monkeypatch):
    path = tmp_path / "fresh"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    (path / "cfg.py").write_text("lr = 0.1\n")
    git(path, "add", "cfg.py")
    (path / "cfg.py").write_bytes(b"lr = 0.9\n")  # edited after staging
    monkeypatch.chdir(path)
    assert hunch("run", "-m", "first", "--", sys.executable, "-c", "pass") == 0
    assert saved_files(only_run(path))["cfg.py"] == b"lr = 0.9\n"


@pytest.mark.skipif(os.name == "nt", reason="creating symlinks needs extra rights on Windows")
def test_symlinks_are_not_followed(repo, tmp_path):
    secret = tmp_path / "credentials"
    secret.write_text("token")
    (repo / "creds").symlink_to(secret)
    assert wrap("train.py") == 0
    run = only_run(repo)
    assert "creds" not in saved_files(run)
    assert "creds" in json.loads((run.dir / "files.json").read_text())["skipped"]


def test_text_that_is_not_valid_unicode_is_still_written(tmp_path):
    # On Linux, a non-UTF-8 file name in the arguments arrives as a lone surrogate.
    path = tmp_path / "run.json"
    write_json(path, {"command": ["python", "train.py", "caf\udce9.csv"]})
    assert read_json(path)["command"][2] == "caf\udce9.csv"


def test_config_survives_pickle_and_deepcopy(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"args": [], "values": {}}))
    run = Run(tmp_path, owned=False, config={"lr": 0.1})
    assert pickle.loads(pickle.dumps(run.config)) == {"lr": 0.1}
    assert copy.deepcopy(run.config) == {"lr": 0.1}


def test_summary_cannot_change_after_the_run_finishes(tmp_path):
    run = Run(tmp_path, owned=False)
    run.finish()
    with pytest.raises(RuntimeError):
        run.summary["acc"] = 0.9
    assert not (tmp_path / "summary.json").exists()


def test_the_store_never_leaves_the_repository(tmp_path, monkeypatch):
    home = tmp_path / "home"
    Store(home).ensure()  # a stray .hunch above the project, such as in the home folder
    project = home / "project"
    project.mkdir()
    git(project, "init", "-q", "-b", "main")
    monkeypatch.chdir(project)
    assert Store.find().root == project.resolve()


def test_a_wrapped_python_can_print_any_character(repo, monkeypatch):
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    assert hunch("run", "-m", "unicode", "--", sys.executable, "-c", "print('done ✓ тест')") == 0
    run = only_run(repo)
    assert run.status == "finished"
    assert "✓ тест" in (run.dir / "output.log").read_text(encoding="utf-8")


def test_a_leftover_background_process_does_not_hold_up_the_wrapper(repo, monkeypatch):
    monkeypatch.setattr(runner, "DRAIN_SECONDS", 1)
    code = "import subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(12)']); print('main done')"
    begin = time.monotonic()
    assert hunch("run", "-m", "background helper", "--", sys.executable, "-c", code) == 0
    assert time.monotonic() - begin < 9
    assert only_run(repo).end["duration"] < 5


def test_errors_writing_the_output_log_never_reach_the_command(tmp_path):
    class FullDisk:
        def write(self, chunk):
            raise OSError(28, "No space left on device")

        def flush(self):
            pass

        def close(self):
            pass

    log = runner._Log(tmp_path / "output.log")
    log._file.close()
    log._file = FullDisk()
    log.write(b"step 1")
    log.write(b"step 2")
    assert log._file is None


@pytest.mark.skipif(os.name != "nt", reason="Windows status codes")
def test_large_windows_exit_codes_pass_through(repo):
    ctrl_c = -1073741510  # 0xC000013A as a signed 32-bit number
    assert hunch("run", "-m", "ctrl c", "--", sys.executable, "-c", f"import os; os._exit({ctrl_c})") == ctrl_c
    assert only_run(repo).status == "crashed"
