import json
import os
import subprocess
import sys

from conftest import hunch, wrap

from hunch.gitinfo import strip_credentials
from hunch.snapshot import read_object
from hunch.store import Store


def only_run(root):
    runs = Store(root).runs()
    assert len(runs) == 1
    return runs[0]


def saved_files(run) -> dict[str, bytes]:
    """The files in a run's code snapshot, as they were when it started."""
    objects = run.dir.parent.parent / "objects"
    files = json.loads((run.dir / "files.json").read_text())["files"]
    return {path: read_object(objects, entry["sha"]) for path, entry in files.items()}


def test_wrapper_records_code_config_env_metrics_and_output(repo):
    (repo / "train.py").write_text((repo / "train.py").read_text() + "# tweak\n")
    (repo / "notes.txt").write_text("small untracked text")
    (repo / "big.txt").write_text("x" * 200_000)
    (repo / "huge.txt").write_text("x" * 2_000_000)
    (repo / "weights.bin").write_bytes(b"\0\1\2")

    assert wrap("train.py --lr 0.2", "-m", "try a higher lr") == 0

    run = only_run(repo)
    assert run.status == "finished"
    assert run.why == "try a higher lr"
    assert run.line == "main"
    assert run.meta["source"] == "wrapper"
    assert run.meta["git"]["commit"] and run.meta["git"]["dirty"]
    files = saved_files(run)
    assert files["train.py"] == (repo / "train.py").read_bytes()  # including the uncommitted edit
    assert b"# tweak" in files["train.py"]
    assert files["notes.txt"] == b"small untracked text"
    assert len(files["big.txt"]) == 200_000
    assert set(json.loads((run.dir / "files.json").read_text())["skipped"]) == {"huge.txt", "weights.bin"}
    config = json.loads((run.dir / "config.json").read_text())
    assert config["values"]["lr"] == 0.2
    assert config["args"][1:] == ["train.py", "--lr", "0.2"]
    env = json.loads((run.dir / "env.json").read_text())
    assert env["source"] == "process"  # the training process replaced the wrapper's guess
    assert env["executable"]
    assert len(run.rows()) == 5
    assert run.final_metrics()["val_loss"] == 0.4
    assert "done training" in (run.dir / "output.log").read_text()


def test_exit_code_is_passed_through_and_marks_a_crash(repo):
    assert wrap("train.py --fail") == 3
    run = only_run(repo)
    assert run.status == "crashed"
    assert run.end["exit_code"] == 3


def test_unknown_command_is_an_error(repo, capsys):
    assert hunch("run", "--", "no-such-command-hunch") == 2
    assert "command not found" in capsys.readouterr().err


def test_init_without_the_wrapper_records_a_run(repo):
    subprocess.run([sys.executable, "train.py"], cwd=repo, check=True, capture_output=True)
    run = only_run(repo)
    assert run.meta["source"] == "init"
    assert run.status == "finished"
    assert run.meta["git"]["commit"]


def test_init_without_the_wrapper_records_a_crash(repo):
    proc = subprocess.run([sys.executable, "train.py", "--crash"], cwd=repo, capture_output=True)
    assert proc.returncode != 0
    assert only_run(repo).status == "crashed"


def test_only_rank_zero_records(repo):
    env = {**os.environ, "RANK": "1"}
    subprocess.run([sys.executable, "train.py"], cwd=repo, check=True, capture_output=True, env=env)
    assert Store(repo).runs() == []


def test_parallel_runs_do_not_collide(repo):
    command = [sys.executable, "-m", "hunch", "run", "-m", "parallel", "--", sys.executable, "train.py"]
    procs = [subprocess.Popen(command, cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(3)]
    assert [p.wait() for p in procs] == [0, 0, 0]
    runs = Store(repo).runs()
    assert len({r.id for r in runs}) == 3
    assert all(r.status == "finished" and len(r.rows()) == 5 for r in runs)


def test_the_why_can_come_from_the_environment_untouched_by_any_shell(repo, monkeypatch):
    why = "don't decay the lr; \"quoted\" & $HOME %PATH% !x"
    monkeypatch.setenv("HUNCH_WHY", why)
    assert wrap("train.py") == 0
    assert only_run(repo).why == why
    assert wrap("train.py", "-m", "explicit wins") == 0
    assert max(Store.find(repo).runs(), key=lambda r: r.started).why == "explicit wins"


def test_an_empty_why_in_the_environment_records_none_without_asking(repo, monkeypatch):
    monkeypatch.setenv("HUNCH_WHY", "  ")
    assert wrap("train.py") == 0
    assert only_run(repo).why is None


def test_only_allowlisted_environment_variables_are_recorded(repo, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SECRET_TOKEN", "do-not-record")
    assert wrap("train.py") == 0
    env_vars = json.loads((only_run(repo).dir / "env.json").read_text())["env_vars"]
    assert env_vars.get("CUDA_VISIBLE_DEVICES") == "0"
    assert "SECRET_TOKEN" not in env_vars


def test_store_folder_ignores_itself(repo):
    assert wrap("train.py") == 0
    assert (repo / ".hunch" / ".gitignore").read_text() == "*\n"
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
    assert ".hunch" not in status


def test_credentials_are_stripped_from_remote_urls():
    assert strip_credentials("https://user:token@github.com/a/b.git") == "https://github.com/a/b.git"
    assert strip_credentials("git@github.com:a/b.git") == "git@github.com:a/b.git"
