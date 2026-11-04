import os
import shutil
import time

import pytest
from conftest import hunch, make_run

from hunch.store import LOST_AFTER_SECONDS, HunchError, Store


def run_id(store, short):
    return store.resolve(short).id


def test_default_parent_is_the_last_ended_run_on_the_line(store):
    make_run(store, "aaaa", start=0)
    make_run(store, "bbbb", start=10)
    make_run(store, "cccc", start=20, duration=5)  # trivial
    make_run(store, "dddd", start=30, notes=[("verdict", "drop")])
    make_run(store, "eeee", start=40, line="other")
    make_run(store, "ffff", start=50, status=None)  # still running
    assert store.default_parent("main") == run_id(store, "bbbb")
    assert store.default_parent("other") == run_id(store, "eeee")
    assert store.default_parent("new-branch") is None


def test_running_and_lost_runs(store):
    folder = make_run(store, "aaaa", start=0, status=None)
    heartbeat = folder / "heartbeat"
    heartbeat.touch()
    assert store.resolve("aaaa").status == "running"
    old = time.time() - LOST_AFTER_SECONDS - 10
    os.utime(heartbeat, (old, old))
    assert store.resolve("aaaa").status == "lost"


def test_latest_note_wins_and_history_is_kept(store):
    make_run(store, "aaaa", start=0, why="first idea")
    assert hunch("note", "aaaa", "helps a little", "--keep") == 0
    assert hunch("note", "aaaa", "actually no", "--drop", "--why", "better idea") == 0
    run = store.resolve("aaaa")
    assert (run.why, run.conclusion, run.verdict) == ("better idea", "actually no", "drop")
    assert [e["kind"] for e in run.notes] == ["note", "verdict", "why", "note", "verdict"]
    assert run.meta["why"] == "first idea"  # the record itself never changes


def test_link_fixes_a_parent_and_rejects_impossible_ones(store, capsys):
    make_run(store, "aaaa", start=0)
    make_run(store, "bbbb", start=10)
    assert hunch("link", "bbbb", "--from", "aaaa") == 0
    assert store.resolve("bbbb").parent == run_id(store, "aaaa")
    assert hunch("link", "aaaa", "--from", "bbbb") == 2
    assert hunch("link", "aaaa", "--from", "aaaa") == 2
    assert hunch("link", "bbbb", "--from", "none") == 0
    assert store.resolve("bbbb").parent is None


def test_resolve_by_ending_and_ambiguity(store):
    make_run(store, "ab12", start=0)
    make_run(store, "cd12", start=10)
    assert store.resolve("ab12").id.endswith("ab12")
    with pytest.raises(HunchError, match="matches 2 runs"):
        store.resolve("12")
    with pytest.raises(HunchError, match="no run matches"):
        store.resolve("zzzz")


def test_runs_copied_from_another_machine_merge_cleanly(store, tmp_path):
    other = Store(tmp_path / "cluster")
    other.ensure()
    make_run(store, "aaaa", start=0)
    make_run(other, "bbbb", start=10)
    for folder in other.runs_dir.iterdir():
        shutil.copytree(folder, store.runs_dir / folder.name)
    assert [r.id[-4:] for r in store.runs()] == ["aaaa", "bbbb"]
    assert hunch("log") == 0


def test_main_metric_setting(store, capsys):
    make_run(store, "aaaa", start=0, metrics=[{"_step": 0, "loss": 1.0, "acc": 0.5}])
    make_run(store, "bbbb", start=10, metrics=[{"_step": 0, "train_loss": 1.0, "val_acc": 0.7}])
    assert store.resolve("aaaa").main_metric() == "acc"  # no "val" metric: the last one
    assert store.resolve("bbbb").main_metric() == "val_acc"
    assert hunch("set", "main-metric", "loss") == 0
    assert store.config()["main_metric"] == "loss"
    assert hunch("set", "main-metric", "auto") == 0
    assert "main_metric" not in store.config()
