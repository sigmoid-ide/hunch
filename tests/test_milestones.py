"""Milestones: entries you save yourself, each covering the runs since the previous one."""

from conftest import hunch, make_run

from hunch import milestones
from hunch.store import Store, write_json


def with_config(folder, **values):
    write_json(folder / "config.json", {"args": ["python", "train.py"], "values": values})
    return folder


def loss(value):
    return [{"_step": 0, "val_loss": value * 2}, {"_step": 1, "val_loss": value}]


def test_each_milestone_covers_the_runs_since_the_previous_one(store):
    a = make_run(store, "aaaa", start=0, line=None).name
    b = make_run(store, "bbbb", start=10, line=None).name
    first = milestones.create(store, "first")
    assert first.runs == [a, b]
    c = make_run(store, "cccc", start=20, line=None).name
    second = milestones.create(store, "second")
    assert second.runs == [c]
    assert second.meta["previous"] == first.id


def test_runs_still_in_progress_wait_for_the_next_milestone(store):
    make_run(store, "done", start=0, line=None)
    running = make_run(store, "busy", start=10, line=None, status=None)
    (running / "heartbeat").touch()
    assert [r[-4:] for r in milestones.create(store, "now").runs] == ["done"]


def test_the_summary_counts_runs_names_what_varied_and_the_best(store):
    with_config(make_run(store, "aaaa", start=0, line=None, metrics=loss(2.0)), lr=0.1, batch=256)
    with_config(make_run(store, "bbbb", start=10, line=None, metrics=loss(1.0), notes=[("verdict", "keep")]), lr=0.3, batch=256)
    with_config(make_run(store, "cccc", start=20, line=None, metrics=loss(3.0), notes=[("verdict", "drop")]), lr=0.5, batch=256)
    milestone = milestones.create(store, "higher lr helps", "but not too high")
    assert milestone.summary == "3 runs (1 kept, 1 dropped). Varied: lr 0.1, 0.3, 0.5. Best val_loss 1 (bbbb)."
    assert milestone.best.endswith("bbbb")
    assert milestone.description == "but not too high"


def test_the_best_run_follows_the_metric_direction(store):
    make_run(store, "low1", start=0, line=None, metrics=[{"_step": 0, "val_acc": 0.7}])
    make_run(store, "high", start=10, line=None, metrics=[{"_step": 0, "val_acc": 0.9}])
    assert milestones.create(store, "accuracy").best.endswith("high")
    make_run(store, "odd1", start=20, line=None, metrics=[{"_step": 0, "weird": 1.0}])
    assert milestones.create(store, "unknown direction").best is None  # no guess for a metric we can't read


def test_the_best_run_can_be_chosen(store):
    make_run(store, "aaaa", start=0, line=None, metrics=loss(1.0))
    make_run(store, "bbbb", start=10, line=None, metrics=loss(2.0))
    assert hunch("milestone", "cleaner code", "--best", "bbbb") == 0
    assert milestones.milestones(store)[0].best.endswith("bbbb")


def test_milestones_only_cover_the_current_branch(repo):
    store = Store(repo)
    store.ensure()
    main = make_run(store, "main", start=0, line="main").name
    make_run(store, "side", start=10, line="other")
    assert milestones.create(store, "on main").runs == [main]
    assert milestones.create(store, "on main").meta["line"] == "main"


def test_log_shows_milestones_with_their_runs_folded(store, capsys):
    make_run(store, "aaaa", start=0, line=None, why="baseline")
    make_run(store, "bbbb", start=10, line=None, why="warmup")
    assert hunch("milestone", "warmup helps", "-m", "the spike is gone") == 0
    make_run(store, "cccc", start=20, line=None, why="cosine")
    capsys.readouterr()

    assert hunch("log") == 0
    out = capsys.readouterr().out
    assert "since the last milestone: 1 run" in out
    assert "M1" in out and "warmup helps" in out and '"the spike is gone"' in out
    assert "2 runs" in out and "(hunch log M1)" in out
    assert "baseline" not in out  # folded

    assert hunch("log", "M1") == 0
    out = capsys.readouterr().out
    assert "baseline" in out and "warmup" in out and "cosine" not in out

    for flag in ("--runs", "--all", "--oneline"):
        assert hunch("log", flag) == 0
        assert "cosine" in capsys.readouterr().out


def test_show_a_milestone(store, capsys):
    make_run(store, "aaaa", start=0, line=None, why="baseline", metrics=loss(1.0))
    assert hunch("milestone", "first result") == 0
    capsys.readouterr()
    assert hunch("show", "M1") == 0
    out = capsys.readouterr().out
    assert "first result" in out and "1 run" in out and "baseline" in out
    assert hunch("show", "M2") == 2


def test_a_milestone_needs_a_title(store, capsys):
    assert hunch("milestone", "  ") == 2
    assert "needs a title" in capsys.readouterr().err
