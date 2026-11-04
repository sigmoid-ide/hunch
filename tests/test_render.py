from conftest import hunch, make_run

from hunch.render import RAMP, View, log_lines, sparkline, visible


def lines(store, show_all=False):
    runs = store.runs()
    return log_lines(visible(runs, show_all), View(runs))


def test_log_draws_the_shape_mock(store):
    base = make_run(store, "3d1c", start=0, why="baseline").name
    warm = make_run(store, "9b40", start=60, parent=base, why="warmup 1k steps").name
    make_run(store, "c21e", start=120, parent=warm, why="warmup 2k", notes=[("verdict", "drop")])
    make_run(store, "a7f3", start=180, parent=warm, why="warmup + cosine", notes=[("verdict", "keep")])
    out = lines(store)
    assert [line[:3].rstrip() for line in out] == ["*", "| x", "|/", "*", "*"]
    assert ["a7f3" in out[0], "c21e" in out[1], "9b40" in out[3], "3d1c" in out[4]] == [True] * 4


def test_log_joins_several_siblings(store):
    base = make_run(store, "base", start=0).name
    for i, short in enumerate(["sib1", "sib2", "sib3"]):
        make_run(store, short, start=10 + i, parent=base)
    out = lines(store)
    assert [line[:5].rstrip() for line in out] == ["*", "| *", "| | *", "| |/", "|/", "*"]


def test_unrelated_trees_are_separated(store):
    make_run(store, "aaaa", start=0)
    make_run(store, "bbbb", start=10, line="other")
    out = lines(store)
    assert out[1] == ""
    assert "bbbb" in out[0] and "aaaa" in out[2]


def test_trivial_runs_are_hidden_unless_something_builds_on_them(store, capsys):
    quick = make_run(store, "quik", start=0, duration=3).name
    make_run(store, "typo", start=5, duration=2)
    make_run(store, "real", start=10, parent=quick)
    shown = {r.id[-4:] for r in visible(store.runs(), show_all=False)}
    assert shown == {"quik", "real"}
    assert hunch("log") == 0
    assert "1 trivial run hidden" in capsys.readouterr().out


def test_log_shows_metric_verdict_and_missing_conclusions(store):
    rows = [{"_step": i, "val_loss": 3.0 - i} for i in range(3)]
    make_run(store, "aaaa", start=0, why="baseline", metrics=rows)
    make_run(store, "bbbb", start=10, why="crashes", status="crashed", metrics=rows)
    out = "\n".join(lines(store))
    assert "val_loss 1" in out
    assert "(no conclusion yet)" in out
    assert "crashed" in out


def test_sparkline():
    assert sparkline([1, 2, 3]) == RAMP[0] + RAMP[4] + RAMP[7]
    assert sparkline([5, 5]) == RAMP[4] * 2
    assert sparkline([1, float("nan"), 3], unicode=False) == "_!#"
    assert sparkline([]) == ""
    assert len(sparkline(list(range(100)))) == 8


def test_show_and_oneline_log_run_without_errors(store, capsys):
    make_run(store, "aaaa", start=0, why="baseline", metrics=[{"_step": 0, "val_loss": 1.0}])
    assert hunch("log", "--oneline") == 0
    assert hunch("show", "aaaa") == 0
    out = capsys.readouterr().out
    assert "baseline" in out and "val_loss 1" in out
