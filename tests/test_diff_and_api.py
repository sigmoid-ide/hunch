import json
import math

from conftest import hunch, wrap

from hunch.api import Run
from hunch.store import Store, read_jsonl


def test_diff_shows_code_config_and_metric_changes(repo, capsys):
    assert wrap("train.py --lr 0.1") == 0
    (repo / "train.py").write_text((repo / "train.py").read_text().replace("2.0 /", "1.5 /"))
    assert wrap("train.py --lr 0.3") == 0
    first, second = Store(repo).runs()
    capsys.readouterr()
    assert hunch("diff", first.id, second.id) == 0
    out = capsys.readouterr().out
    assert "train.py  +1 -1" in out
    assert "lr: 0.1 -> 0.3" in out
    assert "val_loss 0.4 -> 0.3 (-0.1)" in out
    assert hunch("diff", first.id, second.id, "--code") == 0
    assert "+    hunch.log({\"loss\": 1.0 / (step + 1), \"val_loss\": 1.5 / (step + 1)})" in capsys.readouterr().out


def test_diff_of_identical_code_reports_no_changes(repo, capsys):
    assert wrap("train.py") == 0
    assert wrap("train.py") == 0
    first, second = Store(repo).runs()
    capsys.readouterr()
    assert hunch("diff", first.id, second.id) == 0
    assert "code     no changes" in capsys.readouterr().out


def test_config_works_like_a_dict_and_an_object_and_saves_itself(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"args": ["python"], "values": {}}))
    run = Run(tmp_path, owned=False, config={"lr": 0.1})
    run.config.batch = 32
    run.config.update({"lr": 0.2}, allow_val_change=True)
    assert run.config.lr == 0.2 and run.config["batch"] == 32
    assert json.loads((tmp_path / "config.json").read_text())["values"] == {"lr": 0.2, "batch": 32}


def test_log_turns_tensor_like_values_into_numbers(tmp_path):
    class FakeTensor:
        def item(self):
            return 1.5

    run = Run(tmp_path, owned=False)
    run.log({"loss": FakeTensor(), "nan": float("nan")})
    run.log({"loss": 1.0}, step=10)
    run.log({"loss": 0.5})
    rows = read_jsonl(tmp_path / "metrics.jsonl")
    assert rows[0]["loss"] == 1.5 and math.isnan(rows[0]["nan"])
    assert [row["_step"] for row in rows] == [0, 10, 11]


def test_a_run_without_a_folder_records_nothing(tmp_path):
    run = Run(None, owned=False)
    run.log({"loss": 1.0})
    run.config.lr = 0.1
    run.finish()
    assert list(tmp_path.iterdir()) == []
