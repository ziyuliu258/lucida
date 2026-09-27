import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_overfit_summary.py"
SPEC = importlib.util.spec_from_file_location("plot_overfit_summary", SCRIPT)
assert SPEC and SPEC.loader
plot_summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_summary)


def write_train(root: Path, rows: list[dict[str, float]]) -> None:
    with (root / "train_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def write_eval(root: Path, step: int, success: float) -> None:
    directory = root / f"eval-{step}"
    directory.mkdir()
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "closed_loop_success"])
        writer.writeheader()
        writer.writerow({"step": step, "closed_loop_success": success})


def test_combined_plot_offsets_local_steps_and_marks_weighted_phase(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    weighted = tmp_path / "weighted"
    clean.mkdir()
    weighted.mkdir()
    write_train(clean, [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.0}])
    write_eval(clean, 2, 0.4)
    write_train(weighted, [{"step": 1, "loss": 0.7}, {"step": 2, "loss": 0.2}])
    write_eval(weighted, 2, 0.9)
    (weighted / "resolved_config.json").write_text(
        json.dumps({"action_weighting": {"enabled": True}})
    )

    phases = plot_summary.load_phases([("clean", clean), ("weighted", weighted)])

    assert [row["cumulative_step"] for row in phases[0]["train"]] == [1.0, 2.0]
    assert [row["cumulative_step"] for row in phases[1]["train"]] == [3.0, 4.0]
    assert phases[1]["metrics"][0]["cumulative_step"] == 4.0
    assert phases[1]["weighted"] is True

    output = tmp_path / "overfit_report.png"
    plot_summary.plot_summary(phases, output, smooth_window=2)
    assert output.is_file()
    assert output.stat().st_size > 0
