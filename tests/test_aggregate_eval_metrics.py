import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "aggregate_eval_metrics.py"
SPEC = importlib.util.spec_from_file_location("aggregate_eval_metrics", SCRIPT)
assert SPEC and SPEC.loader
aggregate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate)


def summary_row(step: int) -> dict[str, float]:
    return {
        "step": step,
        "actions": 3.0,
        "valid_action_rate": 1.0,
        "stopped": 1.0,
        "add_sb_m": 0.01,
        "object_diameter_m": 0.2,
        "add_sb_fraction": 0.05,
        "add_sb_at_0.10": 1.0,
        "add_sb_at_0.05": 0.0,
        "add_sb_at_0.01": 0.0,
        "iou_3d": 0.7,
        "rotation_error_deg": 2.0,
        "rotation_at_5deg": 1.0,
        "closed_loop_success": 0.0,
    }


def write_summary(root: Path, step: int, row: dict[str, float] | None = None) -> Path:
    directory = root / f"eval-{step}"
    directory.mkdir()
    path = directory / "summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate.METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow(row or summary_row(step))
    return path


def test_aggregate_sorts_summaries_and_writes_plot_schema(tmp_path: Path) -> None:
    write_summary(tmp_path, 1200)
    write_summary(tmp_path, 1000)

    rows = aggregate.collect_summaries(tmp_path)
    output = tmp_path / "eval_metrics.csv"
    aggregate.write_metrics(rows, output)

    with output.open(newline="") as handle:
        result = list(csv.DictReader(handle))
    assert [row["step"] for row in result] == ["1000", "1200"]
    assert tuple(result[0]) == aggregate.METRIC_COLUMNS


def test_aggregate_rejects_step_mismatch(tmp_path: Path) -> None:
    write_summary(tmp_path, 1000, summary_row(999))
    with pytest.raises(ValueError, match="does not match"):
        aggregate.collect_summaries(tmp_path)
