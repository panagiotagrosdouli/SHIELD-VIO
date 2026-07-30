from pathlib import Path

import json

from shield_vio.evaluation.euroc_batch import (
    evaluate_euroc_batch,
    write_batch_csv,
    write_batch_json,
)


def _write_sequence(root: Path, name: str) -> tuple[Path, Path]:
    sequence = root / name
    ground_truth = sequence / "mav0/state_groundtruth_estimate0"
    ground_truth.mkdir(parents=True)
    (ground_truth / "data.csv").write_text(
        "#timestamp,p_x,p_y,p_z\n"
        "1000000000,0,0,0\n"
        "1100000000,1,0,0\n"
        "1200000000,2,0,0\n"
        "1300000000,3,0,0\n",
        encoding="utf-8",
    )
    estimate = root / f"{name}.txt"
    estimate.write_text(
        "0.0 0 0 0\n0.1 1 0 0\n0.2 2 0 0\n0.3 3 0 0\n",
        encoding="utf-8",
    )
    return sequence, estimate


def test_batch_report_contains_success_and_failure(tmp_path: Path) -> None:
    sequence, estimate = _write_sequence(tmp_path, "MH_01_easy")
    missing = tmp_path / "missing_estimate.txt"

    report = evaluate_euroc_batch(
        [(sequence, estimate), (sequence, missing)], max_gap=0.1
    )

    assert report.success_count == 1
    assert report.failure_count == 1
    assert report.results[0].status == "ok"
    assert report.results[0].ate_rmse_m is not None
    assert report.results[1].status == "failed"
    assert report.results[1].error is not None


def test_batch_writers_emit_machine_readable_artifacts(tmp_path: Path) -> None:
    sequence, estimate = _write_sequence(tmp_path, "V1_01_easy")
    report = evaluate_euroc_batch([(sequence, estimate)], max_gap=0.1)
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"

    write_batch_json(report, json_path)
    write_batch_csv(report, csv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["success_count"] == 1
    assert payload["failure_count"] == 0
    assert payload["results"][0]["sequence"] == "V1_01_easy"
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith(
        "sequence,status,estimate_path"
    )
