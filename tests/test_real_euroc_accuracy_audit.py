from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_real_euroc_accuracy.py"
SPEC = importlib.util.spec_from_file_location("audit_real_euroc_accuracy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_metrics(path: Path, ate: float) -> None:
    path.write_text(
        json.dumps(
            {
                "ate_rmse_m": ate,
                "rpe_translation_rmse_m": 0.1,
                "associated_samples": 3,
            }
        ),
        encoding="utf-8",
    )


def test_accuracy_audit_flags_ate_at_path_length(tmp_path: Path) -> None:
    groundtruth = tmp_path / "groundtruth.csv"
    groundtruth.write_text(
        "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n"
        "1,0,0,0,1,0,0,0\n"
        "2,1,0,0,1,0,0,0\n"
        "3,2,0,0,1,0,0,0\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    visual = tmp_path / "visual.json"
    _write_metrics(baseline, 2.0)
    _write_metrics(visual, 0.2)

    result = MODULE.summarize_accuracy(groundtruth, baseline, visual)

    assert result["groundtruth_path_length_m"] == pytest.approx(2.0)
    assert result["runs"]["imu_only"]["catastrophic_drift_warning"] is True
    assert result["runs"]["opencv_rotation"]["catastrophic_drift_warning"] is False
    assert result["runs"]["opencv_rotation"]["accuracy_validated"] is False
    assert result["pipeline_evidence_only"] is True


def test_accuracy_audit_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    groundtruth = tmp_path / "groundtruth.csv"
    groundtruth.write_text("1,0,0,0\n2,1,0,0\n", encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "ate_rmse_m": float("nan"),
                "rpe_translation_rmse_m": 0.0,
                "associated_samples": 2,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid estimator metrics"):
        MODULE.summarize_accuracy(groundtruth, metrics, metrics)


def test_accuracy_audit_rejects_zero_groundtruth_motion(tmp_path: Path) -> None:
    groundtruth = tmp_path / "groundtruth.csv"
    groundtruth.write_text("1,0,0,0\n2,0,0,0\n", encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    _write_metrics(metrics, 0.0)

    with pytest.raises(ValueError, match="path length"):
        MODULE.summarize_accuracy(groundtruth, metrics, metrics)
