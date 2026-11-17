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


def test_real_accuracy_audit_reports_catastrophic_drift(tmp_path: Path) -> None:
    gt = tmp_path / "gt.csv"
    gt.write_text(
        "#timestamp,x,y,z\n"
        "1,0,0,0\n"
        "2,1,0,0\n"
        "3,2,0,0\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    visual = tmp_path / "visual.json"
    baseline.write_text(json.dumps({
        "ate_rmse_m": 4.0, "rpe_translation_rmse_m": 0.4, "associated_samples": 3
    }), encoding="utf-8")
    visual.write_text(json.dumps({
        "ate_rmse_m": 0.2, "rpe_translation_rmse_m": 0.1, "associated_samples": 3
    }), encoding="utf-8")
    result = MODULE.summarize_accuracy(gt, baseline, visual)

    assert result["groundtruth_path_length_m"] == pytest.approx(2.0)
    assert result["runs"]["imu_only"]["catastrophic_drift_warning"] is True
    assert result["runs"]["opencv_rotation"]["catastrophic_drift_warning"] is False
    assert result["runs"]["opencv_rotation"]["accuracy_validated"] is False
    assert result["pipeline_evidence_only"] is True


def test_accuracy_audit_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    gt = tmp_path / "gt.csv"
    gt.write_text("1,0,0,0\n2,1,0,0\n", encoding="utf-8")
    report = tmp_path / "metrics.json"
    report.write_text(json.dumps({
        "ate_rmse_m": float("nan"), "rpe_translation_rmse_m": 0.0,
        "associated_samples": 2
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid estimator metrics"):
        MODULE.summarize_accuracy(gt, report, report)
