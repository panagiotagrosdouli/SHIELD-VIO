from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.compare_real_euroc_runs import build_report


def _write_metrics(path: Path, *, ate: float, rpe: float, samples: int = 10) -> None:
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "sequence": "MH_01_easy",
                "associated_samples": samples,
                "alignment": "se3",
                "ate_rmse_m": ate,
                "rpe_translation_rmse_m": rpe,
            }
        ),
        encoding="utf-8",
    )


def test_builds_paired_real_report(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    visual = tmp_path / "visual"
    _write_metrics(baseline, ate=2.0, rpe=0.5)
    _write_metrics(visual, ate=1.5, rpe=0.4)
    with (visual / "visual_updates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["inlier_ratio", "innovation_nis"])
        writer.writeheader()
        writer.writerows(
            [
                {"inlier_ratio": "0.8", "innovation_nis": "2.0"},
                {"inlier_ratio": "0.6", "innovation_nis": "4.0"},
            ]
        )

    report = build_report(baseline, visual)

    assert report["delta"]["ate_rmse_m"] == pytest.approx(-0.5)
    assert report["delta"]["ate_relative_change"] == pytest.approx(-0.25)
    assert report["opencv_rotation"]["visual_updates"] == 2
    assert report["opencv_rotation"]["mean_inlier_ratio"] == pytest.approx(0.7)
    assert report["opencv_rotation"]["median_innovation_nis"] == pytest.approx(3.0)


def test_rejects_unpaired_runs(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    visual = tmp_path / "visual"
    _write_metrics(baseline, ate=2.0, rpe=0.5, samples=10)
    _write_metrics(visual, ate=1.5, rpe=0.4, samples=9)

    with pytest.raises(ValueError, match="associated_samples"):
        build_report(baseline, visual)
