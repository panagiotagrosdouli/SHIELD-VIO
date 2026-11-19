"""Regression guards for the real EuRoC OpenVINS GitHub Actions shell commands."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/openvins-real-euroc.yml"


def _runtime_command() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]["mh01-openvins"]["steps"]
    matches = [
        step["run"]
        for step in jobs
        if step.get("name") == "Run pinned OpenVINS serial estimator without ground truth"
    ]
    assert len(matches) == 1, "expected exactly one OpenVINS runtime step"
    return matches[0]


def test_rosdep_runtime_arguments_are_single_backslash_continuations() -> None:
    script = _runtime_command()
    match = re.search(r"rosdep install(?:[^\n]*\n){1,6}", script)
    assert match, "runtime step must install ROS dependencies"
    snippet = match.group(0)
    assert "--from-paths" in snippet
    assert "--ignore-src" in snippet
    assert "--rosdistro noetic" in snippet
    assert "-r -y" in snippet
    continuations = [line for line in snippet.splitlines() if line.rstrip().endswith("\\")]
    assert len(continuations) == 4
    assert all(not line.rstrip().endswith("\\\\") for line in continuations), (
        "double backslashes leave a literal slash instead of joining Bash lines"
    )


def test_runtime_does_not_supply_ground_truth_to_estimator() -> None:
    script = _runtime_command()
    assert "rosrun ov_msckf ros1_serial_msckf" in script
    assert "_path_bag:" in script
    assert "_filepath_est:" in script
    assert "ground_truth" not in script
