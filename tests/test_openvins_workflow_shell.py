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
        if step.get("name") == "Build and run pinned OpenVINS serial estimator without ground truth"
    ]
    assert len(matches) == 1, "expected exactly one pinned build-and-run step"
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


def test_estimator_runs_in_same_container_as_dependency_installation() -> None:
    """Cached catkin outputs do not preserve apt libraries across containers."""
    script = _runtime_command()
    assert script.count("docker run --rm") == 1
    assert script.index("rosdep install") < script.index("catkin_make")
    assert script.index("catkin_make") < script.index("rosrun ov_msckf ros1_serial_msckf")
    assert "source \"$ws/devel/setup.bash\"" in script
    assert "ldd \"$executable\"" in script
    assert "set +u" in script
    assert script.index("set +u") < script.index("source \"$ws/devel/setup.bash\"")
    assert script.index("source \"$ws/devel/setup.bash\"") < script.index("set -u")
    assert "apt-get update" in script


def test_estimator_exports_validated_separately() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["mh01-openvins"]["steps"]
    checks = [
        step["run"] for step in steps
        if step.get("name") == "Validate pinned OpenVINS estimator exports"
    ]
    assert len(checks) == 1
    assert "test -s \"$out/state_estimate.txt\"" in checks[0]
    assert "test -s \"$out/state_deviation.txt\"" in checks[0]


def test_runtime_failure_keeps_diagnostics_without_uploading_raw_dataset() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["mh01-openvins"]["steps"]
    runtime = _runtime_command()
    assert "2>&1 | tee results/openvins/MH_01_easy/raw/runtime.log" in runtime
    assert "set -euo pipefail" in runtime, "pipe failures must propagate to CI"
    failures = [
        step for step in steps if step.get("name") == "Upload OpenVINS failure diagnostics"
    ]
    assert len(failures) == 1
    upload = failures[0]
    assert upload["if"] == "failure()"
    files = upload["with"]["path"]
    assert "raw/runtime.log" in files
    assert "euroc_input_audit.json" in files
    assert "MH_01_easy.bag" not in files
    assert "mav0/cam0/data" not in files
