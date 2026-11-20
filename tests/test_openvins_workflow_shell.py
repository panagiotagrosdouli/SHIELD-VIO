"""Regression guards for the real EuRoC OpenVINS GitHub Actions shell commands."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/openvins-real-euroc.yml"


def _step_command(name: str) -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["mh01-openvins"]["steps"]
    matches = [step["run"] for step in steps if step.get("name") == name]
    assert len(matches) == 1, f"expected exactly one step: {name}"
    return matches[0]


def _runtime_command() -> str:
    return _step_command("Run pinned OpenVINS serial estimator without ground truth")


def test_rosdep_build_arguments_are_single_backslash_continuations() -> None:
    script = _step_command("Build pinned OpenVINS serial estimator")
    match = re.search(r"rosdep install(?:[^\n]*\n){1,6}", script)
    assert match, "build step must install ROS dependencies"
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


def test_runtime_reuses_built_image_without_second_dependency_install() -> None:
    build = _step_command("Build pinned OpenVINS serial estimator")
    runtime = _runtime_command()
    assert 'docker commit "$container" shield-vio-openvins:runtime' in build
    assert "shield-vio-openvins:runtime" in runtime
    assert "apt-get update" not in runtime
    assert "rosdep install" not in runtime
    assert "source /workspace/.cache/openvins_noetic_ws/devel/setup.bash" in runtime
    assert "ldd \"$executable\"" in runtime


def test_runtime_does_not_supply_ground_truth_to_estimator() -> None:
    script = _runtime_command()
    assert "rosrun ov_msckf ros1_serial_msckf" in script
    assert "_path_bag:" in script
    assert "_filepath_est:" in script
    assert "ground_truth" not in script
