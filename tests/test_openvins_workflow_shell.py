"""Regression guards for the real EuRoC OpenVINS GitHub Actions shell commands."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/openvins-real-euroc.yml"


def _command(name: str) -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]["mh01-openvins"]["steps"]
    matches = [
        step["run"]
        for step in jobs
        if step.get("name") == name
    ]
    assert len(matches) == 1, f"expected exactly one step: {name}"
    return matches[0]


def _runtime_command() -> str:
    return _command("Run pinned OpenVINS serial estimator without ground truth")


def test_rosdep_build_arguments_are_single_backslash_continuations() -> None:
    script = _command("Build pinned OpenVINS serial estimator")
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


def test_runtime_does_not_supply_ground_truth_to_estimator() -> None:
    script = _runtime_command()
    assert "rosrun ov_msckf ros1_serial_msckf" in script
    assert "_path_bag:" in script
    assert "_filepath_est:" in script
    assert "ground_truth" not in script


def test_runtime_reuses_image_with_installed_dependencies() -> None:
    build = _command("Build pinned OpenVINS serial estimator")
    runtime = _runtime_command()
    assert 'docker commit "$container" shield-vio-openvins:runtime' in build
    assert "shield-vio-openvins:runtime" in runtime
    assert "rosdep install" not in runtime
    assert "apt-get update" not in runtime
    assert "apt-get install" not in runtime
    assert "source /workspace/.cache/openvins_noetic_ws/devel/setup.bash" in runtime
    assert 'ldd "$executable"' in runtime
    assert 'grep -q "not found"' in runtime


def test_offline_replay_applies_headless_visualization_patch() -> None:
    build = _command("Build pinned OpenVINS serial estimator")
    patch_path = Path(__file__).resolve().parents[1] / "patches/openvins/headless-disable-image-transport.patch"
    patch = patch_path.read_text(encoding="utf-8")
    assert 'git -C "$ws/src/open_vins" apply /workspace/patches/openvins/headless-disable-image-transport.patch' in build
    assert 'git -C "$ws/src/open_vins" diff --check' in build
    assert build.index("git -C") < build.index("catkin_make")
    assert "-  image_transport::ImageTransport it(*_nh);" in patch
    assert '-  it_pub_tracks = it.advertise("trackhist", 2);' in patch
    assert '-  it_pub_loop_img_depth = it.advertise("loop_depth", 2);' in patch
    assert '-  it_pub_loop_img_depth_color = it.advertise("loop_depth_colored", 2);' in patch
    runtime = _runtime_command()
    assert "disable_pub_plugins" not in runtime
    assert "__name:=ros1_serial_msckf" in runtime


def test_openvins_receipt_records_headless_patch_hash() -> None:
    receipt = _command("Create OpenVINS execution receipt")
    assert '"visualization_patch_sha256": sha("patches/openvins/headless-disable-image-transport.patch")' in receipt
