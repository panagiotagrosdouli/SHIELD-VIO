import subprocess
import sys
from pathlib import Path

from run_orbslam3_euroc import load_run_config


def write_config(path):
    path.write_text(
        'orbslam3:\n'
        '  executable: /tmp/orbslam3\n'
        '  vocabulary: /tmp/ORBvoc.txt\n'
        '  settings: /tmp/EuRoC.yaml\n'
        '  sequence_path: /tmp/MH_01_easy\n'
        '  output_trajectory: results/test/trajectory.txt\n'
        '  timeout_seconds: 123\n'
    )


def test_load_run_config(tmp_path):
    config_file = tmp_path / 'orbslam3.yaml'
    write_config(config_file)

    config = load_run_config(config_file)

    assert config.executable == Path('/tmp/orbslam3')
    assert config.vocabulary == Path('/tmp/ORBvoc.txt')
    assert config.timeout_seconds == 123


def test_run_orbslam3_dry_run(tmp_path):
    config_file = tmp_path / 'orbslam3.yaml'
    write_config(config_file)

    result = subprocess.run(
        [
            sys.executable,
            'run_orbslam3_euroc.py',
            '--config',
            str(config_file),
            '--dry-run',
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert 'ORB-SLAM3 command' in result.stdout
    assert '/tmp/orbslam3' in result.stdout
