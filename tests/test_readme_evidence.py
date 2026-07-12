from pathlib import Path

from scripts.generate_readme_evidence import generate


def _write(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_generate_readme_evidence(tmp_path: Path) -> None:
    results = tmp_path / "results"
    output = tmp_path / "output"
    results.mkdir()
    _write(results / "estimated_trajectory.csv", "t,x,y", ["0,0,0", "1,1,0.5"])
    _write(results / "ground_truth.csv", "t,x,y", ["0,0,0", "1,1.1,0.4"])
    _write(results / "uncertainty.csv", "t,trace,nis", ["0,0.1,1", "1,0.2,2"])
    _write(results / "risk_score.csv", "t,risk_score", ["0,0.1", "1,0.6"])

    paths = generate(results, output)

    assert [path.name for path in paths] == [
        "trajectory_evidence.png",
        "estimator_health_evidence.png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
