# README evidence pipeline

SHIELD-VIO distinguishes explanatory diagrams from figures generated from executable experiment artifacts.

## Inputs

The evidence-panel generator consumes the following files from a completed synthetic run:

- `estimated_trajectory.csv` with `t`, `x`, and `y`;
- `ground_truth.csv` with `t`, `x`, and `y`;
- `uncertainty.csv` with `t`, `trace`, and `nis`;
- `risk_score.csv` with `t` and `risk_score`.

The generator fails explicitly when an input file, required column, or data row is missing.

## Local reproduction

```bash
python scripts/run_synthetic_demo.py --out results/synthetic_demo --seed 7
python scripts/generate_readme_evidence.py \
  --results results/synthetic_demo \
  --output assets/readme/evidence
```

Expected outputs:

```text
assets/readme/evidence/trajectory_evidence.png
assets/readme/evidence/estimator_health_evidence.png
```

## CI evidence

The CI workflow executes the same deterministic seed-7 run and uploads:

```text
results/readme_evidence/*.png
results/synthetic_demo/*.csv
```

as the workflow artifact `shield-vio-readme-evidence`.

The uploaded CSV files make each panel auditable against its source values. Artifact retention is intentionally finite; publication-quality or release evidence should be attached to a tagged release or archived separately with its configuration and Git commit.

## Evidence classification

These figures are classified as **Synthetic Validation**. They visualize executable repository outputs, but they do not establish:

- public-dataset performance;
- production-quality VIO accuracy;
- calibrated real-world failure probabilities;
- ROS 2 or simulator validation;
- hardware safety;
- formal safety guarantees.

The generated PNGs should only be committed into the main README after the corresponding source run, seed, configuration, and commit are recorded. Explanatory SVG diagrams remain documentation and must not be presented as experimental evidence.
