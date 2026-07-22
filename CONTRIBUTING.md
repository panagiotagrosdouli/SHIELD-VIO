# Contributing to SHIELD-VIO

Thank you for helping improve SHIELD-VIO. This repository is research software: contributions should make the implementation easier to inspect, reproduce, test, and evaluate without overstating the evidence.

## Before opening a pull request

1. Open or reference an issue that explains the scientific or engineering motivation.
2. Keep the scope focused. Separate estimator changes, detector changes, experiments, documentation, and infrastructure when possible.
3. State the validation level of the change: analytical, unit, synthetic, public-dataset, simulation, or hardware.
4. Do not present mocked fixtures, synthetic scenarios, or oracle degradation metadata as real-world benchmark evidence.

## Development setup

```bash
git clone https://github.com/panagiotagrosdouli/SHIELD-VIO.git
cd SHIELD-VIO
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pre-commit install
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pre-commit install
```

## Required checks

Run the focused tests for the code you changed, then run the full repository checks:

```bash
ruff check shield_vio scripts tests
black --check .
pytest -q
```

For experiment or pipeline changes, also run the relevant deterministic reproduction command and record the seed, configuration, command, runtime, and output paths.

## Research contribution standards

### Estimation and numerical methods

- Document coordinate frames, units, state ordering, and sign conventions.
- Add analytical or numerical-invariant tests for Jacobians, covariance propagation, quaternion operations, and bias handling.
- Explain any stabilization, clipping, regularization, or PSD-repair step.

### Failure detection and calibration

- Keep deployable diagnostics separate from privileged labels or degradation metadata.
- Use separate training, calibration, test, and shifted-test data.
- Report threshold-selection procedures and probability-calibration assumptions.
- Include false alarms, missed failures, calibration metrics, and warning lead time when applicable.

### Experiments

- Use deterministic seeds where possible.
- Compare methods on identical scenarios and failure definitions.
- Report aggregate statistics rather than a single favorable run.
- Preserve manifests and source artifacts needed to audit figures and tables.

### Documentation and claims

Use the strongest label supported by the evidence, not by the intended use:

- **Research prototype** — implemented and inspectable, but not deployment validated.
- **Analytical or unit validation** — checked against controlled mathematical or software cases.
- **Synthetic validation** — evaluated on generated scenarios.
- **Pending dataset execution** — adapters or procedures exist, but no public-sequence result is claimed.
- **Hardware validation required** — no physical-platform safety claim is supported.

Do not introduce production, state-of-the-art, hardware-safety, or formal-safety claims without corresponding evidence.

## Pull request checklist

- [ ] The scope and motivation are clear.
- [ ] New behavior has focused tests.
- [ ] Random seeds and experiment settings are recorded.
- [ ] Units, frames, and assumptions are documented.
- [ ] Generated artifacts are reproducible from committed code.
- [ ] Synthetic and real-world evidence are clearly distinguished.
- [ ] README or research-status documentation is updated when the evidence boundary changes.
- [ ] All required checks pass.

## Reporting security or safety concerns

Do not use a public issue for vulnerabilities or sensitive platform-safety concerns. Follow the instructions in `SECURITY.md`.
