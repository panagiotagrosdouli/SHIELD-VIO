<div align="center">

# SHIELD-VIO

## Ενδοσκόπηση Εκτιμητή, Βαθμονομημένη Πρόβλεψη Αποτυχίας και Προστατευτική Πλοήγηση για Visual–Inertial Autonomy

**English:** [README.md](README.md) · **Ελληνικά**

</div>

<p align="center"><img src="assets/readme/shield_vio_research_overview.svg" alt="Ερευνητική αρχιτεκτονική SHIELD-VIO" width="100%" /></p>

<p align="center"><em>Εννοιολογικό research overview. Τα evidence panels του repository είναι Synthetic Validation και δεν αποτελούν hardware ή formal-safety απόδειξη.</em></p>

## Περίληψη

Το **SHIELD-VIO** είναι ένα reproducible research framework που εξετάζει πώς ένα visual–inertial estimator μπορεί να αναγνωρίζει έγκαιρα ότι η εκτίμηση κατάστασης γίνεται αναξιόπιστη, να ποσοτικοποιεί τον κίνδυνο επικείμενης αποτυχίας και να προστατεύει τη downstream πλοήγηση πριν το localization failure γίνει safety-critical.

Η αρχιτεκτονική συνδέει πραγματικό feature tracking, bias-aware IMU preintegration, 15-state error-state EKF, consistency diagnostics, failure detectors, calibration και conformal bounds, domain-shift detection και stateful navigation shield.

## Ερευνητικό ερώτημα

> Πώς μπορεί ένα visual–inertial σύστημα να ανιχνεύει εγκαίρως απώλεια αξιοπιστίας και να ενεργοποιεί κατάλληλες προστατευτικές ή recovery actions υπό sensor degradation και domain shift;

## Ερευνητική ροή

```text
camera frames → feature tracking ┐
                                 ├→ 15-state ESKF
IMU stream → preintegration ─────┘
  → covariance / innovation consistency
  → visual και inertial health
  → failure prediction
  → probability calibration / conformal bounds
  → domain-shift state
  → navigation shield
  → speed limit / hold / relocalize / halt / emergency stop
```

## Επιστημονική διατύπωση

Η nominal κατάσταση είναι:

```math
x=\{p_{WI},v_{WI},q_{WI},b_a,b_g\}
```

με local error state:

```math
\delta x=[\delta p,\delta v,\delta\theta,\delta b_a,\delta b_g]^T.
```

Η innovation consistency παρακολουθείται με:

```math
\mathrm{NIS}=\nu^TS^{-1}\nu,
```

και, όταν υπάρχει ground truth:

```math
\mathrm{NEES}=e^TP^{-1}e.
```

Ένα diagnostic score δεν αντιμετωπίζεται ως πιθανότητα πριν από calibration.

## Υλοποιημένα στοιχεία

- Shi–Tomasi και pyramidal Lucas–Kanade feature tracking,
- forward–backward rejection, persistent tracks και quality diagnostics,
- bias-aware IMU preintegration και covariance propagation,
- 15-state ESKF με Joseph-form updates,
- deterministic visual και IMU degradations,
- rule-based και logistic failure detectors,
- Brier, NLL, ECE, calibration metrics και split-conformal bounds,
- rolling domain-shift states,
- stateful shield με hysteresis, dwell behavior και emergency override.

Η shield state machine υποστηρίζει:

```text
NORMAL → WARNING → SLOW_DOWN → HOLD_POSITION
       → RELOCALIZE_REQUESTED → HALT → EMERGENCY_STOP
```

## Αναπαραγωγή

```bash
git clone https://github.com/panagiotagrosdouli/SHIELD-VIO.git
cd SHIELD-VIO
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/run_all.py
ruff check shield_vio scripts tests
black --check .
pytest -q
```

Repeated scenarios:

```bash
python scripts/run_scenario_suite.py --num-seeds 20 --output results/scenario_suite
```

## Evidence boundaries

| Layer | Evidence |
|---|---|
| Feature frontend | Research Prototype |
| IMU preintegration | Analytical Unit Validation |
| 15-state ESKF | Numerical Invariant Validation |
| Failure prediction και calibration | Experimental |
| Domain-shift detection | Experimental |
| Navigation shield | Closed-loop Unit Validation |
| Multi-seed degradations | Synthetic Validation |
| EuRoC / TUM-VI | Pending Dataset Execution |
| ROS 2 / hardware | Validation Required |

## Περιορισμοί

- Δεν έχει ολοκληρωθεί production-quality integrated VIO backend.
- Τα public-dataset experiments παραμένουν pending.
- Η logistic detector δεν έχει benchmark σε πραγματικά failure data.
- Conformal coverage εξαρτάται από calibration assumptions.
- Loop closure, relocalization, ROS 2, simulation και hardware execution είναι ελλιπή.
- Η navigation shield είναι supervisory research logic και όχι formally verified controller.

## Υπεύθυνη χρήση

Το SHIELD-VIO δεν πρέπει να χρησιμοποιείται ως αυτόνομος μηχανισμός ασφάλειας πραγματικού οχήματος ή ρομπότ χωρίς ανεξάρτητη επικύρωση, fail-safe control και domain-specific certification.
