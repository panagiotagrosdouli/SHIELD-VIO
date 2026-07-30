# Trajectory metrics

SHIELD-VIO evaluates metric position trajectories with two complementary error measures:

- **Absolute Trajectory Error (ATE)** measures the Euclidean position error at every associated timestamp.
- **Relative Pose Error (RPE)** measures displacement error over a configurable number of samples and exposes local drift.

Before evaluation, the estimated trajectory may be rigidly aligned to the reference using an SE(3) transform estimated with the Kabsch algorithm. No scale factor is fitted. This is intentional: metric VIO evaluation should not hide scale error.

```python
from shield_vio.evaluation.trajectory_metrics import summarize_trajectory_metrics

metrics = summarize_trajectory_metrics(
    estimated_positions,
    reference_positions,
    rpe_delta=10,
    align=True,
)

print(metrics.ate_rmse_m)
print(metrics.rpe_rmse_m)
```

For EuRoC, first associate estimator samples with ground truth using `associate_ground_truth`, then pass the associated position arrays to `summarize_trajectory_metrics`.
