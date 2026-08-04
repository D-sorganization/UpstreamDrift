# Launch Monitor Analytics API

The headless API lives in `src.shared.python.launch_monitor` and imports without
PyQt6. The shallow MLP loads scikit-learn only when requested.

## Import and Aggregate

```python
from src.shared.python.launch_monitor import LaunchMonitorProject, import_session

project = LaunchMonitorProject("Player Study")
project.add_session(import_session("trackman-session.csv"))
project.add_session(import_session("garmin-session.csv"))
shots = project.combined_shots()
project.save("player-study.lmproject")
```

Use `ImportOptions` and `ColumnMapping` to override detection, mapping, units,
sign multipliers, or measurement status. `detect_profile(headers)` returns the
selected profile, confidence, matched fingerprints, and alternatives.

## Treat and Filter

```python
from src.shared.python.launch_monitor import FilterRule, TreatmentConfig, apply_treatment

treated = apply_treatment(
    shots,
    TreatmentConfig(
        required_metrics=("club_speed", "ball_speed"),
        outlier_metrics=("club_speed", "ball_speed"),
        filters=(FilterRule("club", "eq", "7 Iron"),),
        exclude_flagged=True,
    ),
)
project.record_actions(treated.audit_log)
```

Input frames are not mutated. `TreatmentResult` contains the analysis view,
row-level flags, and serializable audit actions.

## Relationships and Multivariate Diagnostics

```python
from src.shared.python.launch_monitor import compute_correlations, compute_pca, compute_vif

metrics = ("club_speed", "ball_speed", "attack_angle", "carry_distance")
relations = compute_correlations(
    treated.data,
    metrics=metrics,
    method="spearman",
    controls=("attack_angle",),
)
pca = compute_pca(treated.data, metrics=metrics)
vif = compute_vif(treated.data, metrics=metrics)
```

`CorrelationResult` contains coefficient, raw p-value, adjusted p-value, sample
count, optional partial-correlation matrices, derived metrics, and screened
`DependencyEdge` records.

## Predictive Models

```python
from src.shared.python.launch_monitor import fit_predictive_model

model = fit_predictive_model(
    treated.data,
    target="carry_distance",
    features=("ball_speed", "launch_angle", "spin_rate"),
    model="ridge",
    group_column="session_id",
    random_seed=42,
)
```

Supported model names are `linear`, `ridge`, `lasso`, `elastic_net`, and `mlp`.
The return value includes held-out metrics, predictions/residuals, coefficients
where available, split counts, and the recipe seed.

## Agreement, Dispersion, and Trends

- `compare_monitors(...)` distinguishes matched-shot agreement from unmatched
  descriptive comparison.
- `analyze_dispersion(...)` computes robust center, covariance ellipse, area,
  and radial metrics.
- `analyze_trend(...)` computes time-based slopes, rolling/EWMA series, and
  candidate step changes.

All public functions validate required columns and minimum sample sizes with
descriptive `ValueError` messages.
