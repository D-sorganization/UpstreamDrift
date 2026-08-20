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

## Analysis Contract V2

UpstreamDrift is the canonical Python and API authority for launch-monitor
statistical results. Contract `2.0.0` adds an evidence-bearing envelope around
the unchanged v1 numerical result:

- canonical and display units for every selected variable;
- dataset fingerprint, exact backing-record hashes, content-addressed source
  references, authority repository/commit, and versioned transformations;
- missing, non-numeric, complete, and analysis-specific exclusion counts;
- per-result `available` or `unavailable` states and an overall
  `available`/`partial`/`unavailable` state;
- confidence and multiplicity methods plus their assumptions;
- explicit player-identity trust and evidence;
- vendor, device model, software version, measurement status, and analytical
  model provenance; and
- conservative claim flags that default to descriptive comparison and never
  claim device emulation, certification, or causality.

Use the Python authority directly:

```python
from src.shared.python.launch_monitor import (
    AnalysisContextV2,
    DatasetAuthorityV2,
    FlexibleAnalysisRequest,
    analyze_variables_v2,
)

result = analyze_variables_v2(
    shots,
    FlexibleAnalysisRequest(
        outcome="carry_distance",
        predictors=("ball_speed", "launch_angle", "spin_rate"),
    ),
    context=AnalysisContextV2(
        authority=DatasetAuthorityV2(
            dataset_id="qualified-corpus",
            repository="D-sorganization/Launch-Monitor-Flight-Model-Campaign",
            commit="0123456789abcdef0123456789abcdef01234567",
        )
    ),
)
payload = result.model_dump(mode="json", exclude_none=True)
```

The HTTP surfaces are:

- `GET /tools/launch-monitor-analytics/contracts/v2` for JSON Schema;
- `POST /tools/launch-monitor-analytics/v2/analyze` for v2 results;
- `POST /tools/launch-monitor-analytics/analyze` for compatible v1 clients.

## Source-Backed Strokes Gained

Contract `launch-monitor-strokes-gained-analysis/1.0.0` is the canonical
scoring boundary. A valid request supplies every shot's start and finish lie,
context, target/hole, and distance plus an expected-strokes table conforming to
`launch-monitor-strokes-gained-baseline/2.0.0`. The table carries a source URL,
license declaration, version, and canonical SHA-256. Equivalent numeric values
and row orders produce the same hash; tampering and duplicate course states are
rejected.

The analysis interpolates only within an exact lie/context/target stratum and
never extrapolates outside table support. Its result includes the formula,
units, row and dataset hashes, interpolated benchmark points, exclusions,
sampling and optional benchmark uncertainty, and conservative claim flags.
Player, session, and club summaries require an explicit trusted identifier and
evidence. Longitudinal slopes additionally require an explicit numeric order
field and are descriptive, not causal.

The scoring HTTP surfaces are:

- `GET /tools/launch-monitor-analytics/contracts/strokes-gained/v1`;
- `POST /tools/launch-monitor-analytics/v2/strokes-gained`; and
- `POST /tools/launch-monitor-analytics/v2/outcome-proxy`.

The outcome-proxy endpoint reports target-relative radial error in yards. Its
typed claims explicitly state that it is not strokes gained and is not
source-backed. Carry/lateral dispersion must never be relabeled as SG.

The v2 response model is registered with FastAPI, so it is also present in the
application OpenAPI document. The checked-in schema is generated from the same
Pydantic authority and guarded against drift:

```powershell
python -m scripts.generate_launch_monitor_contract
```

Grouping by a player field fails closed unless `player_identity` declares a
trusted identifier column and evidence. Session, club, source filename, file
layout, and row order are never accepted as player identity. Insufficient or
rank-deficient regression is returned as an explicit unavailable result rather
than an apparently successful null result. Invalid columns, unsafe pooling, and
other request-contract violations still fail with a descriptive error.

Every canonical metric and retained numeric `source::<header>` field remains
selectable. Registry metrics carry registry-authoritative canonical/display
units. A retained source field carries a unit only when the caller declares it
in `AnalysisContextV2.source_units`; that unit is labeled `source_declared`, not
canonical. Without a declaration both units and their authority are `unknown`.
The contract never promotes an unknown source unit into an authoritative unit.

Dataset and analytical-model commits, when present, are full 40-character
lowercase hexadecimal SHAs. Each backing record either joins to a declared
content-addressed source by `source_id` or carries an explicit unlinked reason.
An undeclared `source_id` is a contract error.
