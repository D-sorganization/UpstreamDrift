# Launch Monitor Analytics API

The headless API lives in `src.tools.launch_monitor_model` and imports without
PyQt6. The shallow MLP loads scikit-learn only when requested.

## Import and Aggregate

```python
from src.tools.launch_monitor_model import LaunchMonitorProject, import_session

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
from src.tools.launch_monitor_model import FilterRule, TreatmentConfig, apply_treatment

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
from src.tools.launch_monitor_model import compute_correlations, compute_pca, compute_vif

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
from src.tools.launch_monitor_model import fit_predictive_model

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

`POST /tools/launch-monitor-analytics/v2/trend` runs `analyze_trend` on inline
`records` with the desktop Trends tab's inputs: `metric`, `time_column`
(default `captured_at`) and `rolling_window` (default 10, range 3 to 500). The
response carries every `TemporalTrendResult` field, the rolling series as rows,
and the change candidates. Statistics that cannot be computed are `null`, never
`0`. An unknown column or too few observations returns 400.

`POST /tools/launch-monitor-analytics/v2/dispersion` runs `analyze_dispersion`
on inline `records` with the desktop Dispersion tab's inputs: `forward`
(default `carry_distance`), `lateral` (default `lateral_carry`), and an
optional `group_column` (`monitor_vendor`, `session_id`, or `club`; omitted or
absent from the records means the desktop tab's "(all shots)" choice). The
response carries every `DispersionResult` field per group. Statistics that
cannot be computed are `null`, never `0`. A missing column or fewer than three
complete shots in a group returns 400.

`POST /tools/launch-monitor-analytics/v2/relationships` runs
`compute_correlations` on inline `records` with the desktop Relationships tab's
inputs: at least two `metrics`, optional partial-correlation `controls` (any
control that is also a selected metric is dropped, as on the desktop),
`method` (`pearson`, `spearman` or `kendall`; default `pearson`) and
`edge_threshold` (default 0.3, range 0 to 1). The response carries every
`CorrelationResult` field: the coefficient, p-value, FDR-adjusted p-value,
pair-count and (with controls) partial-coefficient matrices as rows in
`metrics` order, the derived and boolean-projected metric names, and the
screened dependency edges. Statistics that cannot be computed are `null`, never
`0`. A column absent from the records returns 400.

`POST /tools/launch-monitor-analytics/v2/multivariate` runs `compute_pca` and
`compute_vif` on inline `records` with the same `metrics` selection the
desktop Relationships tab's multivariate action reads (at least two
columns). The response carries every `PCAResult` field (explained variance
ratio, loadings, scores, sample count) under `pca`, and every `VIFResult`
field (metric-keyed values, sample count, warning metrics at VIF >= 5) under
`vif`. An infinite VIF from perfectly collinear metrics serializes as `null`,
never `0`. An unknown metric or too few complete rows returns 400.

`POST /tools/launch-monitor-analytics/v2/comparison` runs `compare_monitors`
on inline `records` with the desktop Monitor Comparison tab's inputs: a
`metric`, an optional `match_column` (unset means the desktop's
"(unmatched)") and an optional `reference_monitor` (an empty string means
none, as on the desktop). The response carries every per-monitor summary and
every pairwise comparison field, including each pairwise `warning`; unmatched
results are descriptive, not calibration evidence. Statistics that cannot be
computed are `null`, never `0`. Fewer than two monitors, an unknown reference
monitor or a missing column returns 400.

`POST /tools/launch-monitor-analytics/v2/model` runs `fit_predictive_model` on
inline `records` with the desktop Models tab's inputs: a `target`, one or more
distinct `features`, `model` (`linear`, `ridge`, `lasso`, `elastic_net` or
`mlp`; default `linear`), `random_seed` (default 42) and an optional
`group_column` (`session_id`, `monitor_vendor` or `club`; unset means the
desktop's random split). The response carries every `PredictiveModelResult`
field, with the held-out predictions as rows. Coefficients are `null` for a
model that has none, and non-finite metrics are `null`, never `0`. An unknown
column or too few complete rows returns 400; a model whose optional dependency
is missing returns 503.

`POST /tools/launch-monitor-analytics/v2/treatment` runs `apply_treatment` on
inline `records` with the desktop Data Treatment tab's inputs:
`required_metrics` and `outlier_metrics` (lists; blank entries are dropped, as
the desktop's comma-separated fields drop them), `robust_z_threshold` (modified
Z, default 4.5, range 1 to 20), `exclude_flagged` (default false) and
structured `filters` (`column`, `operator` — `eq`, `ne`, `lt`, `le`, `gt`,
`ge`, `contains` or `in` — and the raw `value` text). The response carries
every `TreatmentResult` field: the analysis view as `data` rows, the row-level
`flags`, and the `audit_log` actions, plus `shot_count` and `flag_count`.
Missing values are `null`, never `0`, and timestamps are ISO strings. The input
records are never mutated. An unknown column returns 400.

`POST /tools/launch-monitor-analytics/v2/report` builds the Reports tab's
plain-text report for inline `records`, a `project_name` (default
`"Untitled Launch Monitor Study"`, the desktop's `clear_project` default), and
`treatment_audit_log`. The web app has no import step, so `session_count` is
the number of distinct non-null `session_id` values in `records` (0 when the
column is absent) and `import_warning_count` is always 0. The response
carries `report_text`, `project_name`, `session_count`, `shot_count`,
`canonical_metrics`, and `treatment_action_count`. A blank `project_name`
returns 400.

`POST /tools/launch-monitor-analytics/v2/export` takes the same payload and
builds the canonical CSV export and reproducibility manifest the desktop
Reports tab's `export_data`/`export_manifest` build — CSV only; Parquet export
stays desktop-only. The response carries `csv` (the CSV text, with the
leading `# export_id=... exported_at=...` comment row), `data_export` (the
export's id, timestamp, file name, and SHA-256 of the CSV bytes), and
`manifest` (the reproducibility manifest, with `sessions` always `[]` since
the web app has no imported-session manifests).

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
from src.tools.launch_monitor_model import (
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

## Player Covariation and Population Synthesis

Contract `launch-monitor-player-covariation/1.0.0` separates four questions
that a pooled correlation cannot answer by itself:

- the association across all pairwise-complete rows;
- the association after centering each variable within player;
- the association between player means; and
- per-player estimates combined with fixed- and random-effects Fisher-z
  population summaries.

The population result reports Q, tau-squared, and I-squared heterogeneity,
contributor counts, explicit confidence methods, and assumptions. Small or
constant player groups and insufficient population evidence are typed
`unavailable`; missing, non-numeric, non-finite, and blank-player rows remain
counted rather than silently disappearing. Aggregation reversals are warned.

Every request requires an explicitly attested or externally verified player
identifier that exactly matches the grouping column. Session, club, source,
filename, and row fields remain forbidden pseudo-identities. The response also
retains selected-variable units, vendor/model provenance, dataset authority,
source references, and source-joinable backing-record hashes. Unknown source
fields may be selected, but an undeclared unit stays `unknown`.

The bounded pair scan is exploratory and deterministic. It exposes unavailable
pairs and multiplicity warnings; rankings do not establish causality or a
universal player relationship. Its HTTP surfaces are:

- `GET /tools/launch-monitor-analytics/contracts/player-covariation/v1`;
- `POST /tools/launch-monitor-analytics/v2/player-covariation`; and
- `POST /tools/launch-monitor-analytics/v2/player-covariation/scan`.

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
trusted identifier column and evidence. `PlayerIdentityV2` rejects `session`,
`session_id`, club, source, file/filename, row-order, and source-row fields even
when the caller attests them. This is a request-contract error (`422` at the
HTTP boundary), not an unavailable statistical result.

`session_identity` separately declares a session identifier, trust level, and
evidence. `order_evidence` declares an order column, whether it is a timestamp,
ordinal, or source sequence, its unit, trust level, and evidence. Missing or
untrusted session/order evidence does not block analyses that do not use it.

## Attested Longitudinal Sessions

Contract `launch-monitor-longitudinal-session/1.0.0` performs longitudinal
analysis only when player identity, session identity, and ordering have trusted
evidence. It never infers those fields from club, source layout, filename, row
position, or shot ID. Every included shot is first collapsed into an
equal-weight player/session/stratum cell. Per-player direction is then computed
from one equal-weight value per ordered session, so repeating shots cannot
reweight a player's trend.

The pooled estimate is a descriptive player-fixed-effects OLS association with
a finite-cluster-corrected sandwich covariance clustered by player. Declared
strata enter as categorical design terms and declared confounders enter as
numeric design terms. Confounder adjustment is not causal control. Fewer than
four player clusters, too few ordered sessions, rank deficiency, degenerate
clustered variance, nonconstant within-session order, missing fields, and a
corpus with no complete finite shots produce typed unavailable states. The
result always retains authority, source, transformation, and row-level backing
hashes, including when the statistical result is unavailable.

The `direction` request field records how a consumer interprets the metric; it
does not transform the reported raw-metric slope into an improvement claim.
Result claims explicitly set shot-level inference and causal improvement to
false. The HTTP surfaces are:

- `GET /tools/launch-monitor-analytics/contracts/longitudinal-sessions/v1`;
- `POST /tools/launch-monitor-analytics/v2/longitudinal-sessions`.

Insufficient or rank-deficient regression is returned as an explicit
unavailable result rather than an apparently successful null result. Invalid
columns, unsafe pooling, and other request-contract violations fail with a
descriptive error.

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

## Data-Free Consumer Conformance Bundle

Contract `launch-monitor-analytics-conformance/1.0.0` publishes one available
and one structured-unavailable synthetic case for analysis v2, player
covariation, attested longitudinal sessions, source-backed strokes gained, and
the distance/target proxy. The fixture contains no input `records` array and no
private or observed player rows. Derived synthetic row outputs and opaque
backing hashes remain only where an underlying result contract requires them.

Every scenario includes units and their authority, conservative claim flags,
separate player/session/order evidence, source references, source-joinable
backing hashes, exclusions, its result contract version, and a canonical
scenario SHA-256. The bundle SHA-256 covers every field except the hash itself;
validation fails after any content mutation.

Consumers should validate both generated artifacts:

- `docs/api/contracts/launch-monitor-conformance-bundle-v1.schema.json`;
- `docs/api/contracts/fixtures/launch-monitor-conformance-bundle-v1.golden.json`.

Regenerate them with `python -m scripts.generate_launch_monitor_contract`.
Unknown fields remain selectable, but their units are `unknown` unless the
synthetic source explicitly declares them. Source-declared units are not
canonical authority. See [ADR 0040](../adr/0040-data-free-launch-monitor-conformance-bundle.md).
