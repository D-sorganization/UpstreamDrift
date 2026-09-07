---
title: Launch Monitor Analytics
tile_id: launch_monitor_analytics
status: complete
---

# Launch Monitor Analytics

## Purpose

Launch Monitor Analytics is a local PyQt6 workbench for launch-monitor shot
data. It harmonizes multi-vendor exports into one canonical schema, maps
impact-parameter interdependence, fits predictive models, compares measurement
systems, and tracks a player's dispersion and trends over time, keeping a full
treatment audit trail throughout.

Entry point `src/tools/launch_monitor_analytics/__main__.py`, which delegates
to `src/tools/launch_monitor_analytics/gui.py`. Run it with
`python -m src.tools.launch_monitor_analytics`, or open the tile from the
launcher. Declared capabilities: `launch_monitor_import`,
`session_aggregation`, `correlation_network`, `regression`, `neural_network`,
`monitor_comparison`, `dispersion`, `longitudinal_trends`.

## Inputs

| Input | Description | Unit |
| --- | --- | --- |
| Record files | CSV, TSV, XLS, XLSX and JSON shot exports | - |
| Vendor header profile | detected or chosen; TrackMan, Foresight, FlightScope, Garmin Golf, SkyTrak, Uneekor, Full Swing, Rapsodo, GSPro / Open Connect, plus a generic mapping | - |
| Column mapping, source unit, sign convention | confirmed per field in the import dialog | as declared per field |
| Session metadata | player, session, device model, software version | - |
| Trusted player identifier and its evidence | required for player-covariation and player grouping | - |
| Session identifier and order column | declared independently of player identity | - |
| Outlier threshold | modified-Z threshold for robust outlier flagging | dimensionless |
| Filters | player, club, session, monitor, time, tag, or value | - |
| Correlation method | Pearson, Spearman or Kendall | - |
| Model family | linear, ridge, lasso, elastic net, or an optional shallow MLP | - |
| Holdout scheme | grouped holdout, normally by session | - |
| `LAUNCH_MONITOR_DATA_ROOT` | root of an authorized private data checkout | filesystem path |
| `UPSTREAMDRIFT_LAUNCH_MONITOR_DATASET_ROOTS` | server-side JSON map of opaque aliases to authorized checkout roots | - |

Canonical metric units come from the metric registry. A retained
`source::<header>` field is labelled `source_declared` only when its unit is
explicitly supplied in the v2 context; otherwise its unit and authority are
reported as `unknown` and never silently promoted to canonical.

## Outputs

| Output | Description |
| --- | --- |
| Canonical shot records | plus the retained original header, value, source unit, unit evidence, file SHA-256, source row, profile and warnings |
| Treatment audit log | every derivation, filter, flag and exclusion, saved in a `.lmproject` file |
| Derived fields | smash factor, face-to-path, roll distance, computed only where their inputs exist |
| Correlation results | coefficient, pair count, p-value, Benjamini-Hochberg adjusted p-value, derived-variable markings, screened network edges |
| PCA and VIF diagnostics | latent structure and multicollinearity |
| Model reports | held-out R-squared, MAE, RMSE, linear coefficients, actual-versus-predicted and residual views |
| Flexible-analysis results | pair-specific sample counts, adjusted p-values, Pearson confidence intervals, OLS coefficient uncertainty, R-squared and adjusted R-squared, residual diagnostics, selected units, group-specific results |
| Dataset fingerprint | deterministic SHA-256 over ordered record content and identity fields |
| Player-covariation views | pooled, within-player, between-player, per-player, and fixed and random Fisher-z meta-analysis with Q, tau-squared and I-squared |
| Dispersion and longitudinal trend views | per the selected strata |
| Exports | source references and source-joinable backing hashes rather than copied restricted row values |

## Method

Import maps known vendor header families into the canonical schema. Support
means the header family can be mapped, not that every vendor release, locale,
report configuration or subscription tier exports identically; unknown columns
are retained rather than discarded. The repository fixtures are synthetic and
vendor-shaped and cannot establish measurement accuracy or equivalence.

Data treatment can require metrics, flag duplicate shot identifiers, flag
robust outliers with a modified-Z threshold, apply structured filters, and
exclude flagged rows from the analysis view without altering the imported data.

Relationship mapping computes Pearson, Spearman or Kendall correlations, with
optional residualization of selected confounders before partial correlation,
and applies Benjamini-Hochberg adjustment across the pair set. Modelling adds a
leakage guard that rejects a target appearing directly or through a registered
identity-derived predictor.

Player covariation separates pooled, within-player (each player mean removed),
between-player (association among player means) and per-player estimates, and
warns when the pooled and within-player directions disagree, which is an
aggregation reversal.

The same versioned contract is available headless and to web clients:
`GET /tools/launch-monitor-analytics/capabilities`,
`POST /tools/launch-monitor-analytics/analyze`,
`GET /tools/launch-monitor-analytics/contracts/v2` and
`POST /tools/launch-monitor-analytics/v2/analyze`. Contract v2 adds canonical
and display units, exact backing-record hashes, content-addressed sources,
authority commit and transformation lineage, missingness and exclusion counts,
explicit unavailable states, uncertainty methods, player-identity trust and
vendor provenance, so an exported result is auditable without copying
restricted source values.

Private corpus loading fails closed on five checks before any row is read: a
missing `_MANIFEST.json`, an unsupported `schema_version`, a declared
`total_rows` above the 300,000-row desktop cap, a row count disagreeing with
the corpus on disk, and a source set disagreeing with the partition
directories. Each refusal names the check that fired, and there is no option to
skip validation. Full-corpus dataset jobs bind an alias plus repository,
40-character commit, corpus-manifest SHA-256, deterministic Parquet-content
SHA-256 and expected row count; they return only source summaries, metric
summaries and correlations, at most 200 aggregate records per page, never shot
rows or server paths, with numeric groups below ten complete observations
suppressed.

Full workflow, evidence table and per-vendor boundaries:
[Launch Monitor Analytics user guide](../user_guide/launch_monitor_analytics.md).
Security and retention rationale for dataset jobs:
[ADR 0037](../adr/0037-immutable-launch-monitor-dataset-jobs.md).

## Limitations

- Vendor support is header-family mapping, not validation. The repository
  fixtures are synthetic; nothing here establishes measurement accuracy,
  device equivalence, firmware reproduction or device certification.
- A flagged observation is not automatically a bad measurement. Review the
  source context before excluding it.
- Pooling can create or reverse an association when the club, player or
  monitor mix changes. Analyze important strata separately before pooling.
- Vendor-specific `source::` fields are blocked from cross-monitor pooling,
  because matching header text does not establish matching measurement
  semantics.
- Aggregate reference observations are never permitted in regression;
  explicitly enabled aggregate correlations are labelled descriptive and warn
  about ecological bias.
- A high held-out score shows prediction within the tested split. It does not
  show causation, and it does not guarantee transport to a new player, monitor,
  environment or club.
- The all-pairs scan is for hypothesis generation only and warns about multiple
  comparisons. Validate any selected relationship on new or held-out sessions
  before coaching on it.
- Player grouping requires an explicitly supplied trusted identifier. Session,
  club, source, filename and row fields are rejected as player identifiers even
  when user-attested.
- Real source rows are not in this repository. They live in a private data
  authority; missing access or a missing pinned file fails closed, and there is
  no public download fallback.
- Dataset jobs are process-local and bounded. A server restart clears their
  status and results; resubmit the same immutable reference.
- Inline analysis is capped at 20,000 records; a larger corpus requires the
  dataset-job path.
- Selected-pair results contain player labels. Store and share them according
  to the source dataset privacy and usage terms.

## See Also
- [Launch Monitor Analytics user guide](../user_guide/launch_monitor_analytics.md)
- [ADR 0037: immutable launch monitor dataset jobs](../adr/0037-immutable-launch-monitor-dataset-jobs.md)
- [Analysis Tools calculation sheet](analysis_tools_api.md)
- [Project Map](../architecture/PROJECT_MAP.md)
