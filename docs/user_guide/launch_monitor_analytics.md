# Launch Monitor Analytics

Launch Monitor Analytics is a local PyQt6 workbench for aggregating shot data,
studying impact-parameter interdependence, comparing measurement systems, and
tracking a player's dispersion and trends over time.

Run it directly:

```powershell
python -m src.tools.launch_monitor_analytics
```

Or open **Launch Monitor Analytics** from the UpstreamDrift launcher.

## Supported Data Sources

The workbench reads CSV, TSV, XLS/XLSX, and JSON record files. It includes
header profiles for TrackMan, Foresight Sports, FlightScope, Garmin Golf,
SkyTrak, Uneekor, Full Swing, Rapsodo, and GSPro/Open Connect, plus a generic
mapping profile.

Support means that known header families can be mapped into the canonical
schema; it does not mean every vendor release, locale, report configuration, or
subscription tier has an identical export. Always review the detected profile,
column mappings, units, and sign convention in the import dialog. Unknown
columns are retained rather than discarded.

### Evidence and Current Boundaries

| Ecosystem                           | Officially Documented Capability                                                                                                                                                                                                                                           | Current Adapter Evidence                                                          |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| TrackMan TPS                        | [CSV export](https://support.trackmangolf.com/hc/en-us/articles/12985883274139-Shot-Analysis-How-To-Export-A-CSV-File-From-TPS) and [parameter definitions](https://support.trackmangolf.com/hc/en-us/articles/5089892383515-Practice-Trackman-Data-Parameter-Definitions) | Tested representative header fixture; actual versions can vary                    |
| Foresight FSX                       | [Configurable CSV export](https://golf.foresightsports.com/sites/default/files/downloads/2021/FSX_UserManual_v2-4.pdf)                                                                                                                                                     | Tested representative header fixture                                              |
| Garmin Golf                         | [Approach R10/R50 CSV export](https://support.garmin.com/en-AU/?faq=pit4ClEw6f019Cbs3Uhw59)                                                                                                                                                                                | Tested representative header fixture                                              |
| FlightScope                         | [Published club and ball parameters](https://flightscope.com/pages/flightscope-data-parameters)                                                                                                                                                                            | Tested representative header fixture; exact export layout needs versioned samples |
| Uneekor                             | Official device manuals list available ball/club metrics                                                                                                                                                                                                                   | Tested representative header fixture; exact export layout needs versioned samples |
| SkyTrak, Full Swing, Rapsodo, GSPro | Product-dependent tabular or connector data                                                                                                                                                                                                                                | Header profile plus generic mapping; verify each source before analysis           |

The repository fixtures are synthetic and vendor-shaped. They are not vendor
validation data and cannot establish measurement accuracy or equivalence.

## Workflow

### 1. Import and Review

1. Open **Sessions** and choose **Import Files...**.
2. Review the detected vendor and confidence.
3. Check every canonical mapping and source unit. Select **retain only** when a
   field should remain available without being normalized.
4. Add player, session, device model, and software-version metadata.
5. Repeat for every session or monitor.

The project retains the original source header/value, file SHA-256, source row,
profile, units, unit evidence, and warnings. Save a `.lmproject` file to preserve
all sessions and the treatment audit log.

### 2. Build an Analysis View

Use **Data Treatment** to:

- require metrics needed for the planned analysis;
- flag duplicate shot identifiers;
- flag robust outliers with a modified-Z threshold;
- add structured filters by player, club, session, monitor, time, tag, or value;
- calculate identity-derived fields such as smash factor, face-to-path, and roll
  distance only where their inputs exist;
- exclude flagged rows from the analysis view without changing imported data.

Every derivation, filter, flag, and exclusion is recorded. A flagged observation
is not automatically a bad measurement: review source context before exclusion.

### 3. Map Relationships

In **Relationships**, select two or more metrics and choose Pearson, Spearman, or
Kendall correlation. Optional controls residualize selected confounders before
partial correlation. Results include pair counts, p-values, Benjamini-Hochberg
adjustment, derived-variable markings, and screened network edges. **Run PCA and
VIF Diagnostics** to inspect latent structure and multicollinearity.

Analyze important strata separately:

- one monitor and one club;
- the same monitor across sessions;
- one session across clubs;
- each monitor separately before pooling;
- the full treated dataset only after checking composition.

Pooling can create or reverse an association when club, player, or monitor mix
changes.

### 4. Fit Predictive Models

The **Models** workspace supports linear, ridge, lasso, elastic-net, and an
optional shallow MLP. Select a grouped holdout (normally session) when repeated
shots within a session are related. The workbench reports held-out R², MAE, and
RMSE plus coefficients for linear models and actual-vs-predicted residual views.

The leakage guard rejects a target appearing directly or through a registered
identity-derived predictor. A high score shows prediction within the tested
split; it does not show causation or guarantee transport to a new player,
monitor, environment, or club.

### 4A. Run Flexible Analysis

Use **Flexible Analysis** when the question is not covered by a fixed chart.
Any numeric canonical field or retained `source::<header>` field can be selected
as the outcome or as one or more predictors. The same versioned contract is
available to headless and web clients at:

- `GET /tools/launch-monitor-analytics/capabilities`
- `POST /tools/launch-monitor-analytics/analyze`

The result includes pair-specific sample counts, Benjamini-Hochberg-adjusted
p-values, Pearson confidence intervals, OLS coefficient uncertainty, R² and
adjusted R², residual diagnostics, selected units, group-specific results, and
a deterministic SHA-256 dataset fingerprint. Choose pairwise, listwise, or
fail-on-missing behavior explicitly. Use **Group By** to retain monitor,
session, player, club, or other strata instead of silently pooling them.

The v1 fingerprint hashes ordered record content and explicit shot/session/
source-row/monitor identity fields. It deliberately ignores the transient
pandas row index, so loading the same records with a different in-memory index
does not change lineage. Unsupported analysis options are rejected by the API
schema. In the desktop panel, user-correctable selection errors appear in the
accessible status region instead of escaping the Qt click handler.

Vendor-specific `source::` fields are blocked from cross-monitor pooling because
matching header text does not establish matching measurement semantics.
Aggregate reference observations are never permitted in regression. Explicitly
enabled aggregate correlations are labeled descriptive and warn about
ecological bias.

### Public Reference Data

The source-traceable public dataset is maintained in the separate
[Launch-Monitor-Data repository](https://github.com/D-sorganization/Launch-Monitor-Data).
Its releases preserve source IDs, URLs, monitor identity, environment,
measurement status, reported and canonical units, aggregation level, and
verification checks. Keep that repository as the immutable evidence layer;
import shot-level exports into `.lmproject` files for regression. Published
aggregate means and standard deviations are reference observations, not
synthetic shots, and must not be expanded into fabricated row-level data.

### 5. Compare Monitors

For defensible bias, scale, and agreement analysis, import paired measurements
of the same shots and select their match identifier. The workbench reports mean
bias, standard deviation of differences, 95% limits of agreement, slope,
intercept, and correlation.

Without a match identifier, only distribution summaries and a standardized
effect size are produced. Those results can be confounded by different players,
clubs, environments, and session composition and must not be described as
calibration.

### 6. Analyze Dispersion and Trends

**Dispersion** accepts any forward/lateral coordinate pair and reports a robust
center, mean bias, 95% covariance ellipse, ellipse area, radial RMSE, median
radial error, and 90th-percentile radial error. Group by monitor, session, or
club to distinguish systematic aim bias from precision.

**Trends** uses actual observation time, not row number. It reports ordinary and
Theil-Sen slopes, a p-value, rolling center/variation, EWMA, early/late means,
and ranked candidate step changes. Candidate changes are prompts for review,
not automatic proof of a swing change.

## Units and Direction Conventions

Canonical numeric storage uses m/s, m, rad, rad/s, and s. The metric registry
also specifies familiar display units. A header unit overrides the profile
default; an explicit user mapping overrides both. The import manifest records
which source supplied the unit assumption.

Direction signs are preserved unless an explicit mapping multiplier changes
them. Vendors do not universally use the same left/right reference or player
handedness convention. Do not pool directional fields until the target line,
handedness, and sign convention have been checked.

## Publication Checklist

- Export the canonical analysis view and reproducibility manifest.
- Report source profiles, device/software versions, sample counts, missingness,
  filters, exclusions, grouping, random seed, and validation split.
- Mark derived and vendor-estimated variables.
- Use matched shots for agreement claims.
- Describe correlations and model results as associations or predictions, never
  as causal effects without an appropriate experiment.
