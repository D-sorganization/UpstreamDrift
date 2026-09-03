# ADR-0046 G1: Launch-Monitor Port Plan

- Status: Proposed — ratified when this document is reviewed
- Date: 2026-09-01
- Decision Makers: repo owner (approval required before any module moves)
- Related: ADR-0046, #9348 (G1), #9354 (G0 gates, 14 pinned divergences D1–D14),
  ADR-0031/0034/0035/0036/0037/0038/0039/0040, ADR-0045 (named-model precedent)

## How This Was Measured

Every number below is a command result, not an estimate. All commands run from
the repository root with `vendor/ud-tools` materialised at the pinned commit
`cc883cba`.

```sh
git submodule update --init --depth 1 vendor/ud-tools

# UD module count and LOC (30 modules, 7369 lines)
wc -l src/shared/python/launch_monitor/*.py | sort -k1 -n

# Tools launch-monitor Python surface (18 modules, 4240 lines)
ls vendor/ud-tools/src/rate_of_closure/ \
  | grep -E '^_?launch_monitor.*\.py$|^_?player_covariation.*\.py$' | wc -l

# Tools TypeScript model twins (18 non-test files)
git -C vendor/ud-tools ls-files \
  | grep -E 'web/src/model/launchMonitor.*\.ts$' | grep -v test | wc -l

# consumer seam: 16 facade imports vs 10 direct submodule imports
grep -rn 'from src.shared.python.launch_monitor import' --include=*.py src/ tests/ scripts/ \
  | grep -v '^src/shared/python/launch_monitor/' | wc -l
grep -rn 'from src.shared.python.launch_monitor\.' --include=*.py src/ tests/ scripts/ \
  | grep -v '^src/shared/python/launch_monitor/' | wc -l
```

Per-module consumers were resolved with an AST pass, not a text grep: the
package `__init__.py` is a re-export facade, so a plain grep attributes a
consumer to the facade rather than to the module that owns the symbol. The pass
builds a symbol-to-module map from the facade's `ImportFrom` nodes, then walks
every file outside the package that mentions `launch_monitor` and attributes
each imported name back to its defining module. Intra-package dependencies came
from the same walk.

Tools counterparts were established by reading both implementations, not by
matching names. Where the plan says "no counterpart" it means a search of
`vendor/ud-tools/src/**.py` for the capability's identifying symbols returned
nothing.

## Inventory Summary

| Classification   | Modules |   LOC | Meaning                                                   |
| ---------------- | ------: | ----: | --------------------------------------------------------- |
| `port-up`        |      23 | 5,657 | UD-only or UD-superset capability → canonical Tools layer |
| `needs-decision` |       5 | 1,710 | A live Tools twin exists and **no G0 gate measures it**   |
| `app-local`      |       2 |   412 | Workbench glue; stays in UD                               |
| `already-home`   |       0 |     0 | See below                                                 |

**`already-home` is empty, and that is the headline result.** ADR-0046 uses
"already home" for three _Tools_ capabilities (canonical v2 wire, private-corpus
boundary, cross-runtime goldens); it does not describe any of UD's 30 modules.
Every UD module that has a Tools counterpart was measured or read to carry
outputs the counterpart cannot produce — G0 proved this numerically for the
three gated pairs (D2/D3/D4 for strokes gained, D6/D7 for dispersion, D10/D13/D14
for longitudinal), and reading established it for the rest. Nothing retires
without something moving first.

One sub-module exception is worth naming so it is not lost: the expected-strokes
**baseline half** of `strokes_gained_types.py` is genuinely already home. G0's
`test_baseline_table_digest_agrees_across_stacks` shows UD's
`baseline_table_sha256` and Tools' `baseline_table_hash` return the identical
digest `188a6eaf…` for the same states, and Tools'
`launch_monitor_strokes_gained_baseline.py` (206 lines) already carries byte
caps and source-URL validation UD lacks. That half retires into the Tools module
at port time; only the request/result/uncertainty half travels.

## Module Inventory

Consumer legend: `gui` = `src/tools/launch_monitor_analytics/gui.py`; `widgets`
and `fa-widget` = its sibling widget modules; `api` =
`src/api/routes/launch_monitor_analytics.py`; `jobs` =
`src/api/services/launch_monitor_dataset_jobs.py`; `gen` =
`scripts/generate_launch_monitor_contract.py`; `fixture` =
`scripts/launch_monitor_conformance_fixture.py`; `companion` =
`scripts/companion_workflow_tasks.py`. "Tests" counts test files that reach the
module. "internal" means no consumer outside the package.

| Module                              | LOC | Class            | Consumers (non-test)                 | Tests | Tools counterpart                                                                  | G0 findings    |
| ----------------------------------- | --: | ---------------- | ------------------------------------ | ----: | ---------------------------------------------------------------------------------- | -------------- |
| `__init__.py`                       | 308 | `app-local`      | 16 of 26 import statements target it |     — | none                                                                               | —              |
| `_scoring_statistics.py`            | 127 | `port-up`        | internal                             |     2 | none                                                                               | D2, D3, D4     |
| `comparison.py`                     | 147 | `port-up`        | gui                                  |     1 | none                                                                               | —              |
| `conformance_bundle.py`             | 207 | `port-up`        | gen, fixture                         |     1 | none (Tools' web consumes the golden this emits)                                   | —              |
| `contract_v2.py`                    | 791 | `port-up`        | api, gen, fixture                    |     4 | `launch_monitor_canonical_v2.py` (397) — pinned client half only                   | D14            |
| `corpus.py`                         | 197 | `needs-decision` | gui                                  |     1 | `launch_monitor_private_corpus.py` (106) — same env var, same path                 | none (ungated) |
| `dataset_reference.py`              |  35 | `port-up`        | api, jobs, gen                       |     1 | none (pure facade over the three below)                                            | —              |
| `dataset_reference_contract.py`     | 134 | `port-up`        | internal (via facade)                |     1 | `launch_monitor_canonical_v2.py` validators — client half only                     | —              |
| `dataset_reference_operations.py`   | 256 | `port-up`        | internal (via facade)                |     1 | none                                                                               | —              |
| `dataset_reference_verification.py` | 338 | `port-up`        | internal (via facade)                |     1 | none                                                                               | —              |
| `dispersion.py`                     |  70 | `port-up`        | gui                                  |     2 | `launch_monitor_performance.analyze_dispersion` — disjoint result surface          | D6, D7, D8, D9 |
| `flexible_analysis.py`              | 415 | `needs-decision` | api, fa-widget, fixture              |     3 | `launch_monitor_analysis.py` + 2 private modules (565)                             | none (ungated) |
| `importer.py`                       | 268 | `port-up`        | gui, companion                       |     1 | `launch_monitor_import.py` (245) — bounded reader, no profiles or units            | —              |
| `longitudinal.py`                   | 301 | `port-up`        | api, gen, fixture                    |     2 | `launch_monitor_longitudinal.py` (307) — different pooled estimator                | D10–D14        |
| `longitudinal_statistics.py`        | 147 | `port-up`        | internal                             |     2 | same module, different estimator                                                   | D10, D11       |
| `longitudinal_types.py`             | 144 | `port-up`        | api, fixture                         |     2 | same module's dataclasses                                                          | D11, D12       |
| `modeling.py`                       | 226 | `port-up`        | gui                                  |     1 | none (no scikit-learn anywhere in `rate_of_closure`)                               | —              |
| `multivariate.py`                   | 108 | `port-up`        | gui                                  |     1 | none                                                                               | —              |
| `outcome_proxy.py`                  | 114 | `port-up`        | api, fixture                         |     1 | `launch_monitor_performance.calculate_target_error` — identical formula            | none (ungated) |
| `player_covariation.py`             | 336 | `needs-decision` | api, gen, fixture                    |     2 | `player_covariation.py` (371)                                                      | none (ungated) |
| `player_covariation_core.py`        | 427 | `needs-decision` | internal                             |     2 | `_player_covariation_scan.py` (100) plus statistics inside the twin                | none (ungated) |
| `player_covariation_types.py`       | 335 | `needs-decision` | api, gen, fixture                    |     2 | `_player_covariation_types.py` (99)                                                | none (ungated) |
| `profiles.py`                       | 255 | `port-up`        | widgets                              |     1 | none                                                                               | —              |
| `project.py`                        | 104 | `app-local`      | gui                                  |     1 | `launch_monitor_workspace.LaunchMonitorProject` — row-free by design               | —              |
| `relationships.py`                  | 187 | `port-up`        | gui                                  |     1 | `_launch_monitor_analysis_statistics.correlations` — plain correlation only        | —              |
| `schema.py`                         | 195 | `port-up`        | gui, widgets                         |     1 | `CANONICAL_DATASET_METRICS` frozenset only — no metric definitions                 | —              |
| `strokes_gained.py`                 | 432 | `port-up`        | api, gen, fixture                    |     2 | `launch_monitor_strokes_gained.py` (345)                                           | D1–D5          |
| `strokes_gained_types.py`           | 440 | `port-up`        | api, fixture                         |     3 | `launch_monitor_strokes_gained_baseline.py` (206) — baseline half already home     | D1, D2, D5     |
| `treatment.py`                      | 215 | `port-up`        | gui                                  |     1 | none                                                                               | —              |
| `trends.py`                         | 110 | `port-up`        | gui                                  |     1 | `launch_monitor_performance.analyze_session_trend` — same name, different estimand | —              |

### Notes on Individual Rows

`project.py` is `app-local` on evidence, not convenience. UD's
`LaunchMonitorProject` holds the imported shot frames and persists them; Tools'
identically named class is deliberately row-free ("keeps private corpus rows out
of persistent project documents"). These are two different artifacts serving two
different retention postures. Porting either onto the other would break one of
them.

`__init__.py` is `app-local` because it is the re-point seam, and the seam is
cheap: 16 of the 26 external import statements resolve through the facade, so
Stage 2 re-pointing is mostly a rewrite of this one file rather than 26 edits.

`trends.py` and `dispersion.py` both collide by name with Tools functions that
compute something else. G0 pinned the dispersion case numerically (D7: Tools'
1-D `rms_yards` 8.397 versus UD's 2-D `radial_rmse` 11.365, 35% apart and not a
unit factor). The trends collision is unmeasured: UD's `analyze_trend` returns a
per-day robust slope, EWMA, and change-point candidates, while Tools'
`analyze_session_trend` returns cumulative session-ordinal means. Both export a
frozen dataclass named `TrendResult`. A mechanical vendor transition that lands
both in one namespace will merge them wrong.

`outcome_proxy.py` is `port-up` rather than `needs-decision` despite being
ungated, because both implementations were read and the closed form is
character-for-character the same statistic — `hypot(carry_yd − target_yd,
lateral_yd)` after identical yard conversion. UD's is a strict superset: it adds
exclusion accounting, an uncertainty summary, an availability status, and an
explicit "this is not strokes gained" claims block. A gate is still required
before the port lands (see the port order), but the classification does not
depend on its outcome.

## Named-Method Decisions Surfaced by G0

These four are the substantive content of G1. They are written as decisions so
that reviewing this document ratifies them.

### Decision G1-D1: Pooled Longitudinal Estimator Becomes a Named-Method Pair

**Decision.** Preserve both estimators, exposed as named, provenance-carrying
options exactly as ADR-0045 preserved the two roll models. Proposed identifiers:

- `ud-cluster-robust-fe/1` — player fixed-effects OLS with standard errors
  clustered by player; UD's existing `method` string is already
  `player_fixed_effects_ols_clustered_by_player`.
- `dl-random-effects/1` — inverse-variance weighting with the
  DerSimonian–Laird between-player variance estimate, carrying `tau_squared`,
  `q_statistic`, `i_squared_pct`, and `improvement_probability`.

Every pooled result document names its estimator. Results from different
estimators are never numerically compared without the names attached. Neither
is removed.

**Rationale.** D10 is not a bug in either stack; it is two defensible answers to
different questions. The same four per-player slopes go in — G0 asserts
`max |UD − Tools| = 0.0` — and the pooled verdicts come out opposite: UD's
interval `[−1.576, +0.525]` crosses zero (p = 0.210), Tools' random-effects
interval `[−1.015, −0.042]` does not, and Tools additionally reports a 98.3%
improvement probability. The point estimates agree to 0.52%; it is the
uncertainty model that differs, and UD's interval is 2.16× wider. Cluster-robust
inference with four clusters is known to be anti-conservative in the opposite
direction from what a t-distribution assumes, and DerSimonian–Laird with four
studies is known to underestimate `tau_squared`. Neither is "the" right answer
at k = 4, and picking one silently would hand a user a significance verdict that
the other method contradicts. The named-method-pair treatment is the only option
that keeps both honest, and it is the treatment the repo already ratified for the
putting roll models.

**Consequence.** The canonical `PooledAssociationV1` gains a required method
identifier and the union of both estimators' output fields (UD's
`standard_error`/`p_value`, Tools' heterogeneity block). D11's per-player
uncertainty gap closes in the same change: `LongitudinalPlayerAssociationV1`
gains `standard_error`, `ci_lower`, `ci_upper`, `p_value`, `r_squared`, and
`first_to_last_change`, which UD cannot express today at all.

### Decision G1-D2: The Canonical Inference Unit Is the Session Cell

**Decision.** The canonical estimand for any longitudinal fit is the
**player-session cell**: shots aggregate to one value per player per session
before any slope is fitted. UD's existing shot-level strokes-gained fit is
preserved as a named variant, proposed `shot-level-sg-trend/1`, and is never
reported as the same quantity as the session-cell fit.

**Rationale.** This is an intra-UD contradiction, not a cross-stack one, and UD
has already argued both sides. `longitudinal.py` aggregates to 20 player-session
cells and its module docstring names "player-session as the inference unit";
D5 records that `strokes_gained.py` fits the same players' trends over all 40
shots each, treating eight shots from one session as eight independent
observations. UD's own module warns against exactly that pseudo-replication.
When a repository's two modules disagree and one of them has already written
down why the other is wrong, the written-down argument wins. Tools independently
made the same choice — its `analyze_longitudinal_performance` collapses to
session points first — so session-cell is also the option that requires no
change on the Tools side.

**Consequence.** The strokes-gained longitudinal summaries change shape:
`sample_count` per player goes from 40 to 5, and the pinned P4 values in G0
(`slope` 0.0758810, `r_squared` 0.1545044) move. Those pins must be re-derived
in the same PR that makes the change, with the old values retained as the
`shot-level-sg-trend/1` variant's pins so the gate keeps measuring both.

### Decision G1-D3: The Canonical Error Posture Is Exclude-and-Audit

**Decision.** The canonical layer excludes a malformed row, records it against a
`reason_code`, sets `status="partial"`, and returns a result. Raising on a
malformed row is not a canonical behaviour. Silently dropping a row is
prohibited outright.

**Rationale.** D1 is the clearest asymmetry G0 found. One malformed row in 161
destroys the entire session result in three of Tools' four failure modes; in the
fourth (blank lie, context, and target) Tools drops the row with no exclusion
record at all, which is the worst outcome available — a wrong-but-plausible
number with nothing to say it is short a row. UD returns `status="partial"`,
`by_reason={reason_code: 1}`, and a mean unchanged at
0.80592372152815683. An audit trail is strictly more information than an
exception, and strictly more than silence. A caller that wants fail-closed
behaviour can raise on `status != "available"`; a caller handed an exception
cannot recover the 160 good rows.

**Consequence.** Tools' `calculate_source_backed_strokes_gained` stops raising
on out-of-baseline states, invalid distances, and unknown strata. The three
`pytest.raises(ValueError, match="outside the baseline")` assertions in G0's
`test_divergence_d1_malformed_row_handling` become assertions on a `partial`
result, and the silent-drop case gains an exclusion record. This is a
deliberate behaviour change on the Tools side and must land in the same PR as
the strokes-gained port, not before it.

### Decision G1-D4: Ungated Twins Get a Gate Before They Get a Classification

**Decision.** No module in the `needs-decision` set moves, and no owner decision
is requested on it, until a G0-style gate measures the pair on the same
`adr0046_cross_stack_session_v1.json` fixture. The gate PR precedes the decision;
the decision precedes the port.

**Rationale.** ADR-0046's own Stage 0 rule is "measure the divergence before
anything moves", and it was applied to the three overlaps the ADR knew about.
Five modules turned out to have live Tools twins the ADR does not mention (see
Corrections below). Classifying them from a reading is exactly the guess this
document is supposed to avoid — the G0 evidence is the reason we know the
dispersion pair shares zero field names and the longitudinal pair disagrees
about significance, and neither was visible from the module names.

## Rows That Need an Owner Decision

Five modules, 1,710 lines, all blocked on the same missing measurement.

| #   | UD module(s)                                                                                           | Tools twin                                                                                                                      | Why it defies classification                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `flexible_analysis.py` (415)                                                                           | `launch_monitor_analysis.py` (228) + `_launch_monitor_analysis_statistics.py` (200) + `_launch_monitor_analysis_types.py` (137) | ADR-0046 lists flexible analysis as a UD-only capability. It is not. The two stacks define six identically named frozen dataclasses — `CorrelationEstimate`, `CoefficientEstimate`, `ResidualDiagnostics`, `DatasetSummary`, `GroupAnalysis`, `RegressionEstimate` — plus `CONTRACT_VERSION`, `AnalysisMode`, `CorrelationMethod`, and `MissingPolicy`. Nothing measures whether they agree numerically. Neither is obviously a superset: UD carries `FlexibleAnalysisRequest`/`Result`, Tools carries `AnalysisRequest`/`AnalysisResult` with a `fingerprint_sha256` and vendor/session provenance UD does not have. |
| 2   | `player_covariation.py` (336), `player_covariation_core.py` (427), `player_covariation_types.py` (335) | `player_covariation.py` (371), `_player_covariation_scan.py` (100), `_player_covariation_types.py` (99)                         | ADR-0046 lists player covariation as a UD-only capability. It is not. Tools has a full within-player association plus cross-player meta-analysis implementation with Fisher-z intervals and a `MIN_FISHER_SAMPLES` floor, structured as the same three-module split. UD's is 1,098 lines to Tools' 570, so a size argument favours UD, but the extra lines are largely the V1 wire contract (ADR-0038) rather than statistics. This is the single largest unmeasured surface in the platform.                                                                                                                         |
| 3   | `corpus.py` (197)                                                                                      | `launch_monitor_private_corpus.py` (106)                                                                                        | Both read the same physical dataset — same `LAUNCH_MONITOR_DATA_ROOT` env var, same `data/authority/database/shot_corpus_parquet` path — with complementary and non-overlapping guarantees. UD canonicalises source-native imperial columns into the ADR-0031 schema; Tools validates the manifest hash, the source partition set, and a `MAX_RETAINED_ROWS` desktop cap. Neither is a subset. The correct outcome is a merge, and ADR-0046's three-way taxonomy has no merge bucket.                                                                                                                                 |

Recommended resolution for all three, for the owner to accept or reject: extend
the G0 fixture with the three missing comparisons (a covariation pair, a
flexible-analysis run, and a corpus round-trip against a synthetic partition),
land them as **G0.1** under the same issue line, then classify. The covariation
gate is the one to build first — it is the largest surface and the only one
where an ADR statement is affirmatively wrong.

## Owner Rulings (2026-09-02)

G0.1 landed the flexible-analysis and player-covariation comparisons
recommended above and pinned fourteen divergences (D15–D26) across
`tests/integration/launch_monitor_drift/test_flexible_analysis_drift.py` and
`tests/integration/launch_monitor_drift/test_player_covariation_drift.py`. The
repo owner has now ruled on four of those, narrowing what the eventual
flexible-analysis and covariation gates need to encode. The remaining pins in
each file are unruled and do not carry a decision here.

**D15 — FDR multiplicity denominator**
(`test_divergence_d15_multiplicity_denominator_differs`). UD keeps an
under-sampled predictor's raw p value in the Benjamini-Hochberg pool (below
`min_samples`) and only blanks the reported values afterwards, correcting
against k=4; Tools drops it before correcting, against k=3. **Ruling:** the
canonical layer excludes under-sampled predictors from the FDR denominator
before correcting — Tools' existing posture. UD's count-all behaviour is a
defect, not a preserved method. This applies to the canonical
`relationships.py`/flexible-analysis modules; the P7 port lands UD verbatim
first, and a follow-up PR applies this ruling with updated pins.

**D17 — boolean columns**
(`test_divergence_d17_boolean_columns_are_analysed_only_by_ud`). UD's
`pd.to_numeric` projects `True`/`False` to 1.0/0.0 and analyses the column;
Tools' `finite_launch_monitor_scalar` refuses booleans and raises "Constant
variables cannot be analyzed". **Ruling:** the canonical layer analyses
booleans as 0/1 — UD's capability is preserved — but the projection must be
explicit in the result: a column analysed via boolean projection is labelled
as such and can never read as native numeric. Tools' refusal message becomes
a pointer to the explicit path rather than a dead end.

**D22 — low-dof Fisher intervals**
(`test_divergence_d22_between_player_interval_exists_only_in_tools`). With
four player means Tools returns a Fisher-z interval on n-3 = 1 degree of
freedom; UD sets `include_interval=False` for that scope and returns `None`.
**Ruling:** the canonical layer withholds the between-player Fisher interval
when degrees of freedom make it uninformative — UD's posture — with the
threshold documented and the absence explained in the result rather than
silently `None`. Applies at P18 (`player_covariation*`).

**D23 — unit labelling**
(`test_divergence_d23_unit_resolution_differs`). Tools' column-name-suffix
heuristic labels `start_distance_yards` as `"s"` (seconds); UD resolves units
from the canonical registry and returns `canonical_unit="unknown"` rather than
guessing. **Ruling:** the suffix heuristic is a defect and is deleted. The
canonical layer resolves units from the canonical registry and returns
unknown rather than guessing. Applies at P18.

## Port Order

Target for every `port-up` module: `src/shared/python/launch_monitor/<same
filename>` in Tools, mirroring the UD name so the Stage 2 vendor transition is a
mechanical import rewrite (`src.shared.python.launch_monitor.X` →
`shared.python.launch_monitor.X`) rather than a symbol remap. This also keeps the
new package out of `rate_of_closure`, which contains the colliding names.

Ordering is dependency-legal first, then smallest-first inside each tier. The
intra-package graph has six modules with no internal dependencies at all; those
form tier 0 and can proceed in any order.

| PR  | Source module(s)                                                           |   LOC | Tests that travel                                                                                                                                             | Gate that must stay green                                                                                                                                                                            |
| --- | -------------------------------------------------------------------------- | ----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1  | `dispersion.py`                                                            |    70 | dispersion half of `tests/unit/launch_monitor/test_analysis.py::test_dispersion_and_longitudinal_trend_capture_change`, split into a new `test_dispersion.py` | `tests/integration/launch_monitor_drift/test_dispersion_drift.py` (all 7)                                                                                                                            |
| P2  | `multivariate.py`                                                          |   108 | `test_analysis.py::test_pca_and_vif_expose_multicollinearity`                                                                                                 | all three G0 files (regression only)                                                                                                                                                                 |
| P3  | `trends.py`                                                                |   110 | trend half of the same split test                                                                                                                             | all three G0 files; plus the `TrendResult` rename (executed in Tools#4899 as `TemporalTrendResult`, deliberately with no back-compat alias — Stage 2's import rewrite must special-case this symbol) |
| P4  | `comparison.py`                                                            |   147 | `test_analysis.py::test_matched_monitor_comparison_recovers_bias_and_slope`                                                                                   | all three G0 files                                                                                                                                                                                   |
| P5  | `schema.py`                                                                |   195 | `test_importer.py` mapping cases                                                                                                                              | all three G0 files                                                                                                                                                                                   |
| P6  | `treatment.py`                                                             |   215 | two `test_analysis.py` treatment cases                                                                                                                        | all three G0 files                                                                                                                                                                                   |
| P7  | `relationships.py`                                                         |   187 | `test_analysis.py::test_correlations_include_counts_significance_and_derived_warning`                                                                         | all three G0 files                                                                                                                                                                                   |
| P8  | `modeling.py`                                                              |   226 | three `test_analysis.py` model cases including the leakage guard                                                                                              | all three G0 files                                                                                                                                                                                   |
| P9  | `profiles.py` + `importer.py`                                              |   523 | `tests/unit/launch_monitor/test_importer.py` (6 of 7 cases)                                                                                                   | all three G0 files                                                                                                                                                                                   |
| —   | **blocked on G1-D4** — `contract_v2.py` imports `flexible_analysis`        |       |                                                                                                                                                               |                                                                                                                                                                                                      |
| P10 | `flexible_analysis.py` (after G0.1 + decision)                             |   415 | `test_flexible_analysis.py`                                                                                                                                   | new flexible-analysis gate                                                                                                                                                                           |
| P11 | `contract_v2.py`                                                           |   791 | `test_contract_v2.py`                                                                                                                                         | all three G0 files                                                                                                                                                                                   |
| P12 | `strokes_gained_types.py` (minus baseline half) + `_scoring_statistics.py` |   567 | `test_strokes_gained_contract.py`, `tests/api/test_routes_launch_monitor_analytics.py` baseline cases                                                         | `test_strokes_gained_drift.py`                                                                                                                                                                       |
| P13 | `outcome_proxy.py`                                                         |   114 | `test_strokes_gained_contract.py` proxy cases                                                                                                                 | new target-error gate landed in this PR                                                                                                                                                              |
| P14 | `strokes_gained.py`                                                        |   432 | `test_strokes_gained_contract.py`                                                                                                                             | `test_strokes_gained_drift.py`, with D1 re-pinned per G1-D3 and D5 per G1-D2                                                                                                                         |
| P15 | `longitudinal_types.py` + `longitudinal_statistics.py`                     |   291 | `test_longitudinal_sessions.py`, `tests/api/test_launch_monitor_longitudinal.py`                                                                              | `test_longitudinal_drift.py`                                                                                                                                                                         |
| P16 | `longitudinal.py`                                                          |   301 | `test_longitudinal_sessions.py`                                                                                                                               | `test_longitudinal_drift.py`, carrying G1-D1's named-method pair                                                                                                                                     |
| P17 | `conformance_bundle.py`                                                    |   207 | `test_conformance_bundle.py`                                                                                                                                  | Tools' `launchMonitorConformanceGolden.test.ts`                                                                                                                                                      |
| P18 | `player_covariation*` (after G0.1 + decision)                              | 1,098 | `test_player_covariation_contract.py`, `tests/api/test_routes_launch_monitor_covariation.py`                                                                  | new covariation gate                                                                                                                                                                                 |
| P19 | `corpus.py` (after G0.1 + decision)                                        |   197 | `test_corpus.py`                                                                                                                                              | new corpus gate                                                                                                                                                                                      |
| P20 | `dataset_reference*` (4 modules)                                           |   763 | `test_dataset_reference_jobs.py`, `tests/api/test_launch_monitor_dataset_jobs.py`                                                                             | all three G0 files                                                                                                                                                                                   |

Two structural facts shape this order and should be read before it is changed.

`tests/unit/launch_monitor/test_analysis.py` is a single 199-line file covering
seven modules across nine test functions, and one of those functions
(`test_dispersion_and_longitudinal_trend_capture_change`) covers two modules at
once. "Tests travel with the module" therefore requires splitting this file
before P1, not during it. The same is true to a lesser degree of
`test_importer.py`, which covers `importer`, `profiles`, `schema`, and the
`app-local` `project`.

`contract_v2.py` imports `flexible_analysis` — its own docstring says "The
numerical implementation remains in `flexible_analysis`." The entire v2 contract
layer, and everything above it (`strokes_gained_types`, `longitudinal_types`,
`player_covariation_types`, `conformance_bundle`), therefore sits on top of a
module whose Tools twin has never been measured. The port cannot proceed past
tier 1 until G1-D4's first gate lands. This is the tightest constraint in the
plan and it is not visible from ADR-0046.

## Stage 2 Blocker (G2): The Canonical Import Path Is Shadowed

Discovered 2026-09-02 attempting ADR-0046 Stage 2 wave 1 (`dispersion`,
`multivariate`, `trends`, `comparison`, `schema`, `treatment`) against the
vendored layer at pin `6238889a9`. The port order above prescribes the Stage 2
re-point as a mechanical rewrite:

```
src.shared.python.launch_monitor.X  ->  shared.python.launch_monitor.X
```

**In UpstreamDrift that rewrite is a no-op alias.** It resolves back to the
module it is supposed to replace, and once that module is deleted the import
fails outright.

### The Measurement

`shared` and `shared.python` are reachable through two roots at once —
UpstreamDrift's `src/` and the vendor tree's `vendor/ud-tools/src/` — and both
carry an `__init__.py` for `launch_monitor`. A regular package is not a
namespace portion: the **first** `shared.python.__path__` entry that contains
`launch_monitor` wins outright, and the UpstreamDrift entry precedes the vendor
entry under the repository's own path wiring
(`pyproject.toml` `[tool.pytest.ini_options] pythonpath`, then
`tests/conftest.py::pytest_configure` in its default `--tools-mode=local`,
which inserts `src/shared/python` at `sys.path[0]`).

```sh
# Under the repository's real test wiring:
#   shared.python.__path__ == [src/shared/python, vendor/ud-tools/src/shared/python]
#   shared.python.launch_monitor -> src/shared/python/launch_monitor
```

Performing the full wave-1 rewrite and deleting the six UpstreamDrift modules
produces, on the first import of the package façade:

```
src/shared/python/launch_monitor/__init__.py:8: in <module>
    from shared.python.launch_monitor.comparison import (
src/shared/python/launch_monitor/__init__.py:8: in <module>
    from shared.python.launch_monitor.comparison import (
E   ModuleNotFoundError: No module named 'shared.python.launch_monitor.comparison'
```

The doubled frame is the whole finding: the façade is importing **itself**
under the canonical name.

### Why It Cannot Be Worked Around per Module

This is not per-module coupling — the six wave-1 modules are tier-0 leaves with
no intra-package imports, and each is byte-for-byte equivalent to its canonical
twin modulo the port's docstrings, `__all__`, and the P3 `TrendResult` rename
(pinned by `tests/unit/launch_monitor/test_canonical_layer_parity.py`). The
blocker is at package granularity, and so is the guard that tracks it:
`tests/unit/repo_hygiene/test_no_shadow_of_tools_shared.py` enumerates
**top-level entries** under `vendor/ud-tools/src/shared/python/` and compares
them with top-level entries under `src/shared/python/`. `launch_monitor` is one
entry either way. Retiring six of its thirty files changes nothing the guard
can see, and `scripts/config/shadow_modules.yaml` cannot be narrowed to "six
files retired" because it has no per-file vocabulary.

`shadow_modules.yaml`'s own header states the resolution procedure, and it is
all-or-nothing: "move the canonical implementation into Tools, land it there,
bump the `vendor/ud-tools` pin, **delete the UD-side copy**, and drop the line".
Module-by-module retirement of a shadowed package is not something
UpstreamDrift's import layout supports today. That constraint is invisible from
ADR-0046, which assumed the two layers could coexist during the transition.

### Options for the Owner

Each unblocks Stage 2; they differ in blast radius and in what they cost the
wave structure.

1. **Move UpstreamDrift's transitional copy out of the `shared.python`
   namespace** (for example to an app-local package beside the workbench).
   This clears the `launch_monitor` ledger entry immediately, makes
   `shared.python.launch_monitor` unambiguous, and lets waves 1..N proceed
   exactly as the port order writes them. It is a mechanical move of the
   remaining modules plus their import statements, with no behaviour change,
   and it wants its own PR so that the retirements that follow stay reviewable
   as no-ops. ADR-0048 already classifies `__init__.py` and `project.py` as
   `app-local`, so this is where two of the thirty modules were going anyway.
2. **Retire the whole package atomically** in one Stage 2 PR. Cheapest in
   sequencing, but it forces the `needs-decision` rows (`flexible_analysis`,
   `player_covariation*`, `corpus`) and the behaviour-changing merges
   (P19's mandatory manifest validation, G1-D1/D2/D3) through in the same
   change, which contradicts ADR-0046's "measure before anything moves" and
   its per-tab validation rule.
3. **Overlay the vendored directory onto the package's `__path__`** so retired
   modules fall through to the vendor tree while the import strings stay as
   they are. It works and it is deterministic, but the consuming code no longer
   says where its behaviour comes from, and it introduces a failure mode for
   any distribution of UpstreamDrift that does not carry the vendor tree.
   Recorded for completeness; not recommended.

**Executed 2026-09-02 (#9420): Option 1.** UpstreamDrift's transitional copy
moved from `src/shared/python/launch_monitor/` to
`src/tools/launch_monitor_model/`, beside the workbench; the `launch_monitor`
shadow-ledger entry cleared, and `shared.python.launch_monitor` now resolves to
the vendored canonical package, unblocking Stage 2 wave 1.

Before that, Stage 2 wave 1 was blocked at step 2 of its own procedure. Wave
1's step 1 (consumer inventory) and its identity premise were complete and
pinned in `test_canonical_layer_parity.py`; no UpstreamDrift module had been
retired, because retiring one with a failing import is the explicit non-goal
ADR-0046 names.

## Risks

### The Vendored-Pin Cadence Is Already Load-Bearing and Already Unguarded

ADR-0046 lists this as a Consequence to watch. It is worse than that today: the
safety net does not run.

`tests/integration/launch_monitor_drift/conftest.py` calls
`require_vendored_tools_stack()`, which issues `pytest.skip(...,
allow_module_level=True)` when `vendor/ud-tools/src/rate_of_closure/launch_monitor_strokes_gained.py`
is absent. `grep -rn 'submodules' .github/workflows/*.yml` shows nine workflows
requesting submodules and `ci-standard.yml` is not among them — its
`actions/checkout` steps take no `with:` block for submodules. All 28 G0 gates
therefore skip in the repository's main lane, and a skipped gate reports green.

Meanwhile `vendor-freshness.yml` runs `git submodule update --remote
vendor/ud-tools` on a schedule and opens an auto-labelled bump PR whose body is
"Review the Tools CHANGELOG before merging" — with no gate execution attached.
So the pin advances automatically, and the only thing that would catch a
numerical regression from the new pin is a test suite that does not run.

Recommended before P1: make the drift gates hard-fail rather than skip when an
environment variable such as `UD_REQUIRE_VENDORED_TOOLS=1` is set, set it in
`ci-standard.yml`, and add `submodules: recursive` to the tests job's checkout.
Add the same job to the `vendor-freshness` bump PR. Without this, every
statement in the rest of this plan about "the gate that must stay green" is
aspirational.

### The TypeScript-Twin Obligation Is Unsized

Tools' `rate_of_closure` posture is one TypeScript model twin per Python model
module, plus a pinned cross-runtime fixture. The counts are currently 1:1 — 18
Python `launch_monitor_*`/`player_covariation*` modules against 18 non-test
`web/src/model/launchMonitor*.ts` files (32 including tests).

ADR-0046's Consequences frame the twins as a positive: UD-only capabilities
"arrive in the canonical layer's TS twins". It does not size that. Landing 23
`port-up` modules and 5,657 lines into the canonical layer takes the obligation
from 18 twins to 41 if the 1:1 posture holds — roughly a tripling of the web
model surface, plus a pinned fixture each. Some of it is not portable at all:
`modeling.py` (226 lines) fits a scikit-learn `MLPRegressor`, and there is no
scikit-learn anywhere in `rate_of_closure`.

Recommended: the owner rules explicitly on one of three options — twins for the
gated overlap set only, twins for everything with a documented exemption class
for desktop-only modules, or a deferred-twin policy where the Python module
lands first and the twin is a tracked follow-up. Whichever is chosen belongs in
ADR-0046's Consequences, because it changes the size of Stage 1 by more than the
port itself.

**Owner ruling (2026-09-02):** deferred-twin — the Python module lands
first and stands alone; the TS twin is a tracked follow-up, prioritized
when a web surface (Stage 2 re-pointing) needs it. Recorded in
[ADR-0046's Consequences](0046-launch-monitor-analytics-single-model-layer.md#consequences).

### Tools' Stated Architecture Contradicts Hosting the Canonical Inferential Layer

ADR-0046 places the canonical layer in Tools because Tools is the fleet's DRY
leaf. Tools' own modules say the opposite about statistics specifically.
`launch_monitor_performance.py`: "Inferential statistics remain an UpstreamDrift
concern." `launch_monitor_workspace.py`: "Statistics remain owned by the
UpstreamDrift backend." `launch_monitor_v2_client.py` (267 lines) exists solely
to call UD over HTTP for exactly the analyses this plan proposes to move into
Tools, and `launch_monitor_canonical_v2.py` (397 lines) is a pinned client copy
of UD's v2 schemas.

If the canonical layer lands in Tools, roughly 664 lines of client seam become
either redundant or ambiguous, and the UD API routes become a server for
contracts whose definitions live downstream of them. That is not a reason to
reject ADR-0046 — the DRY-leaf argument still holds and the alternative was
already rejected as inverting the fleet dependency — but it is a decision the
ADR did not record. The owner should state whether the v2 HTTP seam survives the
port, and if so what it is for.

### Name Collisions Will Merge Silently if the Namespace Is Flattened

Beyond the two G0 pinned cases, reading turned up `LaunchMonitorProject`
(different retention posture), `load_private_corpus` (different validation),
`CONTRACT_VERSION` (three different values), and six identically named frozen
dataclasses shared between UD's `flexible_analysis` and Tools'
`_launch_monitor_analysis_types`. Mirroring UD's filenames into a new
`src/shared/python/launch_monitor/` package in Tools keeps these in a different
package from `rate_of_closure`, which contains the colliding definitions — this
is the main reason to prefer that target path over merging into
`rate_of_closure`. The containment lasts exactly as long as nobody adds a
convenience re-export.

### ADR-0046's Status Line Is Stale

The ADR file still reads `Status: Proposed`, while #9348 and this plan's task
framing describe it as accepted. Whoever ratifies this document should update
that line in the same review, so the two do not disagree in the record.

## Corrections to ADR-0046

Three factual claims in the accepted ADR did not survive measurement. They are
listed here rather than edited into the ADR, since amending an accepted decision
record is the owner's call.

1. **"Capabilities that exist only in UD's layer (comparison, treatment, player
   covariation, dataset-reference verification, flexible analysis)."** Two of
   the five are not UD-only. Tools has `player_covariation.py` plus two private
   support modules (570 lines) and `launch_monitor_analysis.py` plus two private
   support modules (565 lines). Comparison, treatment, and dataset-reference
   verification are confirmed UD-only.
2. **"Tools stack (vendored): `rate_of_closure/launch_monitor_*` (~12 Python
   modules)."** The count is 18 modules and 4,240 lines — the ADR's pattern
   misses the two private `_launch_monitor_analysis_*` modules and the three
   `player_covariation*` modules, which is why correction 1 was missed.
3. **The UD module list is complete but omits `_scoring_statistics.py`**, the
   127-line shared uncertainty and grouping helper that is the actual
   implementation behind divergences D2, D3, and D4. It has no Tools
   counterpart and is the highest value-per-line module in the `port-up` set.
