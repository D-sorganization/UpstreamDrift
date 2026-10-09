# Predeclared Face-Bounds Supplement

Issue #12000 addresses a matching-search artifact. Original 324-cell primary
and 72-cell control matrices remain immutable records of their original bounded
protocol. This plan adds a separately identified supplement; no Monte Carlo is
authorized before the gates below and completion of the active frozen workers.
The exact cell inventory is `SUPPLEMENTAL_BOUNDS_PLAN.json`.

## Evidence and Scope

PW zero-lean strong curves are feasible at face ±0.260459639°, speed
32.045914261 m/s, with 0.000211396 m fine-replay residual. PW 5° lean at the
same 77.402477249 m target is feasible at face ±0.076707602°, speed
29.161308145 m/s, with 0.000289371 m fine-replay residual. The original face
boxes excluded the necessary sign. See the two source-bound probe JSON files.

Driver 5° fixed-target strong curves remain bounded-unavailable. Three starts
per sign converge to face approximately ±4.8716056°, speed 56.25 m/s (the
unchanged ceiling), and an endpoint 0.635088 m short with negligible lateral
residual. The face bound is inactive. This is evidence of an active speed
constraint and stable bounded solve, not a global physical-impossibility proof.
Neither its speed limit nor the target will be changed in this supplement.

## Fixed Experimental Inputs

- Absolute nominal face-search box [−6°, +6°] for every pattern, replacing only
  the reference-centered search box. Face-minus-path is 0° or ±3° at curve ×2.
- PW speed bounds [24, 40] m/s; raw endpoint residual tolerance 0.05 m.
- Unchanged static geometry, contact law, head properties, native flight,
  numerical settings, scoring baselines, seed 20261009, noise covariance,
  per-pattern 10,000 attempts, and no resampling or post-flight adjustment.
- Eighteen new within-lean PW zero-lean curve ×2 cells: two shaft-twist SDs
  (1°/2°), each with all nine original noise profiles. Three patterns per cell
  give 540,000 attempted shots.
- Four new fixed-target PW 5°-lean curve ×2 cells: two shaft-twist SDs and
  `shaft_only`/`combined2`. Three patterns per cell give 120,000 attempts.
- Four zero-lean fixed-target control entries reference the corresponding
  supplemental primary cells. They are provenance aliases, not four new runs.
  Total unique new execution is 22 cells and 660,000 attempted shots.

The nine profiles are `shaft_only`, `shaft_path`, `shaft_speed1`, `shaft_speed2`,
`shaft_strike`, `combined1`, `combined2`, `cov_facepath_positive`, and
`cov_facepath_negative`. Derive noise from the existing public profile provider;
never duplicate its covariance implementation.

## Comparisons and Reuse

Compare each supplemental primary curve ×2 cell with its original PW zero-lean
curve ×1 counterpart of identical profile and shaft SD. Verify nominal targets,
noise samples, seed, and physics projection before combining any records.
The widened search is a solver-domain change, not an input-distribution change;
replay the original available nominals under the new box to confirm that fact.

Individual density plots remain conditional on each pattern's supported shots.
For a curve-strength comparison table, use the common supported shot-index
intersection across all six populations (three patterns × two curve strengths).
Report its denominator and unsupported counts. Paired bootstrap intervals must
resample the same indices across compared populations. Distinguish pattern-wise
curve-strength changes from curve-versus-Straight contrasts; do not subtract
independently conditioned summary means and call them a paired effect.

For control reuse, a null within-lean target request and an explicit fixed-target
request are different orchestration configs. An alias must resolve both to the
same physical target and verify all other physical/noise/scoring parameters,
nominal endpoints, source projection, and exact saved input samples. Retain both
request configs and reference the unique source cell manifest/CSV hashes. Do not
forge a second execution receipt. Existing available PW 10° control records may
participate only after the same equivalence checks; original unavailable records
remain visible alongside supplement references.

## Acceptance and Audit Gates

1. Finish the frozen workers. Review configurable bounds and the separate DRY
   supplement runner test-first in the isolated implementation worktree.
2. Verify the numerical/contact/flight/scoring source projection against accepted
   evidence. Changes outside that projection still need focused software tests.
3. Replay supplemental nominal solutions at contact 512 and flight dt 0.005 s;
   keep the 0.05 m residual gate. Record original versus widened-box nominal
   equivalence for reference cells, and every remaining unsupported/bounded case.
4. Freeze a new supplement execution snapshot and run a separately named 32-shot
   pilot. Independently verify plan size, identities, paired samples, support,
   SG, aliases, and source stability before immutable final-run authorization.
5. Extend the independent auditor with an explicit supplement protocol and exact
   allowlisted 18+4 unique inventory, not a generic exemption from 324/72 checks.
   Test missing/extra/duplicate cells, altered target/bounds, aliases that imply
   independent execution, stale sources, and unsupported-denominator handling.
6. After authorized 10,000-shot execution, audit all 22 unique cells and four
   control references, plus comparisons to original curve ×1 and lean controls.
   Publish original and supplemental protocols separately with source hashes.

This supplement repairs goal coverage for strong wedge curves. It neither
converts the illustrative physical model into empirical qualification nor
relaxes the original numerical acceptance criteria.

## Separately Accepted Driver Speed-Bound Sensitivity

A later authorized diagnostic allowed upper speed 58.5 m/s (1.30× reference)
while retaining the face box, target, and all numerical tolerances. Driver 5°
strong curves match at speed 56.345074274 m/s, only 0.095074274 m/s above the
original 56.25 m/s ceiling. Draw face/path are +4.894851870°/+7.894851870°; fade
is mirrored. Fine replay residual is 0.000943089 m. Thus the original control
result is a speed-bound exclusion, not an established physical impossibility.

`DRIVER_SPEED_SENSITIVITY_PLAN.json` predeclares four separately labeled control
cells (two SDs × two profiles), totaling 120,000 additional attempts only after the same software, source, refinement, pilot, and execution gates. Keep
this speed change separate from the 22-cell face-only supplement. The combined
accepted plan is 26 uniquely executed cells and 780,000 attempts,
plus four provenance aliases. Original records remain immutable.

## Protocol Timing and Resolved Presentation

This supplement was adapted after the nominal feasibility audit and predeclared
before its Monte Carlo execution. It is not part of the original predeclared
324/72-cell protocol. Root accepted the separate driver speed extension, making
the exact new inventory 26 unique cells plus four PW zero-lean control aliases.

A resolved report may show the newly matched outcomes alongside the original
unavailable rows only with explicit source-protocol columns and links to the
underlying unique cell manifests. Never rewrite original matrix files or
describe supplemental cases as originally completed. Face-only and speed-bound
protocols remain distinguishable in every combined table or plotted claim.
