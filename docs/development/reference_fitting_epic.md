# C3D Reference Fitting Epic

Governing issue: #9914. State: in_progress. Owner: codex.
Branch: `feat/c3d-reference-overlay-9914`.

## Acceptance and Execution Plan

1. Explicit, persisted marker profiles map C3D measurements to reconstruction
   landmarks in metres and a right-handed frame. Missing members stay missing.
2. Fit every registered articulated model using the existing continuous solver;
   record source hash, exact model specification, options, all-point errors,
   rejection coverage and native support limitations.
3. Export full joint-tree animations as existing ReferenceMotion library assets.
   Preserve the source clock and all model geometry, including unobserved joints.
4. Reuse comparison registration, event synchronization and rendering. Add tested
   fixed similarity estimation from selected correspondences; never align each
   frame independently and thereby erase swing differences.
5. Run Tour Average driver and iron data, record quantitative evidence, qualify
   playback and document reproduction plus extension to further model adapters.

## Reuse and Boundaries

Merged #9863 supplies assets, registration, synchronization and the capture UI.
#9709 supplies the articulated model registry and continuous robust solver.
No vendor edits, mock native fits, or changes to other agents' worktrees.
Marker surface positions are proxies, not measured joint centers; unconstrained
twist and unobserved degrees of freedom must not be presented as measurements.

## Validation

RED: `python3 -m pytest tests/motion_capture/test_reference_fitting.py -q -n 0
--no-cov --timeout=60` fails because `reference.fitting` does not yet exist.

GREEN: 85 tests in `tests/motion_capture/test_reference*.py` pass with
`python3 -m pytest <expanded test paths> -q -n 0 --no-cov --timeout=60`.
Ruff check passes. File-size and design-manual governance checks pass. The
central development-log validator detects pre-existing duplicated entries and
missing fields/SHA values elsewhere in the log; the new issue-keyed entry has
its verifying baseline recorded. Native OpenSim fitting remains unavailable;
the bundled MyoSuite body assets are placeholders, explicitly inventoried.
