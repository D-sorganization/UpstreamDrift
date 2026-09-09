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

## Tour Average Evidence

`reference_fit_qualification.json` records the exact two saved jobs, source
hashes, reference identities, model inventory, residuals and bundle manifest
hashes. All output file hashes were independently verified for all 18 bundles.
The survey uses stride 12, 50 evaluations per robust stage, and explicitly
learnable dimensions. It retains source timestamps. Full-rate solves are
additional qualification runs, not the basis of this survey evidence.

| Model               | Driver RMS (mm) | Iron RMS (mm) |
| ------------------- | --------------: | ------------: |
| golfer              |           48.51 |         49.69 |
| double_pendulum     |           33.10 |         26.86 |
| triple_pendulum     |            9.60 |          8.01 |
| pinocchio_golfer    |          167.60 |        171.85 |
| pinocchio_golfer_ik |          167.60 |        171.85 |
| drake_golfer        |          131.69 |        138.96 |
| simple_humanoid     |          196.30 |        188.12 |
| human_subject       |          101.11 |         96.52 |
| mujoco_humanoid     |          236.01 |        226.31 |

Scores use different mapped landmarks and are not a ranking. Fixed native
geometry, missing endpoints and surface proxies materially limit the fits.
The articulated golfer has the closest full-body fit in this tested catalog;
this does not establish anatomical validity or optimal convergence.

On combined main `18c8f922e` and this implementation, all 134 reference and
Capture Rig reference UI tests pass. The normal pre-push gates passed on
`74e867786`: Ruff, formatting, governance, mypy, Bandit and unit tests.
The merge preserves the capture agent's calibration work and current handoff.

The first full-rate driver run is withdrawn: it contained a negative learned
shoulder offset despite finite positions. Its prior RMS does not qualify it.
The invalid library asset was moved to the rejected artifact directory, and
its result remains explicitly marked in `withdrawn_runs` for audit.

The architecture budget initially caught a 104-line MJCF loader. Extracting
compiled-topology validation restores the 100-line budget; native FK parity,
Ruff, file-size, architecture and SPEC-duplicate checks pass afterward.

Final DbC audit: a manufactured incompatible capture first learned a negative
segment length (RED). The shared continuous solver now bounds learned lengths
strictly positive; a separate reference postcondition rejects negative/nonfinite
solver output before asset creation. Ten focused tests pass (GREEN). Both full
captures and the catalog were rerun with this corrected solver.

## Final Corrected Qualification

All 20 bundles from solver commit `2e84bb810` pass file-hash and finite-positive
dimension verification. Driver full-rate (360 Hz): 654 frames, 9,800 retained observations,
48.86 mm RMS and 100.54 mm maximum. Iron full-rate (359 Hz): 657 frames, 9,839 retained
observations, 49.92 mm RMS and 96.62 mm maximum. Neither run rejects observations;
both use 300 evaluations across three stages, which is not a convergence claim.

All 177 reference, Capture Rig reference UI and existing articulated/image-space
solver tests pass. All 26 documentation-governance tests pass after preserving
the primary blocker order. Exact duplicate field-reference tables were removed
to keep the development log within its byte budget; feature entries remain.

A metadata-only follow-up adds capture/model/sample-rate library titles (RED/GREEN
identity/title test and compositor test). Published library copies have matching
display labels; the numerical bundles and their original hashes are unchanged.
The final library is `../reference-fit-artifacts-9914/final-library`. Earlier
provisional folders are not the final deliverable. PR #9918 follows protected CI;
the inherited GUI LoD cleanup is owned by #9917 and will be integrated from main.
