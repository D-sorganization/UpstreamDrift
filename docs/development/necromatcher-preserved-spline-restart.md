# Necromatcher Preserved Hermite Restart

## Purpose and Qualification

Issue #11235 adds an opt-in restart from the exact preserved parent Hermite
spline. A sampled-parent restart reconstructs knot positions and slopes from
stored frame poses; a preserved-spline restart instead retains the original
physical coefficients, knots, and derivatives. These are distinct numerical
initializations. Neither establishes calibrated physical time, historical
anatomy, contact, dynamics, or accepted captured motion. This procedure is
separate from the canonical QMD engineering design manual.

## Public Snapshot Contract

`ImageSplineStart` is a frozen public historical_fit record with six fields:

- `knot_times`: immutable finite strictly increasing source-clock times.
- `spline_coefficients`: immutable canonical physical q/v coefficients.
- `coordinate_order`: unique native coordinate names.
- `free_coordinates`: unique ordered optimized names within coordinate_order.
- `model_sha`: the native plant's exact definition identity.
- `coefficient_sha256`: verified SHA256 of little-endian float64 coefficient
  bytes, with sha256: prefix.

Construction copies arrays to tuples, validates shape and identity, and rejects
a coefficient hash mismatch. `from_coefficients` explicitly computes a new
snapshot identity. `to_record` returns detached JSON lists;
`from_record` requires exactly the six declared fields and never silently
repairs a malformed declaration.

Physical packing is all knot positions followed by all knot velocities,
knot-major and free-coordinate-minor, matching CubicHermiteSplineTrajectory.
The coefficient hash alone does not bind camera, frame identity, coordinate
units, observations, or priors; their existing immutable parent provenance
remains required. The native definition identity differs from the XML resource
hash and must not be substituted for it.

## Solver and Pure Evaluation

Both public functions accept optional sixth parameter `initial_spline`:

```python
fit_image_trajectory(native, attachments, camera, inputs, config, initial_spline)
initialize_image_trajectory(native, attachments, camera, inputs, config, initial_spline)
```

Preserved mode requires strict initialization policy, exact compiled native
model identity, exact native and free coordinate orders, exact input knot grid,
and a source evidence interval equal to the spline's first/last knots. Existing
hard coordinate bounds still validate the unmodified candidate; an infeasible
start fails rather than being projected. The shared preparation path skips
sample interpolation, slope reconstruction, and authored initialization.

Original observed pixels/confidence and explicit current seed/prior targets
remain unchanged. Initial image RMS measures the exact start against original
observations. The canonical MAP solve remains the only optimizer. A strict
preserved pure evaluation is also available without optimization; it returns
optimizer_ran=false, convergence=false, and no authored initialization receipt.

`ImageFitResult.initial_spline` identifies the actual prepared initial
coefficients in every mode. Authored initialization remains a separate optional
receipt. A preserved restart cannot pretend that an authored projection ran.

## Owned Worker and Job Option

`NativeRefitOptions.initialization_source` is `sampled_parent` by default or
`preserved_spline` explicitly. Preserved mode requires operation=fit and strict
initialization policy. The requested knot_count must equal the saved count;
requested source evidence must include the full saved source interval.

The worker reconstructs the saved typed snapshot, validates it against the
compiled native binding, and compares canonical spline poses with every saved
parent frame using its exact pts_ticks/timebase source clock. Numerical source
sample agreement uses rtol=1e-8 and atol=1e-10; coefficient and knot identities
themselves are preserved exactly. Locked coordinates remain the original
explicit job seed/prior, and consistency checks reject conflicting parent
samples.
Public workspace `preserved_fit_spline` is shared by the read-only plan and
worker. Legacy records without knots/coefficients are unavailable; declared
snapshots and partial malformed legacy records reject. Native compilation is
deferred to the worker, which independently validates the shared snapshot.

New saved original_fit evidence includes:

- `spline_start`: the final physical spline snapshot for future restarts.
- `initial_spline`: the actual optimizer start snapshot.
- `initial_coefficient_sha256`: its byte identity.
- `initialization_source`: the explicit job mode.
- `initialization`: an authored receipt only when that separate policy ran.

The existing matching scheduler, source/runtime stamps, cancellation guard,
new immutable version identity, and rejected research qualification remain in
use. No existing parent fit is overwritten.

## Repeatable Validation

```powershell
python3 -m pytest tests/unit/motion_matching/test_historical_spline_restart.py tests/unit/motion_matching/test_historical_image_fit.py tests/unit/workspace/test_necromatcher_fit_worker.py tests/unit/workspace/test_necromatcher_fit_jobs.py -q --disable-warnings
pre-commit run mypy --hook-stage pre-push --files src/shared/python/motion_matching/historical_fit/contracts.py src/shared/python/motion_matching/historical_fit/solver.py src/shared/python/workspace/necromatcher_fit_worker.py src/shared/python/workspace/necromatcher_fit_jobs.py
python3 scripts/check_document_title_case.py docs/development/necromatcher-preserved-spline-restart.md
```

The first restart collection failed because ImageSplineStart was absent. Worker
and job tests then failed because the mode and reconstruction seam were absent.
Regression tests trap resampling and authored initialization calls, inspect the
exact coefficient vector handed to the canonical optimizer, and independently
compare q, velocity, and acceleration before/after a wiring-only solve.
Model/order/hash/knot-clock incompatibilities fail explicitly. Worker tests
also reject parent poses inconsistent with the saved spline and verify unchanged
pixel evidence/current prior seed. Wiring-only tests do not establish optimizer
convergence or scientific acceptance.
Final combined validation passed 58 tests across restart, existing image-fit,
worker, and job suites; scoped Ruff check/format, the pinned pre-push mypy hook,
and the document title audit passed. Source files remain below 1200 lines and
all modified functions remain below 100 lines and eight arguments. No actual
user-library restart or scientific acceptance was produced by these tests.

## Remaining Qualification

Run separate source-stamped preserved restarts on the actual bounded Tiger/Hogan
versions, then independently reassess dense original PTS, adjacent midpoints,
coordinate extrema, image residuals, grip closure, and authored heel contacts.
Retain continuous_certified=false for nonlinear constraints and all historical
qualification blockers. Design-manual release remains blocked pending its
existing inventory and approval requirements.
