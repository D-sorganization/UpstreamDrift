# Necromatcher Hermite Bounds Domain

## Scope and Status

Issue #11235 adds a pure public coefficient-domain adapter in
`src/shared/python/estimation/hermite_bounds.py`. This procedure describes the
adapter, not the canonical engineering design manual. Estimator integration,
native range extraction, and actual historical trials remain separate work.
No motion acceptance or nonlinear continuous certification follows from this
coordinate-domain implementation. The design-manual inventory release block
remains in force.

## Public Contract

`HermiteBoundsDomain(knot_times, coordinate_bounds)` copies construction inputs
to immutable tuples. Bounds are finite pairs in physical coordinate units;
`None` explicitly means unbounded. Coordinate identity and unit conversion are
the caller's responsibility. One-sided infinite bounds are rejected rather than
silently discarded. Times are finite and strictly increasing. Durations,
reciprocals, bound spans, and bounded velocity widths must be representable.

The public properties are `knot_times`, `coordinate_bounds`, `n_knots`, `n_dof`,
`physical_size`, and `decision_size`. Public methods are `encode`, `decode`,
`decode_jacobian`, and `decision_bounds`.

Physical coefficients retain the canonical CubicHermiteSplineTrajectory order:
all knot positions followed by all knot velocities, each knot-major and
coordinate-minor. Decisions retain that ordering while eliminating coordinates
whose lower and upper limits are equal. For bounded nonfixed coordinates,
velocity decisions are normalized slopes in [-1, 1]. For unbounded coordinates,
both decisions remain ordinary physical coefficients.

Equal bounds decode to constant position L and zero velocity, with no free
columns. An entirely fixed domain has zero decisions and a physical-by-zero
Jacobian. The estimator integration must handle that case without calling
SciPy with equal bounds or an empty decision vector.

## Whole-Segment Bound Construction

For a segment with duration h, the cubic Bernstein controls are q0,
q0+h*v0/3, q1-h*v1/3, and q1. All four controls inside [L,U] guarantee the
entire coordinate curve stays inside the interval by the convex hull property.
This sufficient condition is conservative: some bounded cubics have controls
outside their coordinate limits and are excluded.

At each knot the outgoing segment supplies the velocity interval
[3*(L-q)/h_out, 3*(U-q)/h_out]. The incoming segment supplies
[3*(q-U)/h_in, 3*(q-L)/h_in]. The decoder intersects all incident intervals,
using lo=max(lower endpoints) and hi=min(upper endpoints), then sets
v=(1-theta)*lo+theta*hi, theta=(w+1)/2. End knots use their sole incident
segment. A single shared decoded velocity at each knot preserves C1 continuity
on nonuniform knot grids. No clipping or second optimization solver occurs.

`encode` requires a feasible physical candidate. It rejects positions or
velocities outside the domain and rejects nonzero fixed-coordinate velocities.
It never silently repairs an existing historical fit. A caller may explicitly
construct a feasible initial guess with bounded positions and zero velocities;
that is an initialization assumption, not a measured observation or accepted
output. At a collapsed interior velocity interval, zero velocity encodes with
w=0; normalized slopes are nonunique there.

## Differential Contract

The decoder Jacobian has physical coefficient rows and internal decision
columns. Position rows are identity entries; fixed rows are zero. Unbounded
velocity rows are identity entries. Bounded velocity derivatives are
dv/dw=(hi-lo)/2 and dv/dq=(1-theta)*dlo/dq+theta*dhi/dq.

Active branch slopes are +/-3/h. At an exact active-branch tie, the adapter
returns their mean as a symmetric generalized derivative. The decoder is not
classically differentiable there: one-sided derivatives can differ. Tests
independently verify both one-sided limits and the symmetric central derivative.
This contract does not imply trust-region optimizer convergence.

Estimator integration must preserve the callback's physical MapDecisionLayout.
For physical coefficient Jacobian Jc, decoder Jacobian D, and shared parameter
Jacobian Jp, the internal solver Jacobian is [Jc*D, Jp]. Shared parameter locks,
identifiability gates, and prior rows must retain their existing semantics.
Saved estimator coefficients must be decoded physical coefficients so existing
trajectory evaluation and replay remain valid.

## Verification and Repeatability

Run from the native-fit worktree:

```powershell
python3 -m pytest tests/unit/estimation/test_hermite_bounds_domain.py -q --disable-warnings
python3 -m ruff check src/shared/python/estimation/hermite_bounds.py tests/unit/estimation/test_hermite_bounds_domain.py
python3 -m ruff format --check src/shared/python/estimation/hermite_bounds.py tests/unit/estimation/test_hermite_bounds_domain.py
pre-commit run mypy --hook-stage pre-push --files src/shared/python/estimation/hermite_bounds.py tests/unit/estimation/test_hermite_bounds_domain.py
python3 scripts/check_document_title_case.py docs/development/necromatcher-hermite-bounds.md
```

The first scoped collection failed because the new module was absent. After
implementation, the initial 19 tests passed. Two representability tests then
failed because overflowing finite endpoint differences were accepted; explicit
constructor guards corrected that behavior. A large but representable feasible
seed independently exposed normalization overflow; division before scaling
corrected it without clipping. The final suite verifies independent Bernstein
controls and polynomial stationary extrema, canonical physical packing, C1
velocities, nonuniform intervals, mixed unbounded/fixed coordinates, immutable
input copies, infeasible seed rejection, and numerical Jacobian comparisons.
Final scoped validation: 23 tests passed; Ruff check and format check passed;
the pinned pre-push mypy hook passed; the document title audit found zero
violations. Design-manual governance verified two QMD sources and zero
registered calculations, retaining release=blocked-inventory-required.

## Remaining Qualification

Integrate through the canonical MAP solver with explicit opt-in behavior and
independent complete-objective Jacobian tests. Extract finite authored native
ranges with verified coordinate names and units. Reassess historical outputs
against exact cubic extrema and dense image residuals. Nonlinear grip closure,
planted contact, dynamics, monocular depth, camera assumptions, and physical
time remain separate qualification obligations. Keep continuous_certified=false
for nonlinear constraints and do not infer a scientific success from this
coordinate bound alone.

## Canonical Estimator and Historical Fit Integration

`MapEstimatorOptions.trajectory_domain` opts the single-trial solver into this domain. The physical callback layout remains unchanged; analytic Jacobians are chained through the decoder, and shared-parameter prior rows retain their independent columns. A fully fixed problem evaluates its objective without calling SciPy with zero decisions. Multi-trial MAP explicitly rejects this single-trial capability instead of ignoring it. Default fitting keeps its identity coefficient domain.

`ImageFitConfig.coordinate_bounds` contains immutable `(native_name, lower, upper)` tuples; JSON transport uses arrays decoded by `from_record`. Unknown names, duplicate names, unsupported endpoints, inconsistent locked coordinates and infeasible initial Bernstein controls fail before optimization. Limits are authored in the model's named native units. Unspecified coordinates remain unbounded. No v7 trajectory is silently clipped, repaired or retroactively accepted.

Red-first integration tests cover configuration transport, actual native image targets outside an authored range, locked/unknown names, adversarial canonical targets, fixed/shared decisions and an independent complete-objective Jacobian with shared priors. Original source observations and image point counts remain unchanged. New historical trial initialization must be explicitly authored feasible; coefficient-domain certification alone does not qualify contact, closure, timing, camera, anatomy or dynamics.
