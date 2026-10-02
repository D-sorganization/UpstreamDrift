# Necromatcher Authored Feasible Initialization

## Scope and Interpretation

Issue #11235 adds the pure public helper `initialize_authored_hermite` in
`src/shared/python/estimation/hermite_initialization.py`. It constructs an
explicitly authored optimizer start. It neither modifies a finished fitted
trajectory nor certifies historical accuracy, contact, physical source time,
or dynamics. This procedure is separate from the canonical QMD engineering
design manual, whose inventory release block remains in force.

The default strict rejection of an infeasible candidate remains the caller's
responsibility. A caller invokes this helper only under an explicit authored
initialization policy. Native definition/XML hash verification, coordinate
name and unit binding, and rejection of locked native coordinates outside
authored bounds belong to the surrounding native fitting workflow.

## Public API and Immutable Receipt

```python
initialize_authored_hermite(
    domain: HermiteBoundsDomain,
    coefficients: np.ndarray,
    coordinate_names: tuple[str, ...],
) -> AuthoredHermiteInitialization
```

The input domain contains the free physical coordinate subset, in the same
order as the unique nonempty immutable coordinate names. Coefficients retain
the canonical CubicHermiteSplineTrajectory packing: all knot positions followed
by all knot velocities, each knot-major and coordinate-minor. A finite real
numeric one-dimensional vector of exactly the domain's physical_size is
required. Input arrays are copied and remain unchanged.

For each bounded coordinate, knot positions are explicitly projected into its
authored finite range. Equal endpoints produce that constant position. Positions
of unbounded coordinates remain unchanged. ALL domain knot velocities become
zero, including unbounded coordinates. This policy is
`authored_range_project_zero_slopes`; it does not estimate velocities from
footage. The public bounds domain verifies feasibility of the initialized
coefficients as a postcondition. No optimizer or native engine is called.

The returned frozen DTO contains immutable coefficient and receipt tuples:

- `coefficients`: initialized physical q/v coefficients.
- `changes`: one AuthoredKnotChange for every knot/coordinate whose position or
  velocity changed, including original/new values, knot index, native coordinate
  name, and source time. Unchanged knots are omitted.
- `maximum_position_displacements`: one CoordinateDisplacement per domain
  coordinate, preserving order, with its maximum absolute knot-position change.
  Values remain in that coordinate's native units; no mixed m/rad aggregate is
  reported. Unrepresentable displacement magnitudes fail explicitly.
- `original_coefficient_sha256` and `initialized_coefficient_sha256`: SHA256
  strings with `sha256:` prefix over canonical little-endian float64 coefficient
  bytes. Hashes preserve coefficient packing and ignore input byte order. The
  parent model, coordinate names/units, knot times, source fit, and initialization
  policy must be stamped separately; coefficient hashes alone do not bind them.

Convert `result.coefficients` using `np.asarray` when passing it to the
canonical estimator. The result owns immutable copies and does not alias input
arrays. Original observed pixels/confidence, physical-time qualification, and
prior targets remain unchanged by this helper.

## Integration and Source-Bound Publication

The canonical image-fit path must explicitly choose this policy before the
bounded MAP solve and must measure initial image RMS from the initialized
coefficients against original capture evidence. Preserve original seed/prior
semantics rather than silently substituting the projected knot positions into
the prior. Persist original fit ID/hash, exact model/capture/frame/PTS identities,
the range source, policy, coefficient hashes, and change receipt with any new
research seed version. Measure new reprojection evidence rather than copying
the parent's RMS or convergence status. An initializer-only version must say
that no optimizer ran and retain monocular research qualification.

The current native exporter sets scalar joints to limited=false and exports no
joint ranges. Consequently, authored ranges extracted from the hash-bound
definition are model hypotheses, not compiled MuJoCo limit enforcement. Their
names/types/units still require independent public native binding checks.
Planted heel contact remains a separate authored hypothesis; bounded coordinates
and zero penetration do not establish it.

## Repeatable Validation

```powershell
python3 -m pytest tests/unit/estimation/test_hermite_initialization.py tests/unit/estimation/test_hermite_bounds_domain.py -q --disable-warnings
python3 -m ruff check src/shared/python/estimation/hermite_initialization.py tests/unit/estimation/test_hermite_initialization.py
python3 -m ruff format --check src/shared/python/estimation/hermite_initialization.py tests/unit/estimation/test_hermite_initialization.py
pre-commit run mypy --hook-stage pre-push --files src/shared/python/estimation/hermite_initialization.py tests/unit/estimation/test_hermite_initialization.py
python3 scripts/check_document_title_case.py docs/development/necromatcher-authored-initialization.md
```

The first scoped test collection failed because the public module did not
exist. Implementation then passed the initial thirteen tests. The suite checks
independent canonical spline feasibility on nonuniform times, exact change
receipts, immutable outputs/input preservation, deterministic endian-independent
hashes, fixed/equal-bound and all-fixed domains, zeroing of unbounded velocities,
malformed vectors/names, and unrepresentable displacement rejection.
Final scoped validation passed 15 initializer cases and 23 bounds-domain cases
(38 total), Ruff check/format, the pinned pre-push mypy hook, and the document
title audit with zero violations. The initializer module is 154 lines; its
largest function is 41 lines and has at most four arguments.

## Remaining Qualification

Integrate the opt-in initializer, verified native range extraction, and fresh
image metrics through existing refit jobs. Run actual bounded Tiger/Hogan trials
without rewriting previous fits, then assess original source PTS, adjacent
midpoints, polynomial coordinate extrema, grip, ground, and image fidelity.
Keep nonlinear continuous_certified=false and avoid motion acceptance until
the separately tracked scientific conditions are met.

## Owned Job Operation

`NativeRefitOptions.operation` defaults to `fit`; `author_initialization` is
an explicit alternative using the same MatchingJobService, source stamps,
owned subprocess, cancellation, and immutable new-version publication path.
It requires ImageFitConfig.initialization_policy equal to
`authored_range_project_zero_slopes`. The worker requires configured limits to
match ALL exact bounds returned by NativeFitBinding.authored_coordinate_bounds;
custom limits cannot masquerade as exact native authored ranges. Ordinary fit
operations retain explicit custom-limit support and label that provenance.

The author worker calls public `initialize_image_trajectory` without invoking
the optimizer. A contradictory result (optimizer_ran=true, converged=true, or
missing initialization receipt) is rejected. Stored evidence includes the
initialization receipt, fresh selected/dense/held-out image RMS, explicit
optimizer_ran=false and converged=false, and authored_initialization_only.
The optimizer_not_converged blocker applies only when an optimizer actually
ran. Bounded research versions retain unknown physical clock/camera, unreplayed
dynamics, and uncertified nonlinear continuous constraints. Source fits remain
unchanged, and failed/cancelled jobs prevent publication.
All versions retain historical_anatomy_unqualified. Custom or partial limits
retain native_authored_ranges_not_fully_enforced; only an exact full verified
authored-range receipt clears that coverage blocker. Coordinates lacking
authored ranges remain explicitly reported and blocked as unbounded. Enforcing
generic authored limits does not qualify historical anatomy.

Repeat the focused worker/job checks with:

```powershell
python3 -m pytest tests/unit/workspace/test_necromatcher_fit_worker.py tests/unit/workspace/test_necromatcher_fit_jobs.py -q --disable-warnings
```

Seven new boundary tests initially failed because operation dispatch, receipt
fields, and range provenance were absent. A separate contradictory-receipt
regression failed before its guard was added. The owned publication integration
test uses a stub computation to isolate scheduler/storage behavior; it verifies
a separate rejected research version and original-byte preservation, not a
native scientific trial. Real bounded Tiger/Hogan trials remain independent
qualification work.
Final focused worker/job validation passed thirteen tests, Ruff check/format,
the pinned pre-push mypy hook, and the document title audit. Two additional
blocker regressions were red before exact-range coverage and anatomy guards
were added. Worker source is 340 lines with a largest function of 87 lines;
job source is 290 lines with a largest function of 92 lines. Neither exceeds
the eight-argument function budget.
