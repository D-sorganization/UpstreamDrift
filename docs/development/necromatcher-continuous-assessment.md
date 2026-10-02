# Necromatcher Continuous Spline Assessment

## Scoped Plan

Issue #11235 will gain a standalone, opt-in assessment of the stored historical
fit spline. This step changes no optimizer, native provider, fit contracts,
capture observations, control profile, or acceptance decision. The canonical
engineering manual remains separate and release governance remains
`blocked-inventory-required`.

The public helper in `motion_matching/historical_fit/assessment.py` will consume
an existing `ImageFitResult` and a mapping of named finite authored bounds in
those coordinates' native units. It will reuse the public
`CubicHermiteSplineTrajectory.unpack` and `evaluate` contracts. Locked coordinates
retain the constant values used by `ImageFitResult.evaluate_source_times`.

For each segment, normalized time `u` spans `[0, 1]` and the coordinate is
`q(u) = A*u^3 + B*u^2 + C*u + D`. With segment duration `h`,
`A = 2*q0 - 2*q1 + h*(v0 + v1)`,
`B = -3*q0 + 3*q1 - h*(2*v0 + v1)`, and `C = h*v0`.
Candidate extrema are both endpoints and real interior roots of
`3*A*u^2 + 2*B*u + C`. Actual candidate values will be evaluated by the canonical
trajectory. Nonuniform source intervals and lower-degree segments are retained.
Equal extrema use the earliest source time deterministically.

Stored free-coordinate samples must agree with the public spline evaluation
using absolute tolerance `1e-10` in coordinate-native units and relative
tolerance `1e-8`. Scalar evaluation batches bound Jacobian allocation during this
validation. Locked coordinate samples must be exactly constant; empty free
coordinate sets explicitly fail because the public spline requires positive DOF.
The assessment records both source-consistency tolerances.

Coordinate bounds may have equal endpoints, representing an authored locked
limit. Both endpoints must be finite; one-sided or infinite bounds explicitly
fail. Adapters must preserve such constraints separately and identify these
coordinates as outside this helper's finite two-sided assessment domain. Source
intervals require strictly increasing endpoints. Non-string coordinate keys and
boolean limits are rejected.

Frozen typed DTOs will report each coordinate's minimum and maximum, their
source times, optional authored bounds, and violation magnitudes. The aggregate
will identify bounded, unbounded, and violating coordinates. Bounds are authored
assumptions; unknown names and invalid bounds fail explicitly. The record will
state `continuous_certified=False` and grip/ground `not_assessed`: coordinate
extrema do not prove nonlinear native closure or contact feasibility.

## Red-First Validation

Tests will first fail against the absent helper. Cases will include an interior
overshoot despite bounded endpoints, nonuniform timing, cubic and degenerate
linear/quadratic/constant segments, multiple segments, locked and unbounded
coordinates, invalid identities/shapes/bounds, and immutable outputs. Independent
closed-form polynomial expectations will check extrema and worst times. Original
image observations, RMS, confidence counts, and fit arrays must remain unchanged.
Scoped Ruff, pinned mypy, title case, and file/function budgets will be checked.

## Implemented Boundary and Local Validation

Import `assess_spline_bounds`, `CoordinateExtrema`, and `SplineBoundsAssessment`
from `src.shared.python.motion_matching.historical_fit.assessment`. The helper
returns immutable scalar/tuple DTOs and changes no image-fit arrays or metrics.

```python
assessment = assess_spline_bounds(existing_image_fit, authored_native_unit_bounds)
```

The actual extrema come from the preserved spline coefficients. Source samples
serve as a consistency check, not a replacement interpolant. Locked coordinates
remain constant. Equal extrema use the earliest source time. Violation magnitudes
remain per coordinate; the helper does not compare translation and rotation units
or turn unbounded coordinates into a passed bounds check.

The initial 17 tests failed against the missing helper. Subsequent red tests
exposed missing source/spline agreement, empty-free-coordinate rejection,
degenerate authored limits, non-string aliases, and DTO name validation. The
final assessment and existing historical image-fit regression suite passes
37 tests (26 assessment cases and 11 image-fit cases). Scoped Ruff and pinned
mypy pass, and this document passes the title-case check. These are local software
checks, not scientific acceptance or remote CI evidence.

No optimizer probes, nonlinear closure/ground assessment, physical-clock
qualification, new library fit, or downstream handoff is implemented by this
step. Issue #11235 continues to own those remaining scientific and integration
boundaries. The canonical engineering design manual remains separate and its
release gate remains `blocked-inventory-required`.
