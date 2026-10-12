# Analytic Marker and Fixed-Camera Derivatives

## Scope and Public Boundaries

Issue #11394 reuses existing native point derivatives in the historical image
fit. It introduces no solver, sparse matrix mode, cache, shaft derivative,
calibration, anatomy measurement or physical qualification. Requested budgets
and historical records remain unchanged; old unavailable telemetry is not
backfilled. New solves use the #11388 actual counters and scoped durations.

`motion_matching.marker_kinematics` defines the SDK-free frozen
`MarkerLinearization` and separate optional `MarkerLinearizationPlant` and
`MarkerLinearizer` protocols. The pipeline facade exports these contracts.
The required `MatchingPlant` protocol is unchanged. The DTO copies read-only
world-metre positions and point Jacobians, retaining exact marker labels and
native scalar coordinate order. Shapes are `(markers,3)` and
`(markers,3,coordinates)`; finite numeric data and unique nonempty names are
required.

The optional plant factory `create_marker_linearizer(attachments)` returns a
fit-owned public resource. MuJoCo reuses its existing `create_ik` adapter and
analytic `mj_jac` implementation. Its public `marker_linearization(q)` owns
native state updates and private kinematics helpers; historical consumers do
not inspect SDK state. An existing constraint adapter is shared when it exposes
the marker protocol; otherwise the explicit marker factory provides the resource.
Providers without
the factory, or explicitly raising `NotImplementedError`, retain finite
differences. A declared malformed record, shape or identity fails closed.

## Exact Dense Jacobian Composition

The canonical fixed `CameraProjection.project_jacobian(points)` returns
`(points,2,3)` derivatives with respect to world metres. It retains `project()`
depth validation and the same admitted intrinsics, including skew and cross
terms. With camera point `c = R p + t`, each pixel row differentiates
`(K[row,0] c_x + K[row,1] c_y) / c_z + K[row,2]`, then multiplies by `R`.
This is a derivative of the saved projection, not a second camera model.

The body-image Jacobian composes that derivative with native point derivatives,
the free coordinate selection, square-root confidence and exact Hermite basis.
MAP retains its existing hard-bound decoding chain. Residual evaluation and
row order are unchanged: source-only image/prior/speed rows; configured native
grip/ground rows at the existing source/interior/phase union; then optional
additional image rows. Generic shaft terms continue canonical central pose
differences. Legacy position-only closure derivatives also retain finite
differences while the body block can use analytic derivatives.

The dense TRF solver, tolerances, objective, source observations, camera,
coordinate scales, prior seed, constraint schedule, probe grid, nonlinear
branches and hard-bound policy are unchanged. Exact residual parity is distinct
from derivative parity within numerical tolerance: replacing central differences
can change optimization iterates. A future historical comparison requires its
own frozen source/options/inputs, actual telemetry and independent assessment.

## Red-First Validation and Repeatability

Use SDK-free fixtures to test immutable copies, strict shapes/order, malformed
providers, optional fallback, exact residual parity, zero weights, reversed free
coordinate selection and one factory plus one linearization per source pose.
Compare camera derivatives to independent world-point central differences with
nonidentity rotation/translation and nonstandard intrinsics. Compare the complete
bounded Hermite chain with interior constraints to independent internal-decision
differences; preserve existing constraint, shaft and MAP regressions.

Use only the repository synthetic MuJoCo fixture for local offset and nonidentity
native-coordinate derivatives versus independent FK differences. Do not open
historical libraries or run optimizers for the historical players as a software
test. Run focused suites, scoped Ruff/format, pinned pre-push mypy, architecture
budgets, title checks and design-manual governance. Manual release remains
blocked pending the governed calculation inventory.

Deterministic provider call-count assertions establish the algorithmic seam;
they do not measure wall-time speedup. Sparse LSMR, analytic shaft derivatives,
constraint caches and physical convergence remain separately scoped work.
