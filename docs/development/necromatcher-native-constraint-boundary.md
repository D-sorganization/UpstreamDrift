# Necromatcher Native Constraint Boundary

## Scope and Authority

Issue #11235 exposes existing native kinematic constraints through one public
engine-independent boundary. It does not integrate the historical solver,
qualify calibration or physical timing, infer measured contacts, or establish
scientific acceptance. Existing MuJoCo finite-SO(3) grip and ground geometry
remain the computational authorities. Design-manual governance passes with
release still blocked by the required calculation inventory.

## Public Contract

`BaseFullBodyIK.constraint_residual_jacobian(q, options)` returns a frozen
`ConstraintLinearization` with `residual`, `jacobian`, `coordinate_order` and
`row_labels`. Unsupported adapters explicitly raise `NotImplementedError`.
The supported MuJoCo marker adapter supplies a pose Jacobian in its exact
native scalar coordinate order, without exposing SDK state to consumers.

`ConstraintOptions` requires a `GroundPlane`, three nonnegative finite
weights (`position_weight`, `rotation_weight`, `ground_weight`) and three
positive finite scales (`position_scale_m`, `rotation_scale_rad`,
`ground_scale_m`); `pinned_spheres` defaults to an empty tuple. Scales and
weights are declared modeling choices, not measured uncertainty.

Rows always comprise grip position world X/Y/Z (metres), principal grip
rotation world X/Y/Z (radians), then signed ground-distance penalties for
configured native contact spheres in their declared order. Each block is
multiplied by sqrt(weight)/scale; returned residuals are dimensionless.
Zero weights retain zero rows and zero Jacobians. Ground rows are fixed
across penetration sign changes; unpinned above-ground rows are zero and
pinned rows retain signed distance. These are model contact-sphere identities,
not evidence of actual human foot contact. Unknown pinned identities fail.

The principal SO(3) logarithm is discontinuous at a half turn. Analytic
Jacobians are tested away from that branch and ground activation boundaries;
no differentiability or uniqueness claim is made at those boundaries.

## Test-Driven Delivery

Write public-seam tests before implementation: native six-dimensional grip
and ground Jacobians against centered finite differences; finite pose/options
validation; zero weights; fixed ground rows; unsupported base capability and
unknown contact identities. Reuse the existing native marker test fixture.
Then expose validated DTOs, add the base default failure and the native wrapper
around `_append_closure`/`_append_ground`, and run focused tests, Ruff and the
pinned mypy configuration. No Git operations or solver changes are in scope.

## Verification

The initial seven public-capability tests failed before the DTO module existed
and passed after implementation. The complete native marker suite passes 33
tests, covering finite-difference pose derivatives above and below ground,
separate position/rotation/ground scaling, pinned positive depth, zero weights,
unknown contact identities, malformed/finite boundary data, defensive copies
and explicit unsupported/empty-geometry failures. Scoped Ruff checking and
format checking pass, as does the repository pre-push mypy hook. Existing
position-only solving and trajectory tests remain green.
