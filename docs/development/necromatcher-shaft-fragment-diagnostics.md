# Raw Shaft Fragment Derivative Diagnostics

## Public Core and Scope

Reuse the curated `motion_matching.historical_fit` exports
`FragmentDerivativeOptions`, `FragmentDerivativeCheck`,
`FragmentDerivativeAssessment`, `assess_fragment_derivatives` and
`optional_fragment_derivatives`. These pure diagnostics reuse canonical
`CameraProjection.project` / `project_jacobian`, `ShaftAxisSegment` and the
public `MarkerLinearizer` / immutable `MarkerLinearization`. They implement no
FK, solver, renderer, scheduler or library persistence.

A caller must independently bind original PNG/full capture clock, camera,
native model and authored shaft identity through canonical workspace admission.
Typed DTOs alone authenticate none of these. Create a single public marker
resource for the exact two authored shaft attachments and pass its native order
and scalar units. Preserve each source-bound role separately; the old training
and diagnostic-seen Hogan 375 fragments are distinct records, not pooled labels.
Confidence/sigma remain authored assumptions and do not weight this raw check.

## Options and Availability

Options require exact named native order, units, selected radian coordinates,
positive unique FD steps, one explicit finite two-sided bound or None per native
coordinate, and two ordered shaft marker labels. Equal bounds are valid; no
implicit missing-bound assumption is made. Translational or mixed selected
units are outside this first slice. Steps are in radians and derivative units
are px/rad. Native scalar bounds do not constitute clinical coupled ROM.

Available results contain two raw signed perpendicular pixel distances and the
exact rectangular selected-coordinate/step check order. Each immutable check
retains analytic and central derivative pairs with their actual maximum
absolute difference. Bound-limited central values/error are None; poses are
never clipped and one-sided differences are never substituted. Projected
base-axis degeneracy, absent marker capability and explicit abstention have
unavailable results without fabricated numeric rows. Perturbed degeneracy,
identity/order mismatch or nonpositive camera depth fails the operation.

The line-normal derivative is the narrow algebra extracted from reviewed
local-axis diagnostics; existing public authored-line projection has no
Jacobian. Projection and world marker derivatives remain owned by their
canonical providers. The explicit 1e-8-pixel degeneracy threshold follows that
diagnostic convention and is not a calibrated localization uncertainty.

## Repeatable Native-Free Verification

```powershell
python3 -m pytest -o addopts='' tests/unit/motion_matching/test_shaft_fragment_diagnostics.py tests/unit/motion_matching/test_shaft_residuals.py tests/unit/motion_matching/test_shaft_observations.py tests/unit/motion_matching/test_shaft_geometry.py tests/unit/motion_matching/test_shaft_row_timing.py -q
pre-commit run mypy --hook-stage pre-push --files src/shared/python/motion_matching/historical_fit/shaft_fragment_diagnostics.py src/shared/python/motion_matching/historical_fit/__init__.py tests/unit/motion_matching/test_shaft_fragment_diagnostics.py
python3 -m scripts.check_design_manual_governance
```

The focused tests use synthetic rotated-perspective marker providers and an
independent projection/distance calculation at three steps. They cover zero
response, immutable/nested DTOs, strict shapes/bools/nonfinite values, order,
bounds/nulls, depth, unsupported capability, explicit abstention and disabled
no-provider behavior. A cold process verifies no MuJoCo, Drake, Pinocchio or
OpenSim import. Execution fingerprints already include the historical-fit
source directory; the regression verifies inclusion of this module's bytes.

This integration is pure source/test work, not a new historical diagnostic or
motion fit. It changes no objective rows, body RMS denominator, geometry penalty
mass, evidence roles, model, camera, saved curve or acceptance status. Future
workspace/job/native/web integration remains separate and must rebind sources,
reuse existing jobs/renderers/stores and preserve null/raw-unit semantics.

## Preserved Methods and Scientific Limits

See the standalone [Local Axis and Fragment Methods Source](necromatcher-shaft-fragment-methods.tex)
and [Its Review](necromatcher-shaft-fragment-methods-review.json), plus the
[Original-Start Budget Results Source](necromatcher-analytic-marker-budget-results.tex)
and [Its Review](necromatcher-analytic-marker-budget-results-review.json).
These methods/results documents do not become the canonical engineering manual.
Manual release remains blocked pending inventory, freshness and approvals.
Numerical derivative agreement is neither historical camera/physical-clock
calibration nor clinical ROM, anatomical, continuous-motion or reconstruction
qualification. No historical optimizer or native render ran for this change.
