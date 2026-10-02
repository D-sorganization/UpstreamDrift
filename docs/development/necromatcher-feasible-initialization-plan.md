# Necromatcher Feasible Initialization Plan

## Scope and Current Evidence

This is the next implementation step under #11235 and parent #11232, following committed bounds implementation 73ace83145. Both saved v7 fits violate authored ranges; the new strict domain correctly rejects infeasible starting coefficients. No new bounded historical trial has run, and no seed or fitted output has been silently clipped.

## Bound Authority

The exact bound model definition contains `coordinate_ranges_deg`. Its exported MuJoCo scalar joints use `limited="false"` and do not carry these numeric ranges. A new public NativeFitBinding.authored_coordinate_bounds capability must reproduce and verify the exact XML hash, match each authored range to exported scalar hinge identity and compiled radian units, convert finite degrees to radians and return immutable bounds with definition/XML hashes. Its receipt must identify `range_source=bound_native_definition.coordinate_ranges_deg` and `compiled_limits_enforced=false`. Missing ranges remain explicitly unbounded; malformed or incompatible declarations fail. These are authored hypotheses, not measured anatomical limits or existing native dynamics enforcement.

## Explicit Initialization Policy

Add a reusable pure initializer operating on canonical physical Hermite coefficients and HermiteBoundsDomain. The opt-in policy projects bounded free knot positions into their authored intervals and sets free knot velocities to zero, making the initial spline Bernstein-admissible. Retain unbounded positions and report all modified knot/coordinate identities, maximum displacement and original/new coefficient hashes. Locked positions outside their range must fail. Input arrays and finished optimizer results must remain unchanged.

ImageFitConfig.initialization_policy should default to strict rejection. Keep original ImageFitInputs.seed and its prior semantics; change only optimizer initialization under explicit authoring. initial_rms_pixels must measure the actual feasible initial spline against original capture observations. Named hard limits continue to apply throughout optimization. Kink derivatives and nonlinear contact/closure limitations remain as documented in the Hermite bounds procedure.

## Persistent Seed and Trial Evidence

Preserve each authored seed through library.add_fit as a new source-bound research version with exact model/capture parents, original frame identities/PTS, canonical coefficients, fresh real-observation projection RMS, initializer provenance and change diagnostics. Record optimizer_ran=false and an explicit authored initialization status; do not copy old convergence or RMS claims. Heel planting remains a separate explicit contact hypothesis.

Run Tiger and Hogan bounded jobs from committed source, then independently audit preserved-spline extrema, every observed image frame and midpoint grip/contact samples. Preserve failures and new source-sized overlays in the Desktop review package; extend the compiled methods report with actual trial receipts. Unknown camera/anatomy/physical time, controls, fresh replay and downstream simulation/impact qualification remain open.

## Red-First Acceptance Tests

- Default infeasible seed rejection and explicit initialization feasibility.
- Locked-range conflict, nonuniform clocks and copied immutable inputs.
- Unchanged original prior and independently measured initialized image RMS.
- Exact native XML/model/name/unit verification and authored range provenance.
- New-version parent/frame identities and explicit absence of optimization.
- No initialization policy modifying a completed result.

Implement through existing canonical estimation, binding, library and job boundaries. Do not add another optimizer or scheduler. Refresh DL-#11235 in the same implementation commit, preserve one SPEC row and the manual inventory release blocker, and retain scientific rejection until independent acceptance evidence exists.
