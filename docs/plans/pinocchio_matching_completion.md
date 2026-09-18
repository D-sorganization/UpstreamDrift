# Pinocchio Matching Completion and Engine Transfer

Governing epic: #10430; program: #10363. Updated 2026-09-18.
Continuation state belongs in `docs/development/HANDOFF.md`.

## Verified State and Completion Boundary

The reviewed implementation baseline is f15dc23c0, with PF-01/02/04 stacked
in PRs #10442/#10445/#10447. PF-03 is separate PR #10443. Those components
are not an integrated, physically accepted full-swing solution. Driver and
iron evidence remains rejected. Do not overwrite it or relabel its playback.
PF-05 #10435 has another agent's active lease; preserve its ownership.

PF-06 now has `force_nullspace.py`: scaled rank-revealing SVD, independent
balance checks, constrained minimum-effort redistribution and exact circular
friction cones in declared surface frames. Nineteen unit tests include an
independent full-space QP comparison and impossible hard-zero trail effort.
This is a numerical kernel, not an integrated native fitting pipeline, temporal
optimizer, physiological validation or accepted replay. No native Pinocchio
run has been performed for this change.

## Numerical Contract for Integration

At each unchanged native `(q, v, a)` assemble `A x = b`, where
`x = [actuator_efforts, ground_point_forces, grip_reaction_wrench]` and
`b = M(q) a + bias(q, v)`. Use engine-derived actuator selection, world-frame
contact Jacobians and the declared grip Jacobian. Preserve generalized effort
signs. Do not introduce floating-base actuators or an unrecorded slack block.

`ForceNullSpace.from_balance(A, b, variable_scale=..., row_scale=...)`
returns `x0` and `N` with audited `A x0 = b` and `A N = 0`. Choose scales
from documented torque/force capacities and separate equation force/moment
units. Inspect rank at every contact transition; nullity is not fixed.
`redistribute_forces` minimizes `||weights * (x - reference)||^2`.
Weights are inverse physical scales, not squared coefficients. Preserve both
`converged` and independently audited `feasible`; neither means accepted replay.

Supply finite actuator, contact and grip bounds. Set inactive contacts and
hard-zero trail entries to exactly zero bounds. Supply COP inequalities using
native contact locations and the support plane. A `FrictionCone` names three
force indices and an orthonormal frame whose columns are tangent, tangent,
outward normal. Its force entries must use that same world frame. An outer
square friction pyramid is inadmissible. No geometry is inferred from array
position or a presumed world Z direction.

For compliant contact, forces at fixed q/v are determined by the constitutive
law. Fix those forces in the bounds (or an explicitly justified uncertainty
interval); do not optimize arbitrary GRFs and pretend the compliant engine
will reproduce them. For constrained contact, verify acceleration-level
compatibility and use the same contact mode and constraint law in replay.
Internal grip force changes can alter joint compression even when net joint
torque falls. Do not make injury, metabolic or muscle-force claims from this
optimization.

PF-05 must smooth physical `x(t)` using actual capture timestamps. Never
penalize changes of null-space coordinates `z(t)`: SVD signs and dimensions
can change. The single-frame reference argument is not a substitute for that
trajectory optimization. Use the kernel to check selected trajectory frames
and explore constrained alternatives; retain the sparse physical-space solver
for the full-horizon production path until benchmarks justify a reduced one.

## Required Corrections Before Native Acceptance

1. PF-02: Bound/closure projection must preserve consistent q/v/a. Previously,
   clipping q to a constant left returned v at 2 rad/s. Differentiate the final
   manifold-consistent trajectory and verify closure at position, velocity and
   acceleration levels. Wire bounds and closure into the production matcher.
   Finite output alone is not solver convergence.
2. PF-03/04: Reject NaN/Inf at inputs and derived metrics. Missing contact
   geometry cannot imply friction/COP approval. Replace the outer-square
   friction constraints; independently verify the circular cone. Separate
   root force (N) and root moment (N m) limits. Never repair actuator effort
   after bounds have been checked.
3. PF-01: An unavailable strategy is absent with a reason, not a zero array.
   Store each torque strategy with its own ground/grip loads, root residual,
   contact schedule and audit. A paired-load roundtrip test must fail first.
4. Production wiring: Connect calibration, window refinement, smoothing,
   contact inference, trajectory allocation and candidate export in the real
   CLI. A unit-test-only implementation does not satisfy this milestone.

## Motion Interpretation Gates

Freeze source file hash, capture units, timebase, marker validity and capture
to-world rigid transform. Verify handedness with anatomical left/right labels,
not camera appearance. Check address, top, impact and finish against observed
event timestamps. Confirm shaft length, clubhead direction and grip attachment
frames separately for driver and iron. Reject reflections and improper rotation
matrices. Test known translated/rotated fixtures and degree/radian, cm/m errors.

Register models by coordinate name, kind, unit, axis, zero offset and sign.
Do not transfer a 44-coordinate array into a 41-coordinate engine. Verify
native FK marker positions for the same named poses before comparing dynamics.
Use each engine's configuration manifold when nq differs from nv; direct
coordinate interpolation or derivative subtraction is not universally valid.

## Bounded Packets for the Next Agent

Each packet needs a failing integration test before changes, explicit input
contracts, shared code reuse, one focused issue-linked commit and evidence.

| Packet               | Issue  | Concrete Work                                                                                              | Required Evidence                                                                                         |
| -------------------- | ------ | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Strategy Export      | #10431 | Pair each strategy with its exact loads; omit unavailable strategies.                                      | Roundtrip, missing-field and cross-strategy-load rejection tests.                                         |
| Native Matrix Bridge | #10436 | Pass the existing allocator's verified A/b and limits into this kernel; expose a bounded CLI weight sweep. | Native frame identity, unchanged q/v/a hashes, rank/residual/cone report per alternative.                 |
| Comparison Export    | #10436 | Serialize feasible alternatives into CSV/JSON with physical torque, rate, power, GRF/COP and grip metrics. | Repeat-run equality; hard-zero infeasible remains visibly infeasible; no accepted label.                  |
| Pipeline Wiring      | #10440 | Connect validated components to the current matching CLI and candidate package.                            | One real driver and one real iron run with complete source/model/control provenance.                      |
| Viewer Loading       | #10440 | Use the existing viewer owner's adapter to open that exact candidate and model.                            | Target/IK/replay overlays, physical-time scrubbing, address/top/impact/finish images and truthful status. |
| Engine Mapping       | #10439 | Add named-coordinate/unit/frame mapping fixtures to the native MatchingPlant pathway.                      | Native FK and inverse/forward dynamics roundtrips; unavailable engines fail explicitly.                   |

Do not delegate unresolved anatomical calibration, contact-law selection or
derivative mathematics as mechanical wiring. Escalate the actual failed
measurement with its source/model hashes instead of weakening acceptance.

## Engine Sequence and Stop Conditions

Pinocchio: correct the preceding defects, run separate full driver/iron fits,
then uninterrupted native forward replay under #10437. Include all controls,
feedback and applied loads, without target resets. Report whole/per-marker/
per-phase errors, closure, contact feasibility, torque magnitude/rates, runtime
and numerical tolerance convergence. A visibly plausible swing is insufficient.

MuJoCo and Drake: use `pipeline/plant.py` and native plants. The older simplified
multi-engine force adapters are not parity evidence. First qualify model and
FK correspondence, then inverse dynamics and native replay with compatible
contact and grip semantics. Preserve true differences in engine capability.

OpenSim and MyoSuite: qualify anatomy, marker registration and torque-actuated
motion first. Muscle excitation/activation dynamics require their own feasible
solve and replay; matching net torques does not qualify a muscle strategy.
Check #10394 evidence against native execution, not a handoff completion claim.

Simscape: run MATLAB R2025b explicitly, use named coordinates and actual
timeseries controls, and verify solver convergence. Existing short-horizon
Pinocchio/Simscape parity does not establish full-swing acceptance.

All engines: store model/source/runtime hashes and capability status. Native
engine qualification remains open until the full intended horizon passes.
Literal zero marker residual is not the acceptance definition; documented
capture uncertainty and frozen project gates govern. Do not promise same-day
six-engine acceptance when native evidence is missing.

## Validation Commands

```powershell
python3 -m pytest tests/unit/motion_matching/test_force_nullspace.py -q --no-cov
python3 -m ruff check src/shared/python/motion_matching/force_nullspace.py tests/unit/motion_matching/test_force_nullspace.py
python3 -m ruff format --check src/shared/python/motion_matching/force_nullspace.py tests/unit/motion_matching/test_force_nullspace.py
python3 -m mypy src/shared/python/motion_matching/force_nullspace.py --follow-imports=silent
python3 -m scripts.check_design_manual_governance
```

Keep #10436 open: native matrix integration, trajectory tradeoff reports,
segment reactions and PF-07 replay acceptance remain outstanding.
