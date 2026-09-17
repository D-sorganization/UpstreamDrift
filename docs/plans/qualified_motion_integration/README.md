# Qualified Motion Integration

Execution plan for [epic #10254](https://github.com/D-sorganization/UpstreamDrift/issues/10254).
The epic remains the complete scope: constrained Pink IK, full-body Crocoddyl,
useful diagnostics, viewer decision, product integration and physical qualification.
This plan freezes the difficult boundaries before delegating implementation.

## Current Authority

- Review `origin/main` at dispatch; the current audited main revision is
  `0ec64e45f1f259c11e26e36315ee370a5c51c7d9`. Use a separate checkout.
- Start execution with [Worker Turnover](TURNOVER.md) and the
  [Pink Pipeline Packet](PINK_TURNOVER.md), which define dependency order,
  owned files, TDD cases, contracts and native qualification commands.
- Full-body anatomy, contact and capture authority: `full_body_spec`,
  `contact_law`, `tour_capture_contract`, and the existing ground-support
  `motion_matching.pipeline` package.
- Preserve the ongoing #10162 program. Anthropometry refresh #10250 and
  pipeline consolidation #10251 are closed. #10108 provides hip calibration;
  #10159 owns MJX plant reconciliation. #10271 tracks remaining receipt-chain
  integrity; regenerated artifacts are not automatically physically qualified.
- Historical returned81 is a 307-sample, 25-marker, 0.85 s reference. It does
  not certify the complete full-body capture. Ground-support receipts have
  34 modeled labels. The capture contract has 38; coverage must be explicit.
- Regenerate evidence through the consolidated pipeline after validating
  the current input chain. Local derivative probes on immutable old fixtures
  are algebra tests, not regenerated scientific qualification.

## Integration Decisions

1. **One Pipeline, Explicit Backends.** Extend existing address, reference,
   dynamics and receipt stages. The seven-DOF swing optimizer registry has a
   different signature and must not receive full-body data through a pretend
   compatible adapter. Reuse its capability/probe patterns, not its fixed layout.
2. **Explicit Coordinate Mapping.** `coordinate_order` is the canonical scalar
   state order. Pinocchio tree order can differ. Every array conversion has an
   explicit permutation. Derive dimensions; never assume 41 coordinates or
   `nq == nv` for arbitrary future model types.
3. **Underactuation Is Physical.** Native pelvis coordinates are
   `TranslationInputX/Y/Z` and `HipInputX/Y/Z`. Their control effort is zero
   in qualified full-body replay. Root forces from contact are allowed;
   hidden root actuators, pose resets and controller assistance are not
   accepted as open-loop fitting.
4. **Closure Has Six Components.** Translation-only grip residuals cannot
   qualify a weld. Use a consistent relative SE(3) residual, its derivative,
   and separately reported translation/rotation tolerances. Do not reuse a
   constraint velocity Jacobian as the derivative of a finite pose error
   without checking the log-map and frame conventions.
5. **Pink Solves Differential IK.** Weighted tasks are soft objectives;
   equality constraints and inequality limits are separate. Iterative
   projection time must not be confused with physical capture time, and
   multiple inner iterations must not silently multiply allowed velocities.
   Recompute post-integration residuals, limits and declared infeasibility.
6. **Crocoddyl Must Differentiate the Actual Plant.** Preserve the shared
   compliant law and bilateral weld. Differentiate generalized contact
   forces, constrained dynamics and the discrete integrator. A rigid-contact
   surrogate or unconstrained ABA derivative is not interchangeable evidence.
7. **Optimization Is Not Acceptance.** Always replay independently from the
   original state over the full horizon. Feedback, node resets, shorter
   windows and alternate controls are separately named modes. Global
   degree-six controls remain the historical #10062 contract until an
   explicit, versioned contract decision changes it.
8. **Optional Visualizers Share Physics.** Viewer adapters consume geometry
   and replay state; they never own calibration or simulation. Gepetto is a
   bounded optional capability assessment. Native PyQt remains usable, with
   MeshCat evaluated for portable review. No viewer import is proof of a
   functioning scene or scientific validity.

## First Expert Boundary: Contact Derivatives (#10255)

The full-body forward plant computes acceleration
`a(q,v,tau + tau_contact(q,v))`. Before #10255, its inherited derivative method
treated the combined effort as independent. The required chain rule is:

```text
da/dq = da/dq|combined_effort + da/deffort @ dtau_contact/dq
da/dv = da/dv|combined_effort + da/deffort @ dtau_contact/dv
da/dtau = da/deffort

tau_contact = sum(J_position.T @ F_world)
dtau_contact/dq = sum(dJ_position.T/dq @ F_world
                         + J_position.T @ dF_world/dq)
dtau_contact/dv = sum(J_position.T @ dF_world/dvelocity @ J_position)
```

World force derivatives use the exact shared law. Force clipping and contact
activation are piecewise differentiable: expose branch validity instead of
claiming smoothness at the boundary. At zero tangential speed, use the finite
limit of regularized friction. A separate Pinocchio data instance owns contact
kinematic derivatives so it cannot overwrite the constrained solver caches.

Analytical Jacobian time variation along basis velocities is a valid initial
route to the derivative of the frame Jacobian when Hessian bindings are not
available. Verify it against independent finite differences. Profile before
replacing it with a more complex implementation.

### Derivative Validation

The failing-first real-engine checks reproduced missing pose and velocity
terms before the correction. The corrected implementation passes 11 real
Pinocchio integration tests and 22 shared contact-law/derivative tests.
Integration checks compare three seeded independent directions for each
pose, velocity and effort derivative, with active and inactive contact,
using a centered step of `1e-6` and `rtol=atol=2e-5`.

The isolated Ubuntu 24.04 / Python 3.12 runtime uses `pin==3.8.0`,
`cmeel-urdfdom==4.0.1`, `cmeel-tinyxml2==10.0.0`, `numpy==2.3.5`,
`scipy==1.18.1`, `pytest==9.1.1` and `pytest-timeout==2.4.0`.
The explicit URDF/tinyxml pins avoid incompatible shared-library resolution.
These are the tested derivative runtime versions, not a qualified complete
Crocoddyl/Pink/viewer environment.

```bash
python3 -m pytest tests/unit/motion_matching/test_contact_derivatives.py tests/unit/motion_matching/test_contact_law.py -q --timeout=60
python3 -m pytest tests/integration/motion_matching/test_full_body_contact_derivatives.py -q --timeout=60
```

The second command must execute with real Pinocchio: skipped tests do not
qualify the derivative boundary. Full repository Ruff lint/format, focused
Mypy, architecture/file-size budgets and design-manual governance pass.
Manual publication remains blocked by its existing inventory requirement.

## Delegation Contract

Every worker receives one issue, owned files, exact public signatures,
preconditions/postconditions, a failing-first test list, a real validation
command and explicit exclusions. No worker chooses a new plant, denominator,
acceptance threshold or coordinate convention. The integration owner reviews
those decisions and exercises both provider and consumer before merging.

| Package                           | Ownership and Prerequisite                                     | Evidence Required                                                                                                   |
| --------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Contact Force Derivatives         | Bounded worker; #10255; pure shared function and tests only    | Red test before implementation; independent differences, tilted planes, tiny/zero slip, clipping, immutable outputs |
| Full-Plant Derivative Composition | Integration owner; consumes verified force derivative          | Real Pinocchio pose/velocity/effort directional checks, reordered names, active/no contact, cache independence      |
| Runtime Qualification             | Bounded worker after dependency audit                          | Isolated environment; subprocess ABI probe; installed versions; actual IK and dynamics steps; no mock-only pass     |
| Gepetto Adapter                   | Bounded worker after viewer audit                              | Preserved API, persistent scene, correct collision/visual models, observable display, failure/cleanup behavior      |
| Pink Task Assembly                | Worker only after expert closure/Jacobian and timing contracts | Marker masking, equality constraints, limits, finite-step audits, infeasible/dropout tests, real QP solve           |
| Crocoddyl Action Assembly         | Worker only after full-plant derivatives qualify               | Explicit actuation, exact discrete dynamics, state/control derivatives, terminal behavior, independent rollout      |
| Evidence and Product Wiring       | Worker after backend signatures stabilize and #10250 refresh   | Backend selection reaches real solver; cancellation/failure; hashes/labels/timestamps; desktop/web parity           |

## Completion and Handoff

No stage is complete because a worker says it is. Review changed source, red
and green test evidence, real-runtime results, integration tests and required
CI. Follow TDD, DbC, LoD and DRY; failures stay visible in the handoff.

Before changing defaults, compare both clubs with the same model, capture,
masks, contact law, starting state and compute budget. Retain the #10162
full-swing targets (<60 mm driver, <95 mm 7-iron), plus explicit physical and
cross-engine gates. User visual review remains pending. R2025b remains the
Simscape authority. Library integration cannot close independently blocked
physical qualification requirements.

The epic is complete only when each work package's actual current-state
evidence meets its requirement, or the user explicitly changes that scope.
Reasoned library-feature deferrals are recorded; missing required integration
is not silently converted into a deferral.

## Global Coefficient Lift and Discrete Integration (#10265)

The selected Crocoddyl formulation augments physical state `[q, v]` with one
constant vector of global Bernstein coefficients. A zero-time initial action
sets that vector with `nu = 7 * n_actuated`; subsequent physical actions have
`nu = 0` and copy the same coefficient state. The initial physical state is
fixed. This preserves the historical control family and allows FDDP defects
without silently introducing independent effort controls at each time node.
A real Crocoddyl 3.2.1 toy probe exercised varying action control dimensions,
FDDP descent, and bounded BoxFDDP. Full-body action qualification is still open.

Each physical action must use `NativeFullBodyStep` and its exact discrete
linearization. In particular, multiplying a continuous acceleration Jacobian
by the time step is not the derivative of RK4. Every stage and substep carries
state and coefficient sensitivities. Reject or explicitly diagnose contact
kinks when requesting smooth optimization; do not present a branch derivative
as global smoothness. Pure closed-form dynamics and independent native finite
differences establish this boundary before action assembly.

Dense coefficient augmentation increases memory and factorization cost. Measure
native derivative cost and solver memory before choosing a full-capture node
grid. Any temporal coarsening must retain the declared objective, physical
horizon, event coverage and independent full-resolution replay; a shorter or
sparsely audited fit is not equivalent acceptance.
