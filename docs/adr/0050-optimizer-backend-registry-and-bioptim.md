# ADR-0050: Optimizer Backend Registry and the `bioptim` OCP Layer

- Status: Accepted
- Date: 2026-09-08
- Decision Makers: repo owner (Dieter), claude
- Related Issues/PRs: #9760 (registry), #9762 (bioptim epic), #9755, #9756, #9757, #9758, #9759, #9761; epic #8390 (B3 #8398 CasADi, B4 #8399 Crocoddyl)

## Context

The seven-DOF swing of `optimization/_swing_kinematics.JOINTS` could be
optimised through five code paths before this decision: scipy SLSQP
(`swing_optimizer.py`), the CasADi finite-difference backend
(`casadi_backend.py`), Crocoddyl FDDP (`crocoddyl_backend.py`), the Drake
trajopt under `motion_pipeline/matching/trajopt_drake.py`, and JaxSim
differentiable rollouts. Backend selection was a string compare inside
`SwingOptimizer.optimize`; nothing recorded which backend was importable in a
given environment or which problem each one is for. Two of the paths were
also physically wrong in the same way: both dynamic backends consumed the URDF
bridge's `mass = 1.0, I = 1e-2` conditioning placeholders as if they were
segment inertials (#9755), and the CasADi path enforced no dynamics between
nodes at all (#9756).

The bioptim epic (#9762, `docs/issues/EPIC_BIOPTIM_OCP_INTEGRATION.md`) adds
a sixth path for constrained, multiphase, tracking and estimation OCPs. Adding
it without a registry and a problem-class assignment would be sprawl.

Constraints that shaped the decision:

- UpstreamDrift is pip-first. bioptim, biorbd and CasADi-enabled Pinocchio are
  conda-forge only; the PyPI `pin` wheel does not ship `pinocchio.casadi`, and
  mixing conda-forge Pinocchio with the `pin` wheel is the dual-`libpinocchio`
  crash the Crocoddyl backend already documents.
- bioptim ships breaking API changes in every recent minor release and has a
  small core team.
- Every backend must degrade with an install hint when its stack is absent;
  `import src.shared.python.optimization` must never require an optional extra.

## Decision

1. **One registry.** `optimization/backend_registry.py` holds a
   `BackendSpec` per backend: name (the `OptimizationConfig.solver` value),
   problem class, mock-tolerant availability probe, install hint, and a
   flagship-layout `solve` callable. `SwingOptimizer.optimize` selects through
   `get_backend(config.solver)`; any name the registry does not know still
   reaches `scipy.optimize.minimize`, so `solver="SLSQP"` is unchanged.

2. **Problem classes.**

   | Backend                    | Owns                                                     | Status                        |
   | -------------------------- | -------------------------------------------------------- | ----------------------------- |
   | `scipy`                    | quick / legacy smooth NLP on the node grid               | kept                          |
   | `casadi`                   | legacy kinematic fit with a torque check                 | **deprecated** (Phase 6.2)    |
   | `casadi-multiple-shooting` | single-phase torque-driven OCP with no bioptim installed | fallback for the bioptim path |
   | `crocoddyl`                | fast DDP / FDDP, loosely constrained                     | kept, complementary           |
   | `bioptim`                  | constrained, multiphase, tracking and estimation OCPs    | new (epic #9762)              |

3. **bioptim is driven through its custom-model protocol only.** UpstreamDrift's
   own CasADi symbolic model (`ocp/symbolic_model.SymbolicSwingModel`, built on
   `casadi_backend.build_symbolic_rnea` with the anthropometric inertials of
   `model_provider.swing_link_inertials`) implements `bioptim.StateDynamics`.
   biorbd, bioviz, pyorerun and bioptim's `PinocchioModel` are out of scope.
   Pinocchio remains the **numeric oracle**: every CasADi kernel is tested
   against `pin.rnea` / `pin.crba` / `pin.aba`, never used symbolically.

4. **Pin policy.** The `bioptim` extra pins a release SHA of
   `pyomeca/bioptim` (never a branch). Re-pinning is a ticket that re-runs the
   Phase 0–3 tests. All bioptim knowledge lives in `optimization/ocp/`;
   `tests/architecture/test_bioptim_isolation.py` fails on any `import bioptim`
   outside it, so a `casadi.Opti` rewrite of the formulations stays a two-day
   fallback if upstream is abandoned.

5. **Inertials.** Dynamic backends consume
   `model_provider.swing_link_inertials` (anthropometric). The URDF bridge's
   placeholders stay the default for kinematic matching, where they are
   conditioning, not physics.

## Alternatives Considered

1. **Own `casadi.Opti` transcription only (≈300 LOC).** Rejected as the primary
   path: it re-implements multiphase, tracking penalties, parameters, MHE and
   collocation that bioptim already tests upstream. Kept as
   `casadi-multiple-shooting`, the no-bioptim fallback and the parity baseline.
2. **conda-forge stack (bioptim + biorbd + Pinocchio-CasADi).** Rejected: breaks
   the pip-first contract and reintroduces the dual-`libpinocchio` hazard.
3. **Stay with scipy `least_squares` for estimation.** Rejected: the
   `NON_FINITE_RESIDUAL_SENTINEL` (#9757) is a symptom of finite-difference
   Jacobians through an unstable forward model; exact AD removes that class of
   bug.

## Amendment 2026-09-08: The Finite-Difference Path Is Deprecated, Not Rewired

The epic's Phase 6.2 proposed making `solve_swing_casadi` delegate to the
bioptim OCP whenever bioptim is importable. That is rejected: the two do not
solve the same problem. The finite-difference path _maximises_ terminal
clubhead speed, which with the dynamics genuinely enforced is a concave
objective whose optimum sits on the velocity bound and which no interior-point
solver certifies (measured in `docs/estimation/bioptim_parity.md`; the CasADi
multiple-shooting path fails the same way). The OCP path therefore defaults to
a convex _target_-speed objective. Silently swapping one for the other would
change every existing caller's answer.

Instead: `transcription="finite_difference"` emits a `DeprecationWarning`
naming its replacement, the registry marks the `casadi` backend deprecated, and
callers migrate deliberately to `multiple_shooting` (same maximisation, real
dynamics, may not converge) or to `bioptim` (target speed, converges). Removal
follows one release after the warning ships.

## Consequences

- Positive: one place answers "which backend, and is it installed"; adding a
  backend is one `register_backend` call; the FD path can be deprecated on a
  schedule instead of lingering; torque limits and injury scores from the
  dynamic backends refer to real masses.
- Negative: a git-pinned dependency in an extra (not publishable to PyPI as-is,
  which the project does not do); two import shims (`biorbd_casadi`,
  `tkinter`) plus a `matplotlib.cm.get_cmap` alias until #9761 lands upstream;
  bioptim's multiple-shooting solves are slower than the FD fit they replace.
- Follow-ups: #9761 (upstream PR, deletes the shims); Phase 6.2 removes the FD
  path one release after the deprecation warning ships; a closed-loop two-hand
  grip needs `ConstraintFcn.CUSTOM` (bioptim holonomic constraints are
  biorbd-only) and is out of scope.

## Validation

- `tests/unit/optimization/test_backend_registry.py`: registry order, scipy
  always available, engine backends unavailable under the unit tree's mocks,
  install hints, deprecation warning.
- `tests/integration/optimization/test_casadi_swing_live.py`: kernels vs
  Pinocchio; FD defect vs multiple-shooting defect.
- `tests/architecture/test_bioptim_isolation.py`: no `bioptim` import outside
  `optimization/ocp/`.
- `ci-optional-stack.yml` legs for `optimal-control` and `bioptim` (#9759).
- `docs/estimation/bioptim_parity.md` records the parity numbers.
