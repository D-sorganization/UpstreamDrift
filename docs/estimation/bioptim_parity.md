# Swing-Backend Parity: What Each Optimizer Actually Solves

Epic [#9762](https://github.com/D-sorganization/UpstreamDrift/issues/9762),
Phase 2.2. Regenerate with:

```bash
MPLBACKEND=Agg python -m benchmarks.bioptim_parity --nodes 12 --duration 0.6 \
    --out docs/estimation/bioptim_parity_table.md
```

The interesting column is the **dynamics defect**: integrate each solution's
own torques forward from each node and see whether the next node comes back.
A trajectory that violates the equations of motion between its nodes is not a
swing a golfer could make, however fast its clubhead is reported to move.

## Results

Benchmark golfer (`GolferModel()`, `ClubModel()`), anthropometric inertials
(#9755), same sine warm start for every backend, IPOPT capped at 500
iterations. Measured 2026-09-08 on the session container (Python 3.11,
casadi 3.6.7, bioptim 3.4.0, pin 4.1.0); wall times are indicative only.

### 12 Nodes Over 0.6 s (`dt` = 55 ms)

| Backend                   | Converged     | Clubhead speed [m/s] | Wall [s] | Iterations | Max position defect [rad] | Max velocity defect [rad/s] |
| ------------------------- | ------------- | -------------------- | -------- | ---------- | ------------------------- | --------------------------- |
| initial guess (kinematic) | --            | 32.1                 | --       | --         | 1.1                       | 31.4                        |
| scipy SLSQP               | yes           | 64.2                 | 0.2      | 4          | --                        | --                          |
| casadi finite-difference  | yes           | 109                  | 1.7      | 254        | 2.88                      | 101                         |
| casadi multiple shooting  | no            | --                   | 186      | cap        | --                        | --                          |
| crocoddyl FDDP            | not installed | --                   | --       | --         | --                        | --                          |
| bioptim RK4               | yes           | 49.9                 | 38.4     | 97         | **0.26**                  | 15.2                        |
| bioptim collocation       | yes           | 45.9                 | 8.5      | 113        | **0.22**                  | 19.9                        |

### 8 Nodes Over 1.0 s (`dt` = 143 ms)

| Backend                   | Converged | Clubhead speed [m/s] | Wall [s] | Iterations | Max position defect [rad] | Max velocity defect [rad/s] |
| ------------------------- | --------- | -------------------- | -------- | ---------- | ------------------------- | --------------------------- |
| initial guess (kinematic) | --        | 18.2                 | --       | --         | 1.4                       | 28.9                        |
| scipy SLSQP               | yes       | 24.5                 | 0.1      | 4          | --                        | --                          |
| casadi finite-difference  | yes       | 63.1                 | 0.9      | 113        | 3.63                      | 44.7                        |
| casadi multiple shooting  | no        | --                   | 117      | cap        | --                        | --                          |
| bioptim RK4               | yes       | 49.9                 | 89.1     | 337        | 1.77                      | 48.6                        |
| bioptim collocation       | yes       | 49.9                 | 4.2      | 64         | 3.33                      | 63.3                        |

## What the Numbers Say

**The finite-difference path buys its speed by breaking physics.** It reports
109 m/s of clubhead speed on the finer grid. A tour driver swing is around
50 m/s and the fastest ever recorded is under 65 m/s, so that number is not a
swing. Its 2.88 rad position defect says why: the torques it computes at the
nodes would not carry the model from one node to the next. This is the
concrete measurement behind #9756 — the module called itself a direct
transcription while enforcing the dynamics nowhere between nodes.

**The OCP paths cost an order of magnitude in defect, not in credibility.**
bioptim's 0.22-0.26 rad on the same grid is more than ten times better, and
what remains is discretisation error between the transcription's own scheme
and the reference re-integration, not an unenforced constraint. Its 49.9 m/s
is the speed it was _asked_ for and could actually deliver.

**Refining the grid helps the OCPs and hurts the finite-difference path.**
Going from dt = 143 ms to dt = 55 ms cuts the bioptim defect by roughly 7x
(3.33 to 0.22 rad) while the finite-difference defect stays around 3 rad and
its reported speed climbs from 63 to 109 m/s. A transcription converges as
the grid refines; a kinematic fit with a torque check just finds more room to
cheat.

**Maximising terminal speed is the wrong objective, in every backend.** The
`casadi multiple shooting` row fails on both grids, and it fails the same way
the bioptim `maximize_speed` variant does: a negative-weight quadratic is
concave, so with the dynamics genuinely enforced the optimum sits on the
velocity bound and IPOPT never certifies it. The finite-difference path only
appears to converge here because its own consistency constraints pin
velocities to positions, shrinking the feasible set until the maximum is
interior. Both bioptim entries above therefore use the convex **target-speed**
objective, which is also what `crocoddyl_backend` (`target_speed=45.0`) and
`SwingOptimizationConfig.target_clubhead_velocity` have always meant. Ask for
the speed a golfer is trying to reach; do not ask for infinity and take
whatever the bounds allow.

**Collocation is the default for a reason.** Degree-3 collocation reaches the
same answer as RK4 multiple shooting in a fifth of the wall time (8.5 s vs
38.4 s), so `backend_registry`'s `bioptim` entry routes through it.

## Method and Caveats

- **Defect metric.** `casadi_backend.dynamics_defect` re-integrates
  `forward_dynamics` with RK4 at 16 substeps per interval, holding each
  interval's torque constant, and reports the largest per-interval mismatch.
  For the finite-difference path, whose torques are evaluated rather than
  decided, the torque used is what its own stencil implies.
- **Collocation is measured out of its native scheme.** A collocation
  solution satisfies polynomial defects at collocation points, not a
  zero-order-hold RK4 step, so its defect column mixes genuine error with
  scheme mismatch. Compare it to the finite-difference path (same
  measurement, same grid) rather than reading it as absolute integration
  error.
- **scipy has no defect entry.** The flagship optimizer carries its own
  lumped-inertia kinematic model rather than the multibody chain, so there is
  no shared ODE to violate. Its speed column is its own reported metric.
- **crocoddyl was not installed** in the container that produced these
  numbers; its wheels and the PyPI `pin` wheel must not share a process (see
  `crocoddyl_backend`'s stack probe). Rerun on a conda-forge environment to
  fill that row.
- **Wall times are indicative.** Single container, one thread, no warm cache.

## Follow-Ups

- Fill the crocoddyl row on a conda-forge environment.
- Phase 6.2 deprecates the finite-difference transcription once the OCP path
  has carried production traffic for a release.
- A defect metric evaluated in each transcription's own scheme would make the
  collocation column directly comparable.
