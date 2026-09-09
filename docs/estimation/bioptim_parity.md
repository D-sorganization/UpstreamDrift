# Swing-Backend Parity: What Each Optimizer Actually Solves

Epic [#9762](https://github.com/D-sorganization/UpstreamDrift/issues/9762),
Phase 2.2. Regenerate with:

```bash
MPLBACKEND=Agg python -m benchmarks.bioptim_parity --nodes 12 --duration 0.6 \
    --out docs/estimation/bioptim_parity_table.md
```

The interesting column is the **dynamics defect**: integrate each solution's
own torques forward from each node and see whether the next node comes back.
This tests local consistency with the specified mathematical model. It does
not establish that a golfer could execute the trajectory or validate impact
forces, shaft response, ball launch, or acoustics.

## Historical Results

These tables retain their original fixed 16-substep RK4 reference. They have
**not** been regenerated with the adaptive reference introduced for #9830.
Reference integration error was not independently resolved in these runs;
do not interpret the columns as certified continuous-ODE errors.

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

The finite-difference formulation does not enforce interval dynamics. Its
reported speed and node torque limits therefore do not establish a dynamically
feasible swing. The historical 2.88 rad discrepancy motivates checking the
continuous ODE, but its precise value also depends on the unresolved reference.

The OCP formulations enforce their chosen discrete equations. Smaller
historical discrepancies (0.22–0.26 rad and 15.2–19.9 rad/s on the finer grid)
do not establish acceptable accuracy. Both state components require explicit
application-specific numerical budgets and a resolved reference. A target
speed reached by an optimizer is not evidence of physical realizability.

The two tables change both interval spacing and total swing duration. They
are not a controlled mesh-convergence experiment. Nonlinear constraints and
different objectives also prevent treating backend rows as solutions of an
identical optimization problem. A shared initial guess does not ensure a shared
local optimum. The failed speed-maximization runs do not prove that this
objective is universally invalid or that concavity caused those failures;
target-speed penalties do not make the full nonlinear OCP convex.

The recorded collocation run is faster than RK4 and reaches a different speed
(45.9 versus 49.9 m/s). These single runs do not demonstrate equivalent
solutions or a general performance advantage.

## Method and Caveats

- **Historical defect metric.** The tables re-integrated
  `forward_dynamics` with RK4 at 16 substeps per interval, holding each
  interval's torque constant, and reports the largest per-interval mismatch.
  For the finite-difference path, whose torques are evaluated rather than
  decided, the torque used is what its own stencil implies.
- **Current reference.** Calling `dynamics_defect` without `n_substeps` uses
  DOP853 twice on the same model RHS, with tighter tolerances and a smaller
  maximum step on the second pass. Absolute tolerances are separate for
  position (rad) and velocity (rad/s). Each endpoint must pass a componentwise
  refinement check; failure raises instead of emitting qualified metrics.
  `reference_resolution` retains controls, RHS counts and normalized endpoint
  differences. This is numerical refinement evidence, not a rigorous error
  bound or an independent implementation of the dynamics. The four historical
  `to_dict()` fields remain unchanged; archive the resolution evidence as well
  when making a new qualification claim.
- **Discrete feasibility.** An explicit positive `n_substeps` still selects
  fixed RK4. Using the transcription's own substeps tests its discrete
  equations; it does not resolve continuous-ODE accuracy. Reports keep position
  and velocity separate. The legacy `max_defect` mixes units and must not define
  a scientific accuracy budget.
- **Control reconstruction.** Collocation and shooting comparisons must use
  the same interval torque convention and model. Re-integrating an identical
  zero-order-held control with a resolved reference tests local ODE consistency;
  a different control interpolation would also measure reconstruction mismatch.
  Every interval restarts at the candidate node, so these are not accumulated
  whole-swing error bounds.
- **scipy has no defect entry.** The flagship optimizer carries its own
  lumped-inertia kinematic model rather than the multibody chain, so there is
  no shared ODE to violate. Its speed column is its own reported metric.
- **crocoddyl was not installed** in the container that produced these
  numbers; its wheels and the PyPI `pin` wheel must not share a process (see
  `crocoddyl_backend`'s stack probe). Rerun on a conda-forge environment to
  fill that row.
- **Wall times are indicative.** Single container, one thread, no warm cache.

The preserved #9830 six-node, 0.6 s candidate demonstrates the distinction:
eight-substep shooting has an own-grid residual below 1e-8 in each state block,
yet adaptive re-integration gives about 0.4812 rad and 25.857 rad/s. The old
16-substep reference instead gives 0.5191 rad. Passing the historical 0.5 rad
regression ceiling does not qualify this candidate for impact predictions.
Separate 16- and 32-substep optimizations reduced the observed discrepancies;
the settings, source identity and limits are recorded in the
[shooting refinement turnover](../development/shooting_convergence_9830_turnover.md).

## Follow-Ups

- Fill the crocoddyl row on a conda-forge environment.
- Phase 6.2 deprecates the finite-difference transcription once the OCP path
  has carried production traffic for a release.
- Regenerate parity evidence with resolved references, comparable objectives,
  explicit control reconstruction and independent position/velocity budgets.
- Keep discrete feasibility, reference resolution, continuous-ODE discrepancy
  and experimental validation separate when qualifying impact or acoustic use.
