# Bounded One-Step Muscle Recruitment — #11792

## Scope

`src/shared/python/biomechanics/muscle_allocation.py` adds a deterministic, bounded one-step allocator that consumes desired generalized effort from an existing feedback law. It does not fit a desired trajectory, apply commands, run an engine, or qualify a production model. The public API is exported through `biomechanics`.

## Contract and Equations

For each path, the caller supplies ordered named moment arms `r_jq = -dL_j/dq`, raw prior excitation, prior activation, bounds, rate, and timestep. The coordinate effort estimate is `tau_q = sum_j r_jq F_j,tendon`; translation uses N and moment arms m/m, while rotation uses N\*m and m/rad. Every requested coordinate remains in the residual, including unactuated coordinates. Mixed-unit target residuals are scaled by one positive tolerance per coordinate.

Each raw excitation candidate advances the existing activation-dynamics state for one step. The Hill model's current quasistatic `v_CE = 0` muscle equilibrium supplies active, passive, and damping components; damping is zero here. The result keeps along-fiber and tendon-axis projected components distinct and reports tendon/fiber balance residual in N. The numerical objective includes all coordinate residuals and the equilibrium residual. The exact raw command box is intersected with the previous raw command plus/minus rate times `dt`; zero-width channels stay fixed. If a free channel starts at or below its activation floor and its feasible upper bound is above that floor, the original raw candidate is evaluated and retained as the baseline, then the optimizer starts at the midpoint of that channel's admissible above-floor interval. Fixed channels and intervals wholly below the floor are unchanged. `max_nfev` counts full candidate evaluations, including finite-difference probes and the baseline. The function returns a candidate and status without applying it.

The activation kernel floors raw input and activation at its configured `min_activation`. Results therefore expose raw commanded excitation and separately expose `effective_excitations` consumed after the floor. Rate constraints apply to raw commands; no effective-activation slew or OpenSim parity claim follows. `infeasible` means the retained candidate fails the supplied numerical tolerances, not proof of global or physiological infeasibility. Budget exhaustion, solver failure, and evaluation failure are distinct outcomes.

## Shared Force Evaluation

`biomechanics.muscle_force_evaluation.evaluate_quasistatic_muscle_force`
now exposes the allocator's existing zero-fiber-velocity Hill force composition
as an immutable scalar result. The allocator calls this same function, keeping
the prior activation update, raw/effective excitation handling, candidate
budget, and feasibility decisions in the allocator. The result reports the
force-equilibrium residual; it does not classify that residual or qualify a
model. A later replay reader can reuse the primitive only after validating its
serialized artifact. Artifact reading, OpenSim parity, and replay are outside
this change. The module has no allocator/controller dependency, although the
package aggregate initializer still imports the allocator.

## Validation

Focused command on Python 3.12.10:

```powershell
python -m pytest -q -o addopts= tests/unit/shared_python/test_muscle_excitation_allocation.py
```

The allocator regression file passed 25 tests with one existing `calculator_state_mixin.py` deprecation warning. The force-evaluation file adds 15 tests; the combined focused selection passed 40 tests. Ruff check and format check passed for the module, package export, and tests. The allocator analytic fixture has two muscles, three named coordinates, a biarticular path, and an unactuated coordinate. The force-evaluation tests independently check hand-derived components, pennation projection, immutable scalar output, invalid inputs, and failure propagation. A dedicated `max_nfev=1` case proves that an exact-target prior candidate is retained and counted while the public status remains `budget_exhausted`. Additional cases exercise mixed effort units/order, activation lag, raw/effective floor separation, fixed and rate-limited channels, passive force, pennation projection, force-equilibrium residual, malformed bounds/inputs, nonfinite values, solver failure, and the counted evaluation budget. A two-antagonist regression confirms that the deterministic interior seed makes the analytic target feasible without changing raw bounds or floor semantics. There was no native/SDK import, model or study construction, or native solver call. The SciPy numerical optimizer executes as the allocator under test.

## Diagnostic Checkpoint

A separate synthetic one-degree-of-freedom, two-muscle, 40-step feedback/replay
checkpoint compared the existing controller's recorded raw commands with a
held-command replay through its actual activation kernel. The held raw command was 0.2; position RMS was
worse and velocity RMS was better in that comparison, so it supports no overall
controller-improvement claim. The checkpoint is not this force-evaluation
unit fixture, not a fitted profile, and not a full-body replay qualification.
It reinforces that later serialized replay must preserve raw commands,
effective kernel inputs, state, and all force channels as distinct observations.
A future trace must also bind the continuous input basis: held commands and a
piecewise-linear input player are different trajectories even when their
sampled command values match.

## Limits and Next Step

This finite-difference least-squares prototype evaluates per-path equilibrium at every candidate. The tests cover only the small analytic fixture; there is no 557-muscle throughput or scaling evidence. The default 80-evaluation budget counts the initial candidate, duplicate objective evaluations, and finite-difference probes; a problem with roughly 78 free channels can exhaust before a complete dense Jacobian. Cached/sparse Jacobians, solver/factorization warm-start reuse, measured cost, and a bounded closed-loop fixture are needed before full-body integration. The next fixture should feed the allocator desired effort from the existing controller seam, preserve separate desired effort, applied effort, raw commands, and effective excitations, and replay saved commands with feedback disabled under the same declared kernel. No change to qualification in the 17-row model inventory or cross-engine parity follows. Anatomy, sourced moment arms, physiological acceptance, passive-load policy, contact, full-clock replay, and all-model qualification remain open.
