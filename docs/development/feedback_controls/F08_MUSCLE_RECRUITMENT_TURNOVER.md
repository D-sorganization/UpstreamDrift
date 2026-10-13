# Bounded One-Step Muscle Recruitment — #11792

## Scope

`src/shared/python/biomechanics/muscle_allocation.py` adds a deterministic, bounded one-step allocator that consumes desired generalized effort from an existing feedback law. It does not fit a desired trajectory, apply commands, run an engine, or qualify a production model. The public API is exported through `biomechanics`.

## Contract and Equations

For each path, the caller supplies ordered named moment arms `r_jq = -dL_j/dq`, raw prior excitation, prior activation, bounds, rate, and timestep. The coordinate effort estimate is `tau_q = sum_j r_jq F_j,tendon`; translation uses N and moment arms m/m, while rotation uses N\*m and m/rad. Every requested coordinate remains in the residual, including unactuated coordinates. Mixed-unit target residuals are scaled by one positive tolerance per coordinate.

Each raw excitation candidate advances the existing activation-dynamics state for one step. The Hill model's current quasistatic `v_CE = 0` muscle equilibrium supplies active, passive, and damping components; damping is zero here. The result keeps along-fiber and tendon-axis projected components distinct and reports tendon/fiber balance residual in N. The numerical objective includes all coordinate residuals and the equilibrium residual. The exact raw command box is intersected with the previous raw command plus/minus rate times `dt`; zero-width channels stay fixed. If a free channel starts at or below its activation floor and its feasible upper bound is above that floor, the original raw candidate is evaluated and retained as the baseline, then the optimizer starts at the midpoint of that channel's admissible above-floor interval. Fixed channels and intervals wholly below the floor are unchanged. `max_nfev` counts full candidate evaluations, including finite-difference probes and the baseline. The function returns a candidate and status without applying it.

The activation kernel floors raw input and activation at its configured `min_activation`. Results therefore expose raw commanded excitation and separately expose `effective_excitations` consumed after the floor. Rate constraints apply to raw commands; no effective-activation slew or OpenSim parity claim follows. `infeasible` means the retained candidate fails the supplied numerical tolerances, not proof of global or physiological infeasibility. Budget exhaustion, solver failure, and evaluation failure are distinct outcomes.

## Validation

Focused command on Python 3.12.10:

```powershell
python -m pytest -q -o addopts= tests/unit/shared_python/test_muscle_excitation_allocation.py
```

The allocator regression file passed 25 tests with one existing `calculator_state_mixin.py` deprecation warning. Ruff check and format check passed for the changed allocator and test. The analytic fixture has two muscles, three named coordinates, a biarticular path, and an unactuated coordinate. A dedicated `max_nfev=1` case proves that an exact-target prior candidate is retained and counted while the public status remains `budget_exhausted`. Additional cases exercise mixed effort units/order, activation lag, raw/effective floor separation, fixed and rate-limited channels, passive force, pennation projection, force-equilibrium residual, malformed bounds/inputs, nonfinite values, solver failure, and the counted evaluation budget. A new two-antagonist regression was RED at the all-zero start (four total candidate evaluations, one optimizer evaluation, and gtol termination) despite an above-floor feasible witness; the deterministic interior seed makes the same analytic target feasible without changing raw bounds or floor semantics. This is an analytic regression only. A separate OpenSim adapter attempt stopped at an infeasible proposal before Manager advancement, and that native outcome remains unresolved. The allocator unit tests do not import OpenSim or construct native models. A separate authorized OpenSim adapter attempt constructed its small fixture but stopped at an infeasible proposal before Manager advancement; the native-state cause is not established by this analytic regression. No physical or model qualification claim follows.

## Limits and Next Step

This finite-difference least-squares prototype evaluates per-path equilibrium at every candidate. The tests cover only the small analytic fixture; there is no 557-muscle throughput or scaling evidence. Cached/sparse Jacobians, solver/factorization warm-start reuse, measured cost, and a bounded closed-loop fixture are needed before full-body integration. The next fixture should feed the allocator desired effort from the existing controller seam, preserve separate desired effort, applied effort, raw commands, and effective excitations, and replay saved commands with feedback disabled under the same declared kernel. No change to qualification in the 17-row model inventory or cross-engine parity follows. Anatomy, sourced moment arms, physiological acceptance, passive-load policy, contact, full-clock replay, and all-model qualification remain open.
