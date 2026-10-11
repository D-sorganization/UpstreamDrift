---
issue: 11554
summary: "DIME dynamics window: multiple shooting with states and controls as decision variables, real transition defects d_k = x_{k+1} - Phi(x_k, u_k, dt) from the provider step, and real costs evaluated on the returned trajectory"
branch: "fix/11554-multiple-shooting"
---

Behaviour change: the private residual evaluator now takes the packed decision vector `z = [u, x_1..x_N]` (`pack_window_decision`); `solve_dime_dynamics_window` gains keyword-only `initial_controls` / `initial_knot_states` warm starts; the reported `control_effort_cost` now matches the minimised residual (`0.5 * 1e-4^2 |u|^2`, previously `0.5 * 1e-4 |u|^2`). EXACT re-optimises the controls with the knots eliminated (hard defects); knot states carry the provider's internal_state/v_dot from the step prediction. Remaining (#11545): inverse-dynamics variable projection, manifold knot states (n_q != n_v), analytic/sparse Jacobians.
