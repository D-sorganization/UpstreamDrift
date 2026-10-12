# Force Redistribution Convergence

Force redistribution reports two independent outcomes on `RedistributionResult`:

- `feasible` is the post-solve audit of force balance and configured constraints on the returned candidate.
- `converged` reports the optimizer's own success flag; it is not inferred from SLSQP's numeric status code.

An optimizer may return a candidate that passes the feasibility audit while reporting failure. SLSQP statuses 8 (positive directional derivative) and 9 (iteration limit) do not establish optimizer convergence. Consumers may inspect a feasible candidate, but must not treat feasibility as evidence of optimizer convergence or qualification. The no-nullity path does not invoke the optimizer and retains its existing direct feasibility audit.

This is a software-result contract only; it makes no physical validation or qualification claim.
