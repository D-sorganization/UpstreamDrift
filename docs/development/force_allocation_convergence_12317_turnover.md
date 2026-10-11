# Issue 12317: Force-Allocation Convergence Turnover

The single-frame and trajectory force-redistribution paths now set `converged` from `OptimizeResult.success` only. Their independent feasibility audits are unchanged. This prevents unsuccessful SLSQP statuses 8 and 9 from being promoted to convergence when a returned candidate passes the constraint audit. The no-nullity path and numerical objective/constraints are unchanged.

## Validation

- Red: before the fix, the four parameterized status-8/status-9 cases failed because each otherwise-feasible candidate was reported converged.
- Green: `py -3.12 -m pytest tests/unit/motion_matching/test_force_nullspace.py -q -n auto --timeout=60` : 23 passed.
- Ruff check and format passed; file-size budget passed.
- Design-manual governance passed with the repository's current `release=blocked-inventory-required` state.
- MyPy with Python 3.12 reported 31 errors in 23 imported modules, including module-not-callable errors; this is not a clean scoped type-check result.
- Commit hooks passed, including Ruff and governance checks.
- No native calls or physical qualification were performed.

See [the force redistribution design note](../design/force_redistribution_convergence.md).
