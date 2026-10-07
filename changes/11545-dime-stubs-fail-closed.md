---
issue: 11545
summary: "Fail closed on the last DIME stub constants: replay angular drift, cancellation and GRF equilibrium are None (not measured); solver cache records no fabricated 0.0 timing"
branch: "fix/11545-stubs-fail-closed"
---

Refs #11545. Replay metrics never computed are None; store_window_solve elapsed_s defaults to None. Computing those replay metrics is open. Tests: tests/unit/estimation/test_dime_continuous_replay.py, test_dime_solver_cache.py.
