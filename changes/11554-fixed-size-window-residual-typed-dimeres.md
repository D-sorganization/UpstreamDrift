---
issue: 11554
summary: "Fixed-size window residual: typed DimeResidualEvaluationError replaces the np.full(100, 1e8) fake residual; bound residuals fixed-length; DbC postcondition on residual length. Single shooting / real defects deferred to MOSAIC-13 (#11545)."
branch: "fix/11554-fixed-size-residual"
paths: "src/shared/python/estimation/dime_dynamics_window.py"
---
