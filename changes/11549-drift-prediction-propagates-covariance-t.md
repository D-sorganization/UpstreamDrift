---
issue: 11549
summary: "drift prediction propagates covariance through F = df/dx per step, steps the full horizon as H steps of dt, and never actuates the floating-base root by default"
branch: "fix/11549-drift-covariance-propagation"
paths: "src/shared/python/estimation/drift_prediction.py"
---
