---
issue: 11948
summary: "Fit exact-clock masked native site markers with bounded manifold control and complete-state replay."
branch: "feat/f03-native-marker-fit-11787"
---

Added exact-clock, masked native site-marker fitting to the existing
floating-root BoxFDDP controller, with manifold derivative checks,
bounded nonlinear candidate admission, source-bound evidence and
independent complete-state torque replay on a synthetic MuJoCo fixture.
