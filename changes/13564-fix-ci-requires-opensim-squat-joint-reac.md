---
issue: 13564
summary: "Fix CI Requires OpenSim squat joint-reaction and native muscle-effort regressions; align lift ID kinematics and document allocator semantics"
dl_state: "in_review"
next_step: "Merge after CI Requires OpenSim and quality-gate are green on PR #13568"
branch: "cursor/fix-opensim-ci-13564-90e0"
---

- `run_inverse_dynamics` applies `generalized_speeds_from_spline_derivatives` for Rajagopal pelvis ZXY Euler coordinates (same convention as OSL-5 joint reactions).
- `muscle_allocation` sets `optimizer_success=False` when the evaluation budget is exhausted mid-search.
- Governed references: chapter 39 (allocator statuses), `lift_models.tex` (OSL-3 ID kinematics), calculation-registry `UP-D1-native-offline-muscle-matching-inventory`.
