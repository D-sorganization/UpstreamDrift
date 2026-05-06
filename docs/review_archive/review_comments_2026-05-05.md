# Review Comments Archive - 2026-05-05

Generated: 2026-05-05T21:27:06.262675

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4013: src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/private/extract_sim_out.m:337

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Resample signals against actual signal time vectors**

This resampling step uses synthetic sample indices (`1..n`) instead of each signal's real timestamps, so with the default variable-step solver (`ode23t`) or any decimated/nonuniform logging, trajectories get time-warped before cost evaluation. That can silently corrupt optimization results because phase and peak timing are shifted even when raw simulation...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4013#discussion_r3192987724)

---

### PR #4013: src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/tests/test_simulate_with_coefficients.m:84

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Avoid calling private helper directly from test folder**

These tests call `theta_to_polynomial_struct` directly, but the function is defined under `shared/private/`, which is only visible to functions in the parent `shared` folder. Since this test class lives in `shared/tests`, MATLAB cannot resolve the private function, so these unit tests fail before validating behavior.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4013#discussion_r3192987733)

---

