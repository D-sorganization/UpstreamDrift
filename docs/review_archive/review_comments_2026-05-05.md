# Review Comments Archive - 2026-05-05

Generated: 2026-05-05T23:25:27.963172

## Reviewer (chatgpt-codex-connector[bot]) (3 comments)

### PR #4048: src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option1_direct_optimization/fit_swing_hybrid.m:50

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Compare skip tolerance against surrogate RMSE, not raw loss**

`skip_polish_tol_m` is documented/used as a metre tolerance, but this branch compares it to `surrogate_phase.final_loss` and later derives `final_rmse_m` as `sqrt(final_loss)`. The Python surrogate's `final_loss` is a weighted objective (position + quaternion + optional bound penalty), not a pure squared RMSE in metres, so this can incorrectly ski...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4048#discussion_r3193373610)

---

### PR #4048: src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/private/result_to_table_row.m:79

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Parse string-valued option IDs numerically**

When `result.option` is provided as a string (the docstring explicitly allows values like `"1".."4"`), `double(result.option)` returns character code values (e.g. `"1" -> 49`) instead of the intended option number. This silently corrupts leaderboard rows and sort/filter behavior for any results that persist `option` as text.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4048#discussion_r3193373614)

---

### PR #4048: src/shared/python/motion_matching/loaders/c3d.py:216

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Select Gears address frame after short-gap interpolation**

This computes the address frame from raw marker arrays before `extract_cluster_club_pose` runs its short-gap interpolation, so traces with staggered short NaN gaps (each marker recoverable, but no raw frame where all six markers are simultaneously finite) will fail early with `No frame where all Gears cluster markers are simultaneously finite`. That contrad...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4048#discussion_r3193373616)

---
