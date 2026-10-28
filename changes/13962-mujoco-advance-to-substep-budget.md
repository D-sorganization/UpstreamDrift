---
issue: 13962
summary: "Require _advance_to to reach target time within substep budget (UD-37)"
branch: "fix/13962-mujoco-advance-to-substep-budget"
dl_state: "in_review"
next_step: "Merge PR"
---

Require `_advance_to` to reach `t_target` within `max_substeps` budget and fail closed if the substep budget is exhausted before target time is reached.
