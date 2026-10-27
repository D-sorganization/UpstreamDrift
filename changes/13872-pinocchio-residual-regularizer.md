---
issue: 13872
summary: "Pinocchio residual: append regulariser row and impose torque bounds (UD-02)"
branch: "fix/13872-pinocchio-residual-regularizer"
dl_state: "in_review"
next_step: "Merge PR"
---

Append regulariser residual to Pinocchio fit_swing residual vector with DbC postconditions and analytical Jacobian row, imposing box torque bounds via Trust Region Reflective (trf) optimization.
