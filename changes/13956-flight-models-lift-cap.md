---
issue: 13956
summary: "Allow MacDonaldHanzelyModel and constant-coefficient models to use declared cl without 0.155 constant-spin clamp"
branch: "fix/13956-flight-models-lift-cap"
dl_state: "in_review"
next_step: "Merge the PR into main."
---

Allow MacDonaldHanzelyModel and ConstantCoefficientModel to utilize their declared lift coefficients in _spin_ratio_lift_coefficient without unconditional truncation to the 0.155 constant-spin cap.
