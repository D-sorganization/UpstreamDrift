---
issue: 15238
summary: "Split positive and negative work and impulse at the zero crossing instead of clipping the trapezoid"
dl_state: "in_review"
next_step: "Review the PR; operator handles merge authorization."
branch: "claude-desk/t4-work-zero-crossing"
---

Review fix: normalize the crossing weights before division and compute both side fractions independently, preserving finite large peaks and small opposite-sign triangles. Eight work/impulse boundary cases cover both crossing directions; six reproduced the defect before the fix.
