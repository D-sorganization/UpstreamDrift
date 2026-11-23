---
issue: 14756
summary: "Report missing tour metric evidence as unmeasured instead of zero error."
next_step: "Independent review; leave PR unarmed."
branch: "fix/14756-unavailable-tour-metrics"
---

P7 missing-evidence slice only. Empty marker/window/group masks and absent club/pelvis labels yield NaN, using the existing acceptance unmeasured convention. Measured formulas and thresholds are unchanged. Synthetic unit evidence only; no simulation or physical qualification.
