---
issue: 15225
summary: "Pair Tools cache count-pressure correction while preserving integrity and minimum-age checks"
---

Review fix: active and minimum-age-protected model paths also protect overlapping file/directory aliases from automatic deletion through another ID. Four regressions reproduce retained-file loss before the fix. Capacity remains soft when every candidate overlaps retained data.
