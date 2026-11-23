---
issue: 15345
summary: "Bundle adjustment FC-011 guards, feasible tangent projection, bounded FD, and positive prior sigmas"
---

PR #15537 review 6017001213: reproduce all four FIX-NEEDED findings with
regressions before fixing them. Retain inward pose tangents at active bounds,
handle empty fixed-pose tangent bases, cap FK/constraint probes in narrow
boxes, skip fixed-coordinate probes, and reject negative-infinite prior sigmas.

Validation: focused bundle-adjustment pytest suite, Ruff lint and format checks.
Next step: independent acceptance of the pushed head. No self-review,
approval, merge, or auto-merge arming is authorized for this fixer session.
