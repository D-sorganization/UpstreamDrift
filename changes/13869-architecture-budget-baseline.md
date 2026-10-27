---
issue: 13869
summary: "Architecture budget: full-scan mode with checked-in baseline ratchet (UD-25)"
branch: "fix/13869-architecture-budget-baseline"
dl_state: "in_review"
next_step: "Merge PR"
---

Add full-scan `--all` ratchet mode against a checked-in baseline `scripts/config/architecture_budget_baseline.json` for 100-line function and 8-parameter sub-gates, enforcing that new violations fail while pre-existing findings are grandfathered and only allowed to shrink.
