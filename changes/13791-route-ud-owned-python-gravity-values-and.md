---
issue: 13791
summary: "Route UD-owned Python gravity values and MJCF gravity through the canonical constant, guarded by a tested literal scanner."
dl_state: "in_review"
next_step: "Review the ready PR; follow-up #13798 covers the deferred MATLAB/Simscape literals."
branch: "staff/issue-remediator-13791-0c6d70"
paths: "src Python gravity users, owned MuJoCo MJCF, tests/unit/core gravity guard"
---

Validation: gravity literal guard, canonical gravity tests, and PF-04 contact-mode tests pass; Ruff, format, compileall, and diff checks pass. Bunker solver tests cannot collect under the available Python 3.10 runtime (`enum.StrEnum` requires 3.11); Drake fit tests cannot collect because `datetime.UTC` requires 3.11. The `scripts.pre_pr` module is absent from this checkout.
