---
issue: 15522
summary: "Cover Golf Model Suite launcher routing and recovery with headless QTest; repair consolidated module entry points."
dl_state: "in_review"
next_step: "Independent review and CI; keep this PR unarmed per owner."
branch: "codex-desk/15522-suite-launcher-e2e"
---

Headless QTest covers three model launch routes, launch failure recovery,
caller working-directory independence and Qt disposal at desktop/phone sizes.
The Drake route targets the runnable drake_gui_app module rather than the
historical golf_gui placeholder. Focused validation: 32 passed, including 18
new cases; Ruff and scoped mypy pass. Process creation is intercepted, so this
does not establish native engine dynamics or Docker runtime acceptance.

Handoff: independent reviewer owns CI/acceptance; keep the PR unarmed per owner.
