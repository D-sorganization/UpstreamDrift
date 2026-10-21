---
issue: 13595
summary: "Clear CI Standard Actionlint and ShellCheck failures and add the governed fleet-policy consumer"
dl_state: "in_review"
next_step: "Merge the Repository_Management #2253 policy provider before publishing this consumer, then rerun the canonical pre-PR gate."
title: "CI Standard Lint Gate"
owner: "codex"
branch: "codex/13595-ci-lint"
paths: ".github/actionlint.yaml,.github/workflows/ci-standard.yml,tests/ci/test_ci_infrastructure.py,tests/ci/test_ci_drift_gate_arguments.py,AGENT_HANDOFF.md"
---

Branch: `codex/13595-ci-lint`; implementation commit `45f52df558`; governance consumer commit `4e3926ac3a` with trusted `EVENT_NAME` forwarding for approved merge-group batch support. The existing CI Standard repair remains unchanged. The consumer obtains the gate and eligibility implementation from protected RM `main`, selects only an eligible repair mode, and retains strict fallback and visible receipts/debt. Focused consumer contract tests pass (10 cases); full Actionlint/ShellCheck/Ruff and canonical checks remain to be rerun. The provider PR and rollout registry must merge before this consumer is published. The prior full pre-PR run failed its Fleet Policy Gate on the unchanged generated development log: 31 in-progress entries / cap 8, 198 active entries / cap 75, and 587,048 bytes / 200 KB. No push, PR, diagnostic suppression, policy-budget increase, fabricated task state or manual shared-log edit was performed.
