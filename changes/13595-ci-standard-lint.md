---
issue: 13595
summary: "Repair CI lint, governed policy consumption, and pre-push failure guidance"
dl_state: "in_review"
next_step: "Wait for the registry extension and CI cohort drain; resolve the additional RM #2259 provider-job fetch failure before canonical pre-PR checks and publication."
title: "CI Standard Lint Gate"
owner: "codex"
branch: "codex/13595-ci-lint"
paths: ".github/actionlint.yaml,.github/workflows/ci-standard.yml,.github/workflows/fleet-policy-gate.yml,scripts/pre_push_hook.py,tests/ci/test_ci_infrastructure.py,tests/ci/test_ci_drift_gate_arguments.py,tests/ci/test_devlog_repair_workflow.py,AGENT_HANDOFF.md"
---

Branch: `codex/13595-ci-lint`; implementation commit `45f52df558`; governance consumer commits `4e3926ac3a` (trusted `EVENT_NAME` forwarding), `0fabf6ea40` (RM-scoped read token), and `4df00351ff` (fleet runner routing). The pre-push hook guidance update is pending. The existing CI Standard repair remains unchanged. The consumer obtains gate and eligibility code from protected RM `main`, selects only an eligible repair mode, retains strict fallback and visible receipts/debt, and uses a read-only GitHub App token scoped to `Repository_Management`; the target checkout keeps its own `GITHUB_TOKEN`. Missing fork secrets fail the required token step closed. The actual local-only runner guard rejected this workflow on `ubuntu-latest`; the consumer job now routes to `d-sorg-fleet-docker`, and a test executes the guard's inline Python against the workflow fixture. The pre-push hook now preserves a nonzero result when pytest fails, points users to run pytest and fix failures, and no longer suggests bypassing the hook. Focused tests and static checks pass locally. The actual provider job also failed on a fetch error; parent is investigating its cause, so policy-job qualification remains pending. Wait for the registry extension and CI cohort drain before publication. The earlier direct gate retained existing capacity debt (31 in-progress entries / cap 8, 198 active entries / cap 75, 587,048 bytes / 200 KB) without introducing a regression. No push, PR, diagnostic suppression, policy-budget increase, fabricated task state or manual shared-log edit was performed.
