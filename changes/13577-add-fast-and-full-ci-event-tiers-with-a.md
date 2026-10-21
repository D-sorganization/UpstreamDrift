---
issue: 13577
summary: "Add fast and full CI event tiers with a fail-closed required aggregate"
dl_state: "in_review"
next_step: "Resolve lint prerequisite #13595, run all pre-PR gates, then submit; retain live canary and seven-day latency evidence as outstanding"
branch: "codex/13577-ci-fast-full"
paths: ".github/workflows/ci-standard.yml,scripts/ci/classify_event_tier.py,tests/test_ci_standard_workflow.py,tests/test_required_check_contract.py"
---

Branch: `codex/13577-ci-fast-full`. Pinned event provider imported with byte-for-byte provenance; existing OpenSim lane classifier preserved. Initial workflow contract run was red (11 failures); event/waiver corruption tests added another eight red cases before implementation. Focused and surrounding workflow suites pass; exact commands and live acceptance limits are in `docs/operations/ci-standard-tiers.md`. Required Actionlint/ShellCheck validation is blocked by pre-existing main findings tracked in #13595; no suppression or gate bypass was used. No push, deployment, canary or speedup claim. Next step: integrate the merged lint prerequisite, complete full pre-PR validation, and submit a ready PR. Keep #13577 open until live canary and post-deployment observation evidence are recorded.
