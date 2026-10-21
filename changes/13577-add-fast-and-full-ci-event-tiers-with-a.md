---
issue: 13577
summary: "Add fast and full CI event tiers with a fail-closed required aggregate"
dl_state: "in_review"
next_step: "Resolve generated-log governance blocker RM #2253 and lint prerequisite #13595, then submit; live canary and seven-day latency evidence remain outstanding"
branch: "codex/13577-ci-fast-full"
paths: ".github/workflows/ci-standard.yml,scripts/ci/classify_event_tier.py,tests/test_ci_standard_workflow.py,tests/test_required_check_contract.py"
---

Branch: `codex/13577-ci-fast-full`. Pinned event provider imported with byte-for-byte provenance; existing OpenSim lane classifier preserved. Initial workflow contract run was red (11 failures); event/waiver corruption tests added another eight red cases before implementation. Focused and surrounding workflow suites pass; exact commands and live acceptance limits are in `docs/operations/ci-standard-tiers.md`. Required Actionlint/ShellCheck validation is blocked by pre-existing main findings tracked in #13595; no suppression or gate bypass was used. No push, deployment, canary or speedup claim. Next step: integrate the merged lint prerequisite, complete full pre-PR validation, and submit a ready PR. Keep #13577 open until live canary and post-deployment observation evidence are recorded.

Final independent read-only review of the refreshed source snapshot ran on DeskComputer WSL with Codex CLI 0.156.1 and gpt-6-luna, exit 0; no concrete reproducible mismatch remained. The reviewer did not execute tests or establish live GitHub scheduling acceptance. The local focused suite passed 122 cases. Existing CI infrastructure tests passed 89 cases; the selected surrounding workflow suite previously passed 161 with one optional skip.

Submission is also blocked by Repository_Management #2253: the canonical pre-PR Fleet Policy Gate rejects the unchanged generated development log (31 in-progress entries / cap 8; 198 active / cap 75; 587,048 bytes / 200 KB). The lint prerequisite itself passes Actionlint/ShellCheck and its affected/consistency tests. No manual log edits, state fabrication, cap increase, gate bypass or push was made.
