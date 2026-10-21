---
issue: 13577
summary: "Add fast PR/push and full merge-queue/scheduled tiers with a fail-closed required aggregate"
dl_state: "in_review"
next_step: "Complete canonical full pre-PR validation with the published trusted provider; then collect live merge-group canary and seven-day latency evidence"
branch: "codex/13577-ci-fast-full"
paths: ".github/workflows/ci-standard.yml,scripts/ci/classify_event_tier.py,tests/test_ci_standard_workflow.py,tests/test_required_check_contract.py"
---

Branch: `codex/13577-ci-fast-full`. The branch now includes actual `origin/main` commit `1a7779993cfe3a99cabcbbfdbf4313d6a53ccbc3` via normal merge commit `54a1558ce65300d09cb1b1131258c8180b1bca1c`; required start `8a5c5b044b81d2cacb58004350146ad8fa46171f` remains an ancestor. The pinned event provider and provenance are unchanged; the existing OpenSim lane classifier and CI Standard required context are preserved.

After the merge, 323 focused CI/workflow tests passed, including tier mapping, required aggregate failures/cancellations, fork-runner guard, local-only workflow routing and the merged fleet-policy consumer. Actionlint 1.7.7 with ShellCheck passed on `ci-standard.yml` and `fleet-policy-gate.yml`. The full canonical pre-PR run with the published trusted provider remains pending; no push, live canary, deployment, or speedup claim is made. Keep #13577 open until the required live merge-group canary and seven-day latency evidence are recorded.
