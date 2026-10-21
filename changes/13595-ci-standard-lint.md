---
issue: 13595
summary: "Clear CI Standard Actionlint and ShellCheck failures without changing workflow selection or guards"
dl_state: "in_review"
next_step: "Parent review on codex/13595-ci-lint; resolve any review feedback, then rerun the canonical gate in a complete project environment."
title: "CI Standard Lint Gate"
owner: "codex"
branch: "codex/13595-ci-lint"
paths: ".github/actionlint.yaml,.github/workflows/ci-standard.yml,tests/ci/test_ci_infrastructure.py,tests/ci/test_ci_drift_gate_arguments.py,AGENT_HANDOFF.md"
---

Branch codex/13595-ci-lint at 5f78887858; Actionlint 1.7.7/ShellCheck 0.11.0 green. Canonical pre_pr.py with the UpstreamDrift .venv passed lint/format, diff mypy, affected tests (89 passed), consistency (about five minutes), Semgrep/import policy, policy/fragment validators, and workflow lint. Fleet Policy Gate fails on existing DEVELOPMENT_LOG.md WIP/size ceilings (31 in-progress vs 8, 198 active vs 75, >200 KB). The focused CI/workflow suite including the two argv cases passed with one existing skip. No push or PR; parent review pending.
