# Implementation Handoff - Coverage Gate Checker Reads the Budget File and Maps Cobertura Sources (#10965)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/agy-ud-10965-coverage-gates`
- Branch: `agy/ud-10965-coverage-gates`
- Baseline commit: `28b37bd47eb85d977e530e60659c6f54ca0a1a2b` (origin/main)
- Pull request: draft, opened from `agy/ud-10965-coverage-gates`
- Governing issue: #10965 (development log DL-#10965)
- Lease session: `claude-deskcomputer-20260925-ud`

## Objective and Status

- Objective: Make the budget file's coverage gates measured and enforceable (#10965).
- Status: in_review (draft PR)
- Completed:
  1. agy (Gemini 3.8 Flash) on OG Laptop: single gate authority, XML+JSON loaders, DbC budget validation, unit tests.
  2. Orchestrator review: agy's `"src/" in path` heuristic mapped nothing on a real `--cov=src` report (all six gates MISCONFIGURED); replaced by `<source>` resolution with tests written first (3 red -> green), plus `--repo-root`.
  3. Measured the unit lane on OG (`pytest -m unit -n 8 --cov=src`); three xdist workers crashed, so the report undercounts.

## Validation

- `pytest tests/unit/scripts/test_check_coverage_gates.py tests/unit/scripts/test_check_mypy_exclusion_budget.py` -> 23 passed.
- `python scripts/check_coverage_gates.py --report coverage_measured.xml` -> exit 1 with the per-gate table in the PR body.
- `python scripts/check_mypy_exclusion_budget.py --today 2026-10-02` still fails: mypy exclusion expiries are handled by #10969/#10973, coverage ratchet dates are blocked until 2026-10-01 by the #8731 pin.

## Next Steps

1. On or after 2026-10-01, re-date the six ratchets to 2027-01-01 with an owner decision on the gates measured below their floor, then wire the checker after the ci-standard `--cov=src` step.
2. Owner decision: five gates measure below their 30% floor on the unit lane; either add tests or lower a floor with a `Tolerance-Change-Evidence:` trailer - never silently.

## Change Log

- `SELF` — Coverage Gate Checker Reads the Budget File and Maps Cobertura Sources (#10965, DL-#10965).
