# Implementation Handoff - Repository Root Allowlist Check in Docs Governance (#9415)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/agy-ud-9415-root-allowlist`
- Branch: `agy/ud-9415-root-allowlist`
- Baseline commit: `28b37bd47eb85d977e530e60659c6f54ca0a1a2b` (origin/main)
- Pull request: draft, opened from `agy/ud-9415-root-allowlist`
- Governing issue: #9415 (development log DL-#9415)
- Lease session: `claude-deskcomputer-20260925-ud`

## Objective and Status

- Objective: Stop new clutter at the repository root: the root-allowlist checkbox of #9415.
- Status: in_review (draft PR)
- Completed:
  1. `scripts/config/root_allowlist.json` generated from `git ls-tree --name-only HEAD` (78 entries).
  2. `_load_root_allowlist` (DbC: ValueError on missing/non-list entries, non-strings, duplicates, separators, empties), pure `_unexpected_root_entries` / `_stale_root_entries`, and `_root_allowlist_failure` called from `main()`; git failure fails closed.
  3. Existing docs-governance tests stub the two new seams; new unit tests cover each rule and the real repository.
  4. One paragraph in `docs/governance/DOCS_GOVERNANCE.md`.

## Validation

- `python scripts/check_docs_governance.py` -> `docs governance checks passed`.
- `pytest tests/scripts/test_doc_governance_checks.py tests/unit/scripts/test_check_docs_governance_root_allowlist.py` -> 26 tests, 0 failures (junit).
- `python scripts/ci/check_architecture_budget.py` -> OK.

## Next Steps

1. CI green, owner review, merge.
2. #9415 remaining items (history rewrite, five Jules workflows, SPEC.md cap) need owner decisions; leave the issue open.

## Change Log

- `SELF` — Repository Root Allowlist Check in Docs Governance (#9415, DL-#9415).
