# Implementation Handoff

Keep this file current and concise. Replace instructional placeholders; do not append an unbounded transcript.

## Identity

- Repository: `D-sorganization/UpstreamDrift`
- Working directory: `C:/tmp/UD_w9409` (per-issue fleet workspace)
- Branch: `claude/issue-9409-always-on-gate`
- Baseline commit: `191351b` (`191351b` = `feat(reconstruct): many tape-measured segments ... (#9708)`, head of `main` at branch time)
- Implementation commit: `SELF` — the commit containing this update; resolve with `git rev-parse HEAD`
- Pull request: opened immediately after this commit is pushed (`Fixes #9409`); see the branch's open PR list
- Governing issue/epic: D-sorganization/UpstreamDrift#9409 (program: D-sorganization/Repository_Management#1505; workflow governance RM #1505 / #1507)

## Objective and Status

- Objective: make the required `quality-gate` honest for every PR by adding an always-on minimal unit lane, and prevent the conftest `sys.modules["src"]` pivot failure class (issue #9402 / PR #9404) from recurring.
- Status: in_review
- Completed:
  - `.github/workflows/ci-standard.yml`: new `always-on-unit-lane` job (no `if:` gate, `timeout-minutes: 10`) running `scripts/ci/verify_installation.py` (core dependency imports, `src.*` suite module imports, and the shared Tools alias-root import smoke), the top-level `tests/smoke` test files, and `tests/shared_contracts/`. `quality-gate` now `needs` it unconditionally and requires result `success` even on docs-only PRs. The literal `cancel-in-progress: true` is untouched; the main-branch cancel exemption is deliberately NOT implemented here (separate RM campaign).
  - `tests/unit/repo_hygiene/test_no_conftest_src_module_pivot.py`: static AST guard — no `conftest.py` anywhere in the repo may bind/delete/`pop`/`setdefault` `sys.modules["src"]` directly; the shadow must go through `tests.helpers.engine_src_pivot.EngineSrcPivot`. TDD: RED verified by a scratch conftest pivot, then removed for GREEN.
  - `docs/workflows/WORKFLOW_TRACKING.md`: documents the always-on lane.
  - `docs/development/DEVELOPMENT_LOG.md`: `DL-#9409` entry created.
- Remaining (deliberately deferred, tracked on #9409):
  - Main-branch `cancel-in-progress` exemption — governed by the RM campaign, out of scope here.
  - Nightly cross-engine dedupe (#8361 canonical; #8725/#9002) — not touched.
  - Branch-protection ruleset change to require `lod-quality-gate` alongside the literally-named `quality-gate` context — a repository-settings change that cannot be made from a PR; note for the repo owner (see `quality-gate.yml` header comment).

## Files and Decisions

- Files changed:
  - `.github/workflows/ci-standard.yml` — added `always-on-unit-lane` job; extended `quality-gate` needs/env/aggregate script.
  - `tests/unit/repo_hygiene/test_no_conftest_src_module_pivot.py` — new hygiene guard (ratchet ledger `_PREEXISTING_SRC_PIVOT_CONFTESTS` is empty and must only shrink).
  - `docs/workflows/WORKFLOW_TRACKING.md`, `docs/development/HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md` — governance/bookkeeping.
- Key decisions:
  - Reuse `scripts/ci/verify_installation.py` for the import smoke rather than adding a new script: it already resolves every shared Tools alias root through `SharedImportAliasFinder` (`check_shared_alias_roots`) and imports the `src.*` suite modules.
  - The lane installs the same dependency set as `unit-test-gate` (`requirements-dev.lock` + editable install + hypothesis toolcache hygiene + numpy/scipy ABI pin) minus the xvfb/apt steps, because the lane only runs headless-safe tests.
  - The guard is static (AST) so it cannot itself pivot modules while checking; the two legitimate pivot conftests pass because they delegate to `EngineSrcPivot`.
- User-owned or unrelated worktree changes: none observed.

## Validation

- `python -m pytest tests/unit/repo_hygiene/test_no_conftest_src_module_pivot.py -p no:cacheprovider --no-cov -q` — RED with scratch pivot conftest present (guard fails listing the scratch file), GREEN after removal (2 passed).
- `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci-standard.yml', encoding='utf-8'))"` — parses; job count 29 with `always-on-unit-lane` present in `jobs` and in `quality-gate.needs`. Parse-only: execution is validated by CI.
- Focused local test runs only; the machine is shared with concurrent fleet agents, so no repo-wide suite was run. Heavy native stacks were not installed locally; CI validates those (disclosed in the PR body).