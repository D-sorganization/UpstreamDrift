# Implementation Handoff

Keep this file current and concise. Replace instructional placeholders; do not append an unbounded transcript.

## Identity

- Repository: `D-sorganization/UpstreamDrift`
- Working directory: `C:/tmp/UD_w9091`
- Branch: `claude/issue-9091-phantom-path-api`
- Baseline commit: `191351bdf84b8b46e2c95dddc073fc8555ce4b12`
- Implementation commit: `SELF` — the commit containing this update; resolve with `git rev-parse HEAD`
- Pull request: `not created | not applicable`
- Governing issue/epic: `UD #9091` (governed workflow campaign: Repository_Management#1505 / #1507)

## Objective and Status

- Objective: Stop the phantom-guard rule-3 (closes-issue path mismatch) false positive when a shallow base fetch defeats `git merge-base`, by routing the check through the GitHub API changed-file list fallback.
- Status: `ready for review`
- Completed: Path-membership logic extracted from inline `run:` blocks into `scripts/ci/check_phantom_guard_paths.py` (git-diff-first, API-list fallback, fail-closed diagnostic); unit-tested with the API layer mocked (22 focused tests, RED→GREEN captured); workflow rule-3 block replaced by a script invocation; YAML parse validated.
- Remaining: Post-merge observation of `phantom-guard` on the next real PR hitting the shallow-fetch condition (CI-only behavior).

## Files and Decisions

- Files changed: `.github/workflows/anti-phantom-merge.yml` (rule-3 block → script call, minimal edit); `scripts/ci/check_phantom_guard_paths.py` (new gate script); `tests/scripts/test_check_phantom_guard_paths.py` (new); `docs/development/HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md` (fleet-required, new).
- Key decisions: Resolution order is git-diff-on-merge-base-success → caller-supplied `PR_CHANGED_FILES` list → `gh pr view --json files` (the API pattern the workflow already uses) → fail closed with diagnostic; an empty diff from a *successful* merge-base is a valid result and does not trigger the API fallback (preserves rule-3 semantics on genuinely empty PRs; the race retry stays with the workflow's count check). Rules 1/2/4 remain inline — they depend on the count-fallback state and are not the false-positive surface this issue describes.
- User-owned or unrelated worktree changes: `none observed`

## Validation

- `python -m pytest tests/scripts/test_check_phantom_guard_paths.py -q` — pass (22 passed; API layer mocked, no network).
- `ruff check scripts/ci/check_phantom_guard_paths.py tests/scripts/test_check_phantom_guard_paths.py` — pass.
- `python -c "import yaml; yaml.safe_load(open('.github/workflows/anti-phantom-merge.yml'))"` — pass (YAML_OK).
- Shell harness of the new rule-3 wiring with a bogus base SHA and no `GH_TOKEN` — exit 1 with the fail-closed diagnostic forwarded to `fail()` (expected failure; proves fail-closed path).

## Blockers and Risks

- Blockers: `none`
- Risks/assumptions: In CI the script re-derives changed files for rule 3 instead of reusing the workflow's `CHANGED_FILES` variable — one extra `gh pr view` call only in the merge-base-failure path; kept so the script is self-sufficient and the workflow edit stays minimal.

## Next Steps

1. Watch the first `phantom-guard` run after this PR merges to confirm rule 3 defers to the API list on shallow-fetch PRs.

## Change Log

- `SELF` — Initial handoff: extracted rule-3 path-membership gate into `scripts/ci/check_phantom_guard_paths.py` with API-list fallback and fail-closed diagnostics; workflow invokes the script.
