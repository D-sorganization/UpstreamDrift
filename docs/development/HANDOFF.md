# Implementation Handoff

Keep this file current and concise. Replace instructional placeholders; do not append an unbounded transcript.

## Identity

- Repository: `D-sorganization/UpstreamDrift`
- Working directory: `C:/tmp/UD_w9733`
- Branch: `claude/issue-9733-fail-fast`
- Baseline commit: `dbc6727aa` (origin/main)
- Implementation commit: `SELF` — the commit containing this update; resolve with `git rev-parse HEAD`
- Pull request: not created (opened on push against `main`)
- Governing issue/epic: #9733 (pytest livelock with `vendor/ud-tools` uninitialized)

## Objective and Status

- Objective: `import src` fails fast with an actionable ImportError when the
  `vendor/ud-tools` submodule is uninitialized, instead of livelocking pytest
  collection in fallback-finder `find_spec` recursion.
- Status: ready for review
- Completed: guard implemented in `src/__init__.py` (probe treats a
  `shared.python` spec resolving only to UpstreamDrift's own aliased copy as
  "no Tools present" and raises, naming `git submodule update --init
  vendor/ud-tools`); regression test `tests/unit/repo_hygiene/
  test_src_fallback_fail_fast_9733.py` (6 cases incl. a `python -O`
  subprocess check) RED→GREEN; livelock reproduced pre-fix via bounded
  timeout + faulthandler dumps.
- Remaining: PR review/merge. Nothing else.

## Files and Decisions

- Files changed: `src/__init__.py` (fail-fast guard in
  `_register_vendored_tools_fallback`, new
  `_shared_python_spec_is_ud_alias`); `tests/unit/repo_hygiene/
  test_src_fallback_fail_fast_9733.py` (new, distinct file name per fleet
  rule); `SPEC.md` changelog row; `AGENT_HANDOFF.md` + this file; `docs/
  development/DEVELOPMENT_LOG.md` (created, entry `DL-#9733`).
- Key decisions: the guard fires only when the vendored tree is missing AND
  `shared.python` resolves to the repo's own alias — a genuine installed Tools
  distribution (Tools#5048) still registers the fallback, and a checkout with
  no Tools dependency at all keeps today's graceful skip. Explicit `raise`
  (survives `python -O`). The probe is simulated by monkeypatch in tests, so
  the real submodule state is never touched.
- User-owned or unrelated worktree changes: none observed.

## Validation

- `python -m pytest tests/unit/repo_hygiene/test_src_fallback_fail_fast_9733.py -p no:warnings` — 6 passed.
- `python -m pytest tests/unit/repo_hygiene -q -p no:warnings --tb=no` — failure set byte-identical to the pristine-`main` baseline (7 pre-existing environment failures).
- `python -m pytest tests/unit/api -q -p no:warnings --tb=no` — failure set identical to baseline (17 pre-existing; only mock-id/temp-path text differs).
- `python -m pytest tests/unit/engines/myosuite/test_canonical_adapter.py -p no:warnings` — 6 passed (engine `src` pivot intact).
- `python shared_scripts/fleet_hooks.py spec-changelog` — passed; `python C:/tmp/repo_mgmt/shared_scripts/development_log.py --changed docs/development/DEVELOPMENT_LOG.md` — OK.
- Environment: Windows, Python 3.13.3, pytest 9.0.3, ruff 0.15.6. Heavy native stacks not installed locally; CI validates those lanes.

## Blockers and Risks

- Blockers: none.
- Risks/assumptions: the raise assumes a checkout whose only `shared.python`
  resolution is the repo alias has no usable Tools tree; cross-checkout
  sibling resolutions are treated as installed Tools (unchanged behavior).

## Next Steps

1. Open the PR against `main` (`Fixes #9733`), label `agent:claude`.
2. Watch the single required CI run once after opening; record results on the PR.

## Change Log

- `SELF` — created the canonical handoff recording the #9733 fail-fast fix, its RED/GREEN evidence, and the PR state.