# Development Log — UpstreamDrift

State table for every feature in flight in this repository. Update entries
**in place**; never append dated sections. One entry per feature, from proposal
to ship. See the `development-logs` section of `AGENTS.md` for the binding rules
and `shared_scripts/development_log.py` for the validator.

- **Portfolio:** golf
- **WIP limit:** 8
- **Last audited:** 2026-09-08 by claude

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked` reachable
from any live state and `abandoned` from `parked`. `shipped` never returns to
`in_progress`; open a new entry instead.

## Active


### DL-#9409 · Always-on quality gate lane and conftest src-pivot guard

- **State:** in_review
- **Owner:** `claude`
- **Issue:** [#9409](https://github.com/D-sorganization/UpstreamDrift/issues/9409)
- **PR:** opened from `claude/issue-9409-always-on-gate` immediately after the `SELF` commit (body starts `Fixes #9409`)
- **Paths:** `.github/workflows/ci-standard.yml`, `tests/unit/repo_hygiene/test_no_conftest_src_module_pivot.py`, `docs/workflows/WORKFLOW_TRACKING.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`SELF`)
- **Summary:** CI Standard gains an always-on, ≤10-minute `always-on-unit-lane` (verify_installation import smoke over the shared Tools alias roots, top-level smoke tests, contract tests) that `quality-gate` requires `success` on every PR including docs-only ones; a repo-hygiene guard forbids any conftest from pivoting `sys.modules["src"]` directly (must use `EngineSrcPivot`). Deferred on #9409: main-branch cancel exemption (RM campaign) and nightly cross-engine dedupe (#8725/#9002).
- **Next step:** Verify the first CI run of the PR executes `always-on-unit-lane` to `success` within its 10-minute budget.

### DL-#9387 · Unit-Gate Worker Corruption: `src`-Identity Sentinel and Leak Fixes

- **State:** in_review
- **Owner:** claude
- **PR:** #9741 (open; `Fixes #9387`)
- **Paths:** `tests/unit/repo_hygiene/test_src_identity_sentinel.py`,
  `tests/imports/test_gui_import_boundaries.py`,
  `tests/integration/test_golf_launcher_integration.py`,
  `tests/unit/engines/pinocchio/test_tasks.py`,
  `tests/unit/test_ux_enhancements.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`c70d5ddae`)
- **Summary:** Static audit of `tests/` (conftests excluded) found 83
  `sys.modules['src*']` mutation sites in 23 files with 0 unambiguous
  leakers; the judgment-call leaks (ux-enhancements fixture, pinocchio
  tasks fixture, GUI import-boundary drops, golf-launcher pops) are now
  explicitly snapshot/restored, and a runtime sentinel runs each of the
  four documented victim files in a serial subprocess asserting
  `sys.modules['src']` identity and the `src.*` namespace are unchanged.
- **Next step:** Watch this PR's `quality-gate` run once after opening;
  on green, protected squash merge closes #9387, then rerun the flaky
  `unit-test-gate` histories of #9384/#9374/#9404 to confirm no fresh
  worker-corruption victims appear.

### DL-#9494 · Resolve the CLAUDE.md `--no-verify` Contradiction by Fixing the Windows Hook Environment

- **State:** shipped
- **Owner:** claude
- **Issue:** #9494
- **Branch:** `claude/issue-9494-precommit-env`
- **PR:** #9744
- **Paths:** `CLAUDE.md`, `AGENT_HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`, `SPEC.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`dbc6727aa`)
- **Summary:** CLAUDE.md forbade `git commit --no-verify` while agents on
  Windows reported every pre-commit invocation failing (hook virtualenvs
  targeting Python 3.11, absent from the workstation). Investigation found no
  interpreter pin left in `.pre-commit-config.yaml` (`default_language_version:
python: python`, 3.11 pin removed by #1792/#2720), and on Python 3.13.3 every
  commit-stage hook plus pre-push `mypy`/`bandit` passes after a from-scratch
  environment build. Option (a) of the issue is therefore satisfied; CLAUDE.md
  now documents the resolved environment and states that the `--no-verify`
  prohibition stands on Windows with no blanket exception.
- **Next step:** Record CI on PR #9744; on merge, confirm the protected-main
  sync lands the resolved hook environment note.

## Shipped (Last 90 Days)

Entries stay here for 90 days after merge, then move to the archive.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.
