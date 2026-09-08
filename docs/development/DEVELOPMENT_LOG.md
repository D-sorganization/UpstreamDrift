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


### DL-#9499 · Spec Check Reminder Fail-Safe Extraction

- **State:** in_review
- **Owner:** claude
- **Issue:** `#9499`
- **PR:** #9719 (open; `Fixes #9499`)
- **Paths:** `.github/workflows/spec-check.yml`, `scripts/post_spec_reminder.py`, `tests/ci/test_spec_check_workflow.py`
- **Summary:** The `Verify SPEC.md freshness` job posts its SPEC reminder
  through a fail-safe script instead of an inline `github-script` heredoc, so a
  reporting failure prints the diagnostic into the job log while the
  `always()`-guarded staleness step still fails the run.
- **Next step:** Merge the PR filed from `claude/issue-9499-spec-freshness`
  after required checks pass.

## Shipped (Last 90 Days)

Entries stay here for 90 days after merge, then move to the archive.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.

## Field Reference

| Field           | Required                   | Notes                                                          |
| --------------- | -------------------------- | -------------------------------------------------------------- |
| `State`         | Always                     | One of the six states above                                    |
| `Owner`         | Always                     | Agent id from the fleet roster, or `unassigned`                |
| `Issue`         | While live                 | Governing GitHub issue; enforces the entry/issue join          |
| `Branch`        | `in_progress`, `in_review` | Enforces the entry/branch join                                 |
| `PR`            | Always                     | Number and state, or `not created`                             |
| `Paths`         | Always                     | Globs; drives silent-entry detection                           |
| `Started`       | Always                     | Drives cycle time                                              |
| `Last verified` | Always                     | Date plus SHA — the liveness signal                            |
| `Summary`       | Always                     | One or two sentences                                           |
| `Next step`     | While live                 | Exactly one action; if it needs two sentences, split the entry |
| `Parked`        | When `parked`              | Date plus reason                                               |

Never place credentials, tokens, or customer data in a development log.

### DL-#9542 · Bunker Exit State Consistency, Provenance, and Result Envelope

- **State:** in_review
- **Owner:** claude
- **Issue:** #9542
- **Branch:** `claude/issue-9542-exit-envelope`
- **PR:** #9728 (open)
- **Paths:** `src/bunkershot3d/ball/**`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-07 (`SELF`)
- **Summary:** `SandDelivery` now refuses contradictory exit speed/vector pairs and owns copies of list-supplied exit vectors so post-construction mutation cannot invalidate the frozen record; the `to_post_impact_state` boundary carries explicit `ExitVectorProvenance` labels, and `PostImpactEnvelope` wraps the flight handoff with the validity verdict, F0 tier, per-group frames, the proper `HEAD_FRAME_TO_FLIGHT_TRANSFORM`, a schema version, and a SHA-256 source digest with JSON round trip. Reflection rejection itself was already delivered by PR #9574 and is not redone.
- **Next step:** Open the protected PR to `main` with `Fixes #9542`, label `agent:claude`, and RED/GREEN evidence in the body.

### DL-#9409 · Always-On Quality Gate Lane and Conftest Src-Pivot Guard

- **State:** in_review
- **Owner:** `claude`
- **Issue:** [#9409](https://github.com/D-sorganization/UpstreamDrift/issues/9409)
- **PR:** opened from `claude/issue-9409-always-on-gate` immediately after the `SELF` commit (body starts `Fixes #9409`)
- **Paths:** `.github/workflows/ci-standard.yml`, `tests/unit/repo_hygiene/test_no_conftest_src_module_pivot.py`, `docs/workflows/WORKFLOW_TRACKING.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`SELF`)
- **Summary:** CI Standard gains an always-on, ≤10-minute `always-on-unit-lane` (verify_installation import smoke over the shared Tools alias roots, top-level smoke tests, contract tests) that `quality-gate` requires `success` on every PR including docs-only ones; a repo-hygiene guard forbids any conftest from pivoting `sys.modules["src"]` directly (must use `EngineSrcPivot`). Deferred on #9409: main-branch cancel exemption (RM campaign) and nightly cross-engine dedupe (#8725/#9002).
- **Next step:** Verify the first CI run of the PR executes `always-on-unit-lane` to `success` within its 10-minute budget.

### DL-#9533 · Test-Only Extras Reachable From the Dev Lock

- **State:** in_review
- **Owner:** claude
- **PR:** #9716 (open; `Fixes #9533`)
- **Paths:** `pyproject.toml`, `.github/workflows/lock-refresh.yml`,
  `requirements*.lock`, `environment.yml`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`289b3aac2`)
- **Summary:** `openpyxl` and `imageio` were declared only in the `gui-tools`
  and `pose` extras, so the dev-compiled `requirements-dev.lock` never installed
  them and ~24 CI tests failed on import. Both now resolve through the `dev`
  extra; lock regeneration is delegated to a dispatch-only `lock-refresh.yml`
  workflow that runs `make sync-deps` on ubuntu + Python 3.12 and opens a PR,
  since Windows/WSL cannot regenerate correctly (#9533).
- **Next step:** Regenerate the locks via `lock-refresh.yml` so the
  dependency-consistency freshness gate and the 24 previously failing tests
  go green.

### DL-#9091 · Phantom-Guard Rule-3 False Positive on Shallow Base Fetch

- **State:** in_review
- **Owner:** claude
- **PR:** #9717 (open; `Fixes #9091`)
- **Paths:** `.github/workflows/anti-phantom-merge.yml`, `scripts/ci/check_phantom_guard_paths.py`, `tests/scripts/test_check_phantom_guard_paths.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`SELF`)
- **Summary:** The phantom-guard rule-3 closes-issue path-membership check
  produced a false positive when a shallow base fetch defeated `git merge-base`;
  the check is now extracted into `scripts/ci/check_phantom_guard_paths.py`,
  which resolves changed files from local git diff first and falls back to the
  GitHub API changed-file list (`PR_CHANGED_FILES` / `gh pr view --json files`)
  when merge-base fails, failing closed with a diagnostic when neither source
  is available. Workflow rule-3 block invokes the script; rules 1/2/4 unchanged.
- **Next step:** Observe the first post-merge `phantom-guard` run on a PR
  whose base fetch is too shallow for merge-base to confirm rule 3 defers to
  the API list.

## Shipped (Last 90 Days)

Entries stay here for 90 days after merge, then move to the archive.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.

## Field Reference

| Field           | Required                   | Notes                                                          |
| --------------- | -------------------------- | -------------------------------------------------------------- |
| `State`         | Always                     | One of the six states above                                    |
| `Owner`         | Always                     | Agent id from the fleet roster, or `unassigned`                |
| `Issue`         | While live                 | Governing GitHub issue; enforces the entry/issue join          |
| `Branch`        | `in_progress`, `in_review` | Enforces the entry/branch join                                 |
| `PR`            | Always                     | Number and state, or `not created`                             |
| `Paths`         | Always                     | Globs; drives silent-entry detection                           |
| `Started`       | Always                     | Drives cycle time                                              |
| `Last verified` | Always                     | Date plus SHA — the liveness signal                            |
| `Summary`       | Always                     | One or two sentences                                           |
| `Next step`     | While live                 | Exactly one action; if it needs two sentences, split the entry |
| `Parked`        | When `parked`              | Date plus reason                                               |

Never place credentials, tokens, or customer data in a development log.

### DL-#9249 · UI: Pin @vitejs/Plugin-React to ^5 Until Vite 8

- **State:** in_review
- **Owner:** claude
- **PR:** #9718 (open; `Fixes #9249`)
- **Paths:** `.github/dependabot.yml`, `ui/README.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`7cdbb0a3d`)
- **Summary:** Dependabot ignores `@vitejs/plugin-react` major updates
  because 6.x needs Vite 8 (Vite 7 exports no `./internal`); the pairing
  constraint is documented in `ui/README.md`.
- **Next step:** Merge the guard PR; revisit the paired vite@8 +
  plugin-react@6 upgrade once `vitest`/`@react-three/*` are Vite-8 ready.

### DL-#9470 · Launch-Monitor Analysis Handlers Onto the Async_Action Worker

- **State:** in_review
- **Owner:** claude
- **Issue:** #9470
- **PR:** #9742
- **Paths:** `src/tools/launch_monitor_analytics/gui.py`, `src/tools/launch_monitor_analytics/_embed_adapter.py`, `tests/ui/tools/launch_monitor/test_async_actions.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`SELF` on this branch: `tests/ui/tools/launch_monitor` + `tests/tools/test_async_action.py` all passing, PyQt6 6.11.0 offscreen)
- **Summary:** All seven analysis handlers (`treatment`, `relationship`, `multivariate`, `model`, `comparison`, `dispersion`, `trend`) now run their compute on the #8880 `async_action` worker via one shared `AsyncActionBar`; synchronous `present(compute())` paths kept; embed adapter `cleanup()` cancels and joins the worker. First slice of the #9470 tool checklist; the remaining tools are follow-ups.
- **Next step:** Merge PR #9472 (the #8880 mechanism) before this branch — it is stacked on `readiness/p2-8880-async-action-worker`.

### DL-#9387 · Unit-Gate Worker Corruption: `src`-Identity Sentinel and Leak Fixes

- **State:** shipped
- **Owner:** claude
- **PR:** #9741 (merged; `Fixes #9387`)
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

### DL-#9476 · Re-Vendor the Corrected Spec Merge Driver and Pin Drift

- **State:** in_review
- **Owner:** claude
- **Issue:** `#9476`
- **Branch:** `claude/issue-9476-driver-wiring`
- **PR:** #9736 (open, in_review; re-vendor leg #9734)
- **Paths:** `scripts/install_spec_merge_driver.py`,
  `shared_scripts/spec_changelog.py`,
  `tests/unit/scripts/test_spec_merge_driver_vendor_drift.py`,
  `scripts/setup_hooks.py`,
  `tests/unit/scripts/test_setup_hooks_wires_spec_merge_driver.py`, `SPEC.md`,
  `AGENT_HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`875cd501e`)
- **Summary:** The vendored installer still stamped the withdrawn
  merge-abort claim into `$GIT_COMMON_DIR/info/attributes` and both vendored
  copies had no drift detection. Re-vendored both files byte-identical from
  Repository_Management#1521's corrected copies (verified: the false
  `ATTRIBUTE_BLOCK` text is gone; the repo-wide scan's only surviving match
  in UD-owned files is the correction narrative quoting the wrong claim to
  refute it, plus true statements about the half-configured state), and added
  `tests/unit/scripts/test_spec_merge_driver_vendor_drift.py` pinning SHA-256
  digests against the upstream reference so future divergence fails loudly.
  The behaviour change the issue requests followed on this branch:
  `scripts/setup_hooks.py` (the documented local-automation entry point)
  now calls the vendored installer, so the documented setup registers the
  `spec-rows` driver, and the installer docstring names this repository's
  entry point instead of Repository_Management's.
- **Next step:** Land PR #9734, then the wiring PR.

### DL-#9533 · Test-Only Extras Reachable From the Dev Lock

- **State:** in_review
- **Owner:** claude
- **Issue:** `#9533`
- **Branch:** `claude/issue-9533-test-extras`
- **PR:** #9716 (open, in_review)
- **Paths:** `pyproject.toml`, `.github/workflows/lock-refresh.yml`, `requirements*.lock`, `environment.yml`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`289b3aa`)
- **Summary:** `openpyxl` and `imageio` were declared only in the `gui-tools` and `pose` extras, so the dev-compiled `requirements-dev.lock` never installed them and ~24 CI tests failed on import. Both now resolve through the `dev` extra; lock regeneration is delegated to a dispatch-only `lock-refresh.yml` workflow that runs `make sync-deps` on ubuntu + Python 3.12 and opens a PR, since Windows/WSL cannot regenerate correctly (#9533).
- **Next step:** Dispatch `.github/workflows/lock-refresh.yml` from `main` once this PR merges, then confirm the `ci-standard.yml` dependency-consistency freshness gate and the 24 previously failing tests go green.

### DL-#9476 · Re-Vendor the Corrected Spec Merge Driver and Pin Drift

- **State:** in_review
- **Owner:** claude
- **Issue:** `#9476`
- **Branch:** `claude/issue-9476-spec-merge-driver`
- **PR:** #9734 (open, in_review)
- **Paths:** `scripts/install_spec_merge_driver.py`,
  `shared_scripts/spec_changelog.py`,
  `tests/unit/scripts/test_spec_merge_driver_vendor_drift.py`, `SPEC.md`,
  `AGENT_HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`3fd347b72`)
- **Summary:** The vendored installer still stamped the withdrawn
  merge-abort claim into `$GIT_COMMON_DIR/info/attributes` and both vendored
  copies had no drift detection. Re-vendored both files byte-identical from
  Repository_Management#1521's corrected copies (verified: the false
  `ATTRIBUTE_BLOCK` text is gone; the repo-wide scan's only surviving match
  in UD-owned files is the correction narrative quoting the wrong claim to
  refute it, plus true statements about the half-configured state), and added
  `tests/unit/scripts/test_spec_merge_driver_vendor_drift.py` pinning SHA-256
  digests against the upstream reference so future divergence fails loudly.
  Registration wiring into `scripts/setup_hooks.py` is the companion PR the
  issue requests as a separate behaviour change.
- **Next step:** Land the re-vendor PR, then open the wiring PR on
  `claude/issue-9476-driver-wiring`.

### DL-#9533 · Test-Only Extras Reachable From the Dev Lock

- **State:** in_review
- **Owner:** claude
- **Issue:** `#9533`
- **Branch:** `claude/issue-9533-test-extras`
- **PR:** #9716 (open, in_review)
- **Paths:** `pyproject.toml`, `.github/workflows/lock-refresh.yml`, `requirements*.lock`, `environment.yml`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`289b3aa`)
- **Summary:** `openpyxl` and `imageio` were declared only in the `gui-tools` and `pose` extras, so the dev-compiled `requirements-dev.lock` never installed them and ~24 CI tests failed on import. Both now resolve through the `dev` extra; lock regeneration is delegated to a dispatch-only `lock-refresh.yml` workflow that runs `make sync-deps` on ubuntu + Python 3.12 and opens a PR, since Windows/WSL cannot regenerate correctly (#9533).
- **Next step:** Dispatch `.github/workflows/lock-refresh.yml` from `main` once this PR merges, then confirm the `ci-standard.yml` dependency-consistency freshness gate and the 24 previously failing tests go green.

### DL-#9733 · Fail Fast on the Uninitialized Vendored Tools Fallback

- **State:** in_review
- **Owner:** claude
- **Issue:** #9733
- **Branch:** `claude/issue-9733-fail-fast`
- **PR:** #9743 (open; `Fixes #9733`)
- **Paths:** `src/__init__.py`, `tests/unit/repo_hygiene/test_src_fallback_fail_fast_9733.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`160f9b759`)
- **Summary:** When `vendor/ud-tools` is uninitialized, `src/__init__.py`'s
  fallback registration mistook UpstreamDrift's own aliased `shared.python` copy
  for an installed Tools distribution, installed `_VendoredToolsFallbackFinder`,
  and livelocked pytest collection in meta-path `find_spec` recursion. The
  registration probe now raises an actionable ImportError naming the remediation
  command before any finder is installed; the initialized path is unchanged.
- **Next step:** Record CI on PR #9743; on green, protected squash merge
  closes #9733.

### DL-#9631 · Vendor Pin Carries the Tools#5048 Alias-Predicate Fix

- **State:** in_review
- **Owner:** claude
- **Issue:** `#9631`
- **Branch:** `claude/issue-9631-vendor-pin`
- **PR:** #9722 (open; `Fixes #9631`)
- **Paths:** `vendor/ud-tools`, `tests/unit/repo_hygiene/test_pinned_import_alias_contract.py`
- **Started:** `2026-09-08`
- **Last verified:** `2026-09-08` (`e4c47751f`)
- **Summary:** The `vendor/ud-tools` pin `eab74a901a` already carries the Tools#5049
  flattened-install fix (`f8b94bfe` is an ancestor), so the v2.1.2 wheel defect is fixed at
  the pin; this entry lands the repository's own TDD contract test asserting the pinned
  predicate in both layouts and records that the pin must not be rewound.
- **Next step:** maintainer re-cuts the 2.1.3 release via tag/workflow dispatch after the PR merges.

## Shipped (Last 90 Days)

Entries stay here for 90 days after merge, then move to the archive.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.
