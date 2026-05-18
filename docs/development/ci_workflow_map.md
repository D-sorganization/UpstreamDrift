# CI Workflow Map

This map is the reviewer-facing guide for the GitHub Actions surface. It
separates the workflows that represent PR quality from optional automation and
agent infrastructure so a reader does not need to infer signal from the full
workflow list.

Current inventory from `.github/workflows`:

- 59 active top-level workflow files.
- 10 archived workflow files under `.github/workflows/archived/`; GitHub does
  not discover those files as active Actions workflows.
- No workflow was deleted, disabled, or weakened for this map.

## Required PR Quality Checks

These workflows are the checks a reviewer should inspect first. They are the
quality signal for ordinary pull requests.

| Workflow file                 | Check/job to look for               | When it runs                                                      | What it proves                                                                                      |
| ----------------------------- | ----------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `ci-standard.yml`             | `quality-gate`                      | Source-code PRs and pushes to `main`; skipped for docs-only paths | Ruff, format, security lint, mypy, tests, and coverage for product code.                            |
| `docs-ci.yml`                 | `quality-gate`                      | Docs-only PRs that are intentionally ignored by `ci-standard.yml` | Docs-only changes still satisfy the same required check name without running the full source suite. |
| `spec-check.yml`              | `Verify SPEC.md freshness`          | Pull requests                                                     | Source changes either update `SPEC.md` or carry an explicit `spec-exempt` label.                    |
| `docs-governance.yml`         | `docs-governance`                   | Docs or docs-governance script changes                            | Documentation follows the repository governance checks.                                             |
| `critical-files-guard.yml`    | `Verify Critical Files Exist`       | PRs touching protected root files                                 | Required project entrypoint and governance files still exist.                                       |
| `local-only-runner-guard.yml` | `Reject hosted runner routing`      | Workflow or runner-guard script changes                           | Workflow edits do not accidentally route local-only jobs to hosted runners.                         |
| `supply-chain-guard.yml`      | `Verify all actions are SHA-pinned` | Workflow/action-pin script changes                                | External Actions references remain pinned to immutable commit SHAs.                                 |

For an employer or external reviewer, the shortest rule is: inspect
`CI Standard` for code PRs, `Docs CI` for docs-only PRs, and always check
`Spec Check` on source changes.

## Portfolio And Domain Validation Signals

These are not the whole quality gate, but they are the most relevant checks and
docs for the golf-modeling product surface:

- `tauri-build.yml` validates the desktop UI path when `ui/**` changes and on
  releases.
- `ci-optional-stack.yml` is the optional-stack verification lane for API,
  Pinocchio, Pink, Crocoddyl, and PyQt6 integration. Its main job is marked
  `continue-on-error`, so it is diagnostic rather than the core merge gate.
- `nightly-cross-engine.yml` is the scheduled cross-engine parity check for
  physics behavior across supported engines.
- [Golf Modeling Portfolio Demo](../portfolio/golf_modeling_demo.md) is the
  employer-facing demo surface referenced from the project README.
- [Physics Verification](physics_verification.md) documents the golf-physics
  validation approach used by developers.
- [Engine Support Tiers](../engines/support_tiers.md) explains which physics
  engines are expected to be production, experimental, or optional.
- [Visualization and Distribution](VISUALIZATION_AND_DISTRIBUTION.md) covers
  portfolio/demo-facing distribution work.

## Optional Scheduled Automation

These workflows provide maintenance coverage, drift detection, or scheduled
reports. Their output is useful, but they should not be read as the primary
PR-quality signal.

| Workflow file                 | Purpose                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| `ci-optional-stack.yml`       | Optional dependency-stack validation; diagnostic lane.                               |
| `docker-security-scan.yml`    | Container vulnerability scan on image-related changes, schedule, or manual dispatch. |
| `heavy-tests-opt-in.yml`      | Long-running integration tests on schedule or manual dispatch.                       |
| `nightly-cross-engine.yml`    | Scheduled cross-engine physics validation.                                           |
| `stale-cleanup.yml`           | Stale issue and PR maintenance.                                                      |
| `vendor-freshness.yml`        | Vendored submodule freshness checks and optional bumping.                            |
| `Bot-CI-Trigger.yml`          | Scheduled/manual CI retrigger support for bot-authored PRs.                          |
| `Jules-Comment-Processor.yml` | Scheduled or dispatched review-comment processing.                                   |
| `Jules-PR-Cleanup.yml`        | Scheduled or dispatched PR cleanup.                                                  |
| `Jules-PR-Compiler.yml`       | Scheduled or dispatched PR consolidation.                                            |
| `Jules-Control-Tower.yml`     | Orchestrates the Jules worker family by event, schedule, or manual dispatch.         |

## PR Plumbing And Release Workflows

These workflows support repository operations. They can affect labels,
publishing, or release artifacts, but they are not the first quality signal for
ordinary PR review.

| Workflow file                    | Purpose                                                          |
| -------------------------------- | ---------------------------------------------------------------- |
| `auto-update-prs.yml`            | Rebase/update open PRs after `main` changes.                     |
| `pr-auto-labeler.yml`            | Applies scope and size labels to PRs.                            |
| `PR-Comment-Responder.yml`       | Collects PR comments for downstream processing.                  |
| `Comment-to-Issue-Converter.yml` | Converts actionable review comments into issues.                 |
| `release.yml`                    | Tag-driven release, PyPI, Docker, and GitHub Release publishing. |
| `tauri-build.yml`                | Desktop application build and release validation.                |

## Agent And Remediation Workflows

The `Jules-*` workflows are automation workers. They are intentionally separated
from core CI and should not be interpreted as uncontrolled product-quality gates.
Most run only through `workflow_call`, `workflow_dispatch`, selected comment or
review events, or the Control Tower.

| Workflow family                 | Files                                                                                                                                                                                                                                                                                    |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Control and safety              | `Jules-Control-Tower.yml`, `jules-kill-switch.yml`, `Maintenance-Global-Control.yml`, `Manual-Run-All.yml`                                                                                                                                                                               |
| Assessment and audit            | `Jules-Assessment-Generator.yml`, `Jules-Comprehensive-Assessment.yml`, `Jules-Completist.yml`, `Jules-Documentation-Auditor.yml`, `Jules-Physics-Auditor.yml`, `Jules-Sentinel.yml`, `Jules-Tech-Debt-Assessor.yml`                                                                     |
| Remediation workers             | `Jules-Assessment-AutoFix.yml`, `Jules-Assessment-Remediator.yml`, `Jules-Auto-Repair.yml`, `Jules-Code-Quality-Fixer.yml`, `Jules-Conflict-Fix.yml`, `Jules-Hotfix-Creator.yml`, `Jules-Issue-Resolver.yml`, `Jules-PR-AutoFix.yml`, `Jules-Review-Fix.yml`, `Jules-Test-Generator.yml` |
| Documentation and communication | `Jules-Archivist.yml`, `Jules-Comment-Processor.yml`, `Jules-Critics-Comments.yml`, `Jules-Documentation-Scribe.yml`, `Jules-Laymans-Terms-Writer.yml`, `Jules-Issue-Mention-Handler.yml`                                                                                                |
| PR management                   | `Jules-Auto-Assign-Issues.yml`, `Jules-Consolidator.yml`, `Jules-PR-Cleanup.yml`, `Jules-PR-Compiler.yml`, `Jules-Supersede-Check.yml`                                                                                                                                                   |
| Technical debt                  | `Jules-Code-Quality-Reviewer.yml`, `Jules-Tech-Custodian.yml`                                                                                                                                                                                                                            |

Operational note: `.github/workflows/README.md` documents the Jules kill switch.
Create `.github/WORKFLOWS_PAUSED` or set the repository variable
`WORKFLOWS_PAUSED=true` to pause the worker family.

## Deprecated, Archived, Or Experimental

Archived workflows are inert because they live under
`.github/workflows/archived/`, not directly under `.github/workflows/`.

| Status                              | Workflow files                                                                                                                                                                                                                                                      |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Archived/inert                      | `Jules-Auto-Rebase.yml`, `Jules-Auto-Refactor.yml`, `Jules-Cleaner.yml`, `Jules-Competitor-Analyst.yml`, `Jules-Curie.yml`, `Jules-DRY-Orthogonality.yml`, `Jules-Hypatia.yml`, `Jules-Ideas-Generator.yml`, `Jules-Patent-Reviewer.yml`, `Jules-Render-Healer.yml` |
| Manual-only cost-reduced automation | `agent-metrics-dashboard.yml`, `ci-failure-digest.yml`, `Code-Metrics.yml`, `Nightly-Doc-Organizer.yml`                                                                                                                                                             |
| Legacy naming or overlap candidates | `Bot-CI-Trigger.yml`, `Comment-to-Issue-Converter.yml`, `Maintenance-Global-Control.yml`, `Manual-Run-All.yml`, `PR-Comment-Responder.yml`                                                                                                                          |

The existing operations audit at
[`docs/operations/workflow_inventory.md`](../operations/workflow_inventory.md)
contains the deeper filename inventory and rename plan. This development map is
the concise reviewer-facing layer.
