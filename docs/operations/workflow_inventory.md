# GitHub Workflow Inventory

**Generated:** 2026-04-19
**Source issue:** #2717 ("CI / docs / hygiene: 56 workflows, formatter drift, committed artifacts, duplicate LICENSE")
**Actual count at audit time:** 56 workflow files under `.github/workflows/` (+ 12 under `.github/workflows/archived/`)

This document is an audit of every workflow file in the repository. It is the
authoritative map of triggers, purposes, and naming groups. It is the intended
prerequisite for a follow-up **rename sprint** — but no files have been renamed
in this PR because the existing names are referenced by:

- Branch-protection required-check rules (`CI Standard`, `Docs Governance`, `Spec Check`, `Secret Scan`)
- Cross-repo `workflow_dispatch` / `workflow_call` callers (Jules fleet)
- Fleet daemons (GAAI) that match workflow names by regex
- `Jules-Control-Tower.yml` dispatches the `Jules-*` worker family by filename

Renames must be coordinated; see "Follow-up rename plan" at the bottom.

This slice of #2717 addresses the **workflow naming audit** sub-concern only.
Remaining sub-concerns (formatter drift, committed artifacts, duplicate LICENSE
verification) are tracked by PRs #2725, #2735, #2812, #2822 and their
follow-ups.

---

## Naming conventions currently in use

The repo mixes **three** conventions. This is the primary orphan/overlap risk
flagged in #2717.

| Convention          | Count | Example               |
| ------------------- | ----- | --------------------- |
| `Title-Case-Dashed` | 7     | `Bot-CI-Trigger.yml`  |
| `Jules-*` (fleet)   | 32    | `Jules-Archivist.yml` |
| `kebab-case`        | 18    | `ci-standard.yml`     |

Goal state (follow-up): collapse everything to `kebab-case` with a small set of
prefixes — `ci-*`, `docs-*`, `jules-*`, `pr-*`, `ops-*`, `release-*`.

---

## Inventory

Triggers abbreviations: `PR` = `pull_request`, `P` = `push`,
`WD` = `workflow_dispatch`, `WC` = `workflow_call`, `WR` = `workflow_run`,
`S` = `schedule`, `IC` = `issue_comment`, `I` = `issues`, `RV` = `pull_request_review`,
`RVC` = `pull_request_review_comment`, `R` = `release`, `—` = no triggers.

### Group: CI / Quality Gates (kebab-case)

| File                       | `name:` field                   | Triggers     | Purpose                                               |
| -------------------------- | ------------------------------- | ------------ | ----------------------------------------------------- |
| `ci-standard.yml`          | CI Standard                     | P, PR, WD, S | Primary quality gate (ruff, format, tests, coverage). |
| `ci-optional-stack.yml`    | CI Optional Stack               | P, PR, WD, S | Optional engine stack (Drake / Pinocchio / MuJoCo).   |
| `ci-failure-digest.yml`    | Weekly CI Failure Digest        | WD           | Weekly rollup of failing required checks.             |
| `spec-check.yml`           | Spec Check                      | PR           | `SPEC.md` freshness / specification-exempt gate.      |
| `critical-files-guard.yml` | Critical Files Guard            | PR           | Blocks unauthorised edits to critical files.          |
| `docker-security-scan.yml` | Docker Security Scan            | P, PR, S, WD | Trivy / grype scan on published images.               |
| `docker-size-gates.yml`    | Docker Size and Health Check    | P, WD        | Image-size budget enforcement.                        |
| `heavy-tests-opt-in.yml`   | Heavy Integration Tests         | WD, S        | Opt-in long-running integration suite.                |
| `nightly-cross-engine.yml` | Nightly Cross-Engine Validation | S, WD        | Nightly cross-engine parity checks.                   |

### Group: Docs (kebab-case)

| File                  | `name:` field   | Triggers | Purpose                                 |
| --------------------- | --------------- | -------- | --------------------------------------- |
| `docs-ci.yml`         | Docs CI         | PR       | Docs build (Sphinx/MkDocs) on PRs.      |
| `docs-governance.yml` | Docs Governance | PR, P    | Docs quality / placeholder / link gate. |

### Group: PR Plumbing (kebab-case)

| File                  | `name:` field          | Triggers | Purpose                                   |
| --------------------- | ---------------------- | -------- | ----------------------------------------- |
| `auto-update-prs.yml` | Auto-Update PRs        | P        | Rebase open PRs onto freshly-merged main. |
| `pr-auto-labeler.yml` | PR Auto-Labeler        | PR       | Applies scope / size labels to PRs.       |
| `stale-cleanup.yml`   | Stale PR/Issue Cleanup | S, WD    | Closes stale issues and PRs.              |

### Group: Release / Build (kebab-case)

| File              | `name:` field | Triggers     | Purpose                         |
| ----------------- | ------------- | ------------ | ------------------------------- |
| `release.yml`     | Release       | P            | Tag-driven release pipeline.    |
| `tauri-build.yml` | Tauri Build   | P, PR, R, WD | Tauri desktop app build matrix. |

### Group: Ops / Misc (kebab-case)

| File                          | `name:` field              | Triggers | Purpose                                        |
| ----------------------------- | -------------------------- | -------- | ---------------------------------------------- |
| `agent-metrics-dashboard.yml` | Agent Metrics Dashboard    | WD       | Emits fleet / agent metrics.                   |
| `vendor-freshness.yml`        | Vendor Submodule Freshness | S, WD    | Checks vendored submodules for upstream drift. |

### Group: Title-Case-Dashed (legacy; rename candidates)

| File                             | `name:` field                     | Triggers    | Purpose                                                  |
| -------------------------------- | --------------------------------- | ----------- | -------------------------------------------------------- |
| `Bot-CI-Trigger.yml`             | Bot CI Trigger                    | PR, S, WD   | Re-runs failed CI on bot PRs.                            |
| `Code-Metrics.yml`               | Code Metrics                      | WD          | Emits LOC / complexity metrics.                          |
| `Comment-to-Issue-Converter.yml` | Convert Review Comments to Issues | PR, RVC, WD | Converts actionable review comments into issues.         |
| `PR-Comment-Responder.yml`       | PR Comment Collector              | IC, RVC     | Aggregates PR comments for downstream bots.              |
| `Maintenance-Global-Control.yml` | Maintenance Global Control        | WD          | Global kill-switch for maintenance workflows.            |
| `Manual-Run-All.yml`             | Manual Run All Workflows          | WD          | Operator-triggered fanout for recovery scenarios.        |
| `Nightly-Doc-Organizer.yml`      | Nightly Documentation Organizer   | WD          | Tidies docs tree on a nightly (currently WD-only) basis. |

### Group: Jules fleet — dispatched via `Jules-Control-Tower.yml` (32 files)

These are the workforce workflows that the Control Tower orchestrates. Renaming
must happen in lockstep with Control Tower's regex / filename dispatch.

| File                                 | Triggers         |
| ------------------------------------ | ---------------- |
| `Jules-Control-Tower.yml`            | P, PR, WR, S, WD |
| `Jules-Archivist.yml`                | WC               |
| `Jules-Assessment-AutoFix.yml`       | WD               |
| `Jules-Assessment-Generator.yml`     | WC, WD           |
| `Jules-Assessment-Remediator.yml`    | WD               |
| `Jules-Auto-Assign-Issues.yml`       | I                |
| `Jules-Auto-Repair.yml`              | WC, WD           |
| `Jules-Code-Quality-Fixer.yml`       | WC, WD           |
| `Jules-Code-Quality-Reviewer.yml`    | WC, WD           |
| `Jules-Comment-Processor.yml`        | WC, WD, S        |
| `Jules-Completist.yml`               | WC, WD           |
| `Jules-Comprehensive-Assessment.yml` | WC, WD           |
| `Jules-Conflict-Fix.yml`             | WC               |
| `Jules-Consolidator.yml`             | WD               |
| `Jules-Critics-Comments.yml`         | WC, WD           |
| `Jules-Documentation-Auditor.yml`    | WC, WD           |
| `Jules-Documentation-Scribe.yml`     | WC               |
| `Jules-Hotfix-Creator.yml`           | WC               |
| `Jules-Issue-Mention-Handler.yml`    | IC               |
| `Jules-Issue-Resolver.yml`           | WC, WD           |
| `Jules-Laymans-Terms-Writer.yml`     | WC, WD           |
| `Jules-PR-AutoFix.yml`               | WR, WD           |
| `Jules-PR-Cleanup.yml`               | S, WD            |
| `Jules-PR-Compiler.yml`              | WC, WD, S        |
| `Jules-Physics-Auditor.yml`          | WC, WD           |
| `Jules-Review-Fix.yml`               | RV               |
| `Jules-Sentinel.yml`                 | WC, WD           |
| `Jules-Supersede-Check.yml`          | P, WD            |
| `Jules-Tech-Custodian.yml`           | WC               |
| `Jules-Tech-Debt-Assessor.yml`       | WC, WD           |
| `Jules-Test-Generator.yml`           | WC               |

### Archived (quarantined — not active)

Under `.github/workflows/archived/`:
`Jules-Auto-Rebase.yml`, `Jules-Auto-Refactor.yml`, `Jules-Cleaner.yml`,
`Jules-Competitor-Analyst.yml`, `Jules-Curie.yml`, `Jules-DRY-Orthogonality.yml`,
`Jules-Hypatia.yml`, `Jules-Ideas-Generator.yml`, `Jules-Patent-Reviewer.yml`,
`Jules-Render-Healer.yml`, plus `assessment-auto-fix.yml.disabled` and
`auto-remediate-issues.yml.disabled`.

GitHub Actions only discovers workflows directly under `.github/workflows/`,
not in subdirectories, so these are inert. Retained for historical reference.

---

## Orphan / overlap risks called out in #2717

1. **Naming convention drift** — three conventions coexist. `Jules-*` files
   are Title-Case-Dashed and the rest is split between Title-Case-Dashed and
   kebab-case. A rename-only PR must coordinate with Control Tower.
2. **Trigger-less workflows** — none found at audit time; all 56 active files
   have at least one trigger. (The archived set is intentionally inert.)
3. **Likely overlap candidates** — `Maintenance-Global-Control.yml` and
   `Manual-Run-All.yml` both fan out to other workflows via `WD`; consider
   merging when the rename sprint happens.
4. **Dispatch-only worker fleet** — most `Jules-*` files are `workflow_call` /
   `workflow_dispatch` only; they only run when Control Tower or an operator
   dispatches them. That is by design, not an orphan.

---

## Follow-up rename plan (not in this PR)

1. Inventory all external references (branch protection, Control Tower regex,
   cross-repo callers, GAAI daemon configs).
2. Rename in a single PR with `git mv` plus updates to all referrers in the
   same commit. Keep the old filename as a one-line `workflow_dispatch`-only
   stub for one release cycle to cover external callers.
3. Collapse the four conventions to `kebab-case` with prefixes:
   `ci-*`, `docs-*`, `jules-*`, `pr-*`, `ops-*`, `release-*`.
4. Drop the stub files after one release.

This audit is the prerequisite artifact for step 1.
