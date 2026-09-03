# Workflow Tracking Document: Golf Modeling Suite

This document lists all active GitHub Workflows in this repository hub.

| Workflow Name            | Filename                         | Status   | Purpose                                                                                                                                                                                                                                                                            |
| :----------------------- | :------------------------------- | :------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Control Tower**        | `Jules-Control-Tower.yml`        | Active   | Orchestrates agentic workers.                                                                                                                                                                                                                                                      |
| **PR Compiler**          | `Jules-PR-Compiler.yml`          | Active   | Compiles PR info for fleet management.                                                                                                                                                                                                                                             |
| **CI Standard**          | `ci-standard.yml`                | Active   | Core lint/test lane; does not claim full optional-engine coverage.                                                                                                                                                                                                                 |
| **Release**              | `release.yml`                    | Active   | Tag-driven wheel/sdist build, PyPI publish, GitHub release; `build` compiles `ui/dist` with Node 24 before `python -m build` because `build_hooks.py` refuses to package without it (UD #9449); wheel smoke asserts the UI bundle and version on Python 3.11–3.12 only (RM #1507). |
| **Vendor Freshness**     | `vendor-freshness.yml`           | Active   | Submodule staleness + Cargo/pyproject Tools-pin consistency (`check_tools_pins.py`, UD #9406).                                                                                                                                                                                     |
| **CI Fast Tests**        | `ci-fast-tests.yml`              | Active   | Runs unit and integration tests (non-slow).                                                                                                                                                                                                                                        |
| **Nightly Cross-Engine** | `nightly-cross-engine.yml`       | Active   | Dedicated native-engine validation lane with strict import checks.                                                                                                                                                                                                                 |
| **Critical Files Guard** | `critical-files-guard.yml`       | Active   | Prevents accidental deletion of core files.                                                                                                                                                                                                                                        |
| **Assessment Generator** | `Jules-Assessment-Generator.yml` | Active   | Automated architecture & quality audits.                                                                                                                                                                                                                                           |
| **Auto-Repair**          | `Jules-Auto-Repair.yml`          | Disabled | Automatically fixes CI failures (Disabled via `if: false`).                                                                                                                                                                                                                        |
| **Test Generator**       | `Jules-Test-Generator.yml`       | Active   | Generates unit tests for new Python changes.                                                                                                                                                                                                                                       |
| **Doc Scribe**           | `Jules-Documentation-Scribe.yml` | Active   | Maintains CodeWiki and documentation updates.                                                                                                                                                                                                                                      |
| **Scientific Auditor**   | `Jules-Scientific-Auditor.yml`   | Active   | Peer reviews physics and math correctness.                                                                                                                                                                                                                                         |
| **Conflict Fix**         | `Jules-Conflict-Fix.yml`         | Active   | Resolves merge conflicts agentically.                                                                                                                                                                                                                                              |
| **Tech Debt Assessor**   | `Jules-Tech-Debt-Assessor.yml`   | Active   | Tracks and reports technical debt weekly.                                                                                                                                                                                                                                          |

---

## Maintenance

Update this document whenever a new workflow is added or the status of an existing workflow changes. For global standards, see `Repository_Management/docs/architecture/WORKFLOW_GOVERNANCE.md`.

## Notes

- `ci-standard.yml` is the default core PR lane. It is intentionally fast and
  honest about optional-engine coverage.
- `ci-standard.yml` concurrency: the group is per-ref on branches/PRs (a newer
  push cancels the superseded run) but per-commit on `main`, so consecutive
  merges queue instead of cancelling each other and `main` always finishes a
  run (RM #1507, #9409). `cancel-in-progress: true` stays literal because
  `lint-workflow-files.yml` greps for it.
- `nightly-cross-engine.yml` is the repo's dedicated cross-engine lane and is
  the right place to expand stricter native-engine validation over time.
- `ci-standard.yml` job `repo-structure-gates` runs
  `python3 -m scripts.registry.generate_registry_artifacts --check` so
  `src/config/launcher_manifest.json`, the `feature_parity.json` tile
  bindings and the README tile table never drift from the single tile
  registry `src/config/models.yaml` (issue #9412, RM #1507).
- `ci-standard.yml` job `tests` still refuses a pull request that deletes a
  Python test file (#7368) — a deletion has to be reviewed, not silent. Since
  #9412 the review is *recordable*: the guard hands the deleted-path list to
  `python3 scripts/ci/check_reviewed_test_deletions.py`, which requires an
  entry in `scripts/config/reviewed_test_deletions.json` naming the
  replacement test (verified to exist on disk), the tracking issue, and an
  expiry. A deletion with no entry, an expired entry, or a replacement that
  does not exist still fails the job. The record lands in the same diff as the
  deletion, so approving one approves the other (RM #1507).
