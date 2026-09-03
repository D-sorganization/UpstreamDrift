# Workflow Tracking Document: Golf Modeling Suite

This document lists all active GitHub Workflows in this repository hub.

| Workflow Name            | Filename                         | Status   | Purpose                                                            |
| :----------------------- | :------------------------------- | :------- | :----------------------------------------------------------------- |
| **Control Tower**        | `Jules-Control-Tower.yml`        | Active   | Orchestrates agentic workers.                                      |
| **PR Compiler**          | `Jules-PR-Compiler.yml`          | Active   | Compiles PR info for fleet management.                             |
| **CI Standard**          | `ci-standard.yml`                | Active   | Core lint/test lane; does not claim full optional-engine coverage. |
| **CI Fast Tests**        | `ci-fast-tests.yml`              | Active   | Runs unit and integration tests (non-slow).                        |
| **Nightly Cross-Engine** | `nightly-cross-engine.yml`       | Active   | Dedicated native-engine validation lane with strict import checks. |
| **Critical Files Guard** | `critical-files-guard.yml`       | Active   | Prevents accidental deletion of core files.                        |
| **Assessment Generator** | `Jules-Assessment-Generator.yml` | Active   | Automated architecture & quality audits.                           |
| **Auto-Repair**          | `Jules-Auto-Repair.yml`          | Disabled | Automatically fixes CI failures (Disabled via `if: false`).        |
| **Test Generator**       | `Jules-Test-Generator.yml`       | Active   | Generates unit tests for new Python changes.                       |
| **Doc Scribe**           | `Jules-Documentation-Scribe.yml` | Active   | Maintains CodeWiki and documentation updates.                      |
| **Scientific Auditor**   | `Jules-Scientific-Auditor.yml`   | Active   | Peer reviews physics and math correctness.                         |
| **Conflict Fix**         | `Jules-Conflict-Fix.yml`         | Active   | Resolves merge conflicts agentically.                              |
| **Tech Debt Assessor**   | `Jules-Tech-Debt-Assessor.yml`   | Active   | Tracks and reports technical debt weekly.                          |

---

## Maintenance

Update this document whenever a new workflow is added or the status of an existing workflow changes. For global standards, see `Repository_Management/docs/architecture/WORKFLOW_GOVERNANCE.md`.

## Notes

- `ci-standard.yml` is the default core PR lane. It is intentionally fast and
  honest about optional-engine coverage.
- `nightly-cross-engine.yml` is the repo's dedicated cross-engine lane and is
  the right place to expand stricter native-engine validation over time.
- `release.yml` job `companion-protected-main` always runs on pushes to
  `main` (no path filter) and publishes the attested artifact
  `upstreamdrift-companion-<sha>` (30-day retention) plus
  `upstreamdrift-companion-evidence-<sha>`. Payloads: `manifest.json`
  (stable consumer name, byte-identical to `upstreamdrift-companion.v1.json`),
  `capabilities.json`, `screenshots.json` (metadata-only, `pending`), the
  three matching schemas, the acquisition schema, the compatibility policy,
  and a `.sha256` sidecar per file. Tag pushes attach the same set to the
  draft release. The job sets `PYTHONPATH` to the workspace and pins
  `jsonschema`/`pyyaml` so the import-free builder runs on any runner
  (RM #1507, #9416). `ci-standard.yml` runs
  `scripts.companion_publication check` on code PRs.
