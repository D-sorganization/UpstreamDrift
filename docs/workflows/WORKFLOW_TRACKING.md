# Workflow Tracking Document: Golf Modeling Suite

This document lists all active GitHub Workflows in this repository hub.

| Workflow Name | Filename | Status | Purpose |
| :--- | :--- | :--- | :--- |
| **Control Tower** | `Jules-Control-Tower.yml` | Active | Orchestrates agentic workers. |
| **PR Compiler** | `Jules-PR-Compiler.yml` | Active | Compiles PR info for fleet management. |
| **CI Standard** | `ci-standard.yml` | Active | Linting, Formatting, and Type checking. |
| **CI Fast Tests** | `ci-fast-tests.yml` | Active | Runs unit and integration tests (non-slow). |
| **Nightly Cross-Engine** | `nightly-cross-engine.yml` | Active | Validates physics across all engines nightly. |
| **Critical Files Guard** | `critical-files-guard.yml` | Active | Prevents accidental deletion of core files. |
| **Assessment Generator** | `Jules-Assessment-Generator.yml` | Active | Automated architecture & quality audits. |
| **Auto-Repair** | `Jules-Auto-Repair.yml` | Disabled | Automatically fixes CI failures (Disabled via `if: false`). |
| **Test Generator** | `Jules-Test-Generator.yml` | Active | Generates unit tests for new Python changes. |
| **Doc Scribe** | `Jules-Documentation-Scribe.yml` | Active | Maintains CodeWiki and documentation updates. |
| **Scientific Auditor** | `Jules-Scientific-Auditor.yml` | Active | Peer reviews physics and math correctness. |
| **Conflict Fix** | `Jules-Conflict-Fix.yml` | Active | Resolves merge conflicts agentically. |
| **Tech Debt Assessor** | `Jules-Tech-Debt-Assessor.yml` | Active | Tracks and reports technical debt weekly. |

---

## Maintenance
Update this document whenever a new workflow is added or the status of an existing workflow changes. For global standards, see `Repository_Management/docs/architecture/WORKFLOW_GOVERNANCE.md`.
