# GitHub Workflows Documentation

This document describes the automated workflows configured for the Golf Modeling Suite.

## Global Control (Pause/Resume)
To prevent automated runs during major refactoring or maintenance, we use a file-based pause mechanism.
- **To Pause**: Create a file at `.github/WORKFLOWS_PAUSED`.
- **To Resume**: Delete the file at `.github/WORKFLOWS_PAUSED`.
- **Manual Automation**: Use the **Maintenance Global Control** workflow in the Actions tab to toggle this state.

## Core Pipelines
- **CI Standard**: (ci-standard.yml) Ran on every push to main and PR. Handles linting, basic tests, and build checks.
- **CI Fast Tests**: (ci-fast-tests.yml) Optimized parallel test suite with a 10-minute hard cutoff. Failures here block merging.
- **Nightly Documentation Organizer**: (Nightly-Doc-Organizer.yml) Runs daily at 5 AM UTC. Organizes archives and updates this status document.

## Jules AI Agent Suite
Jules is our suite of AI agents that maintain the repository. They are primarily dispatched by the **Jules Control Tower**.

| Workflow | Purpose | Schedule |
|----------|---------|----------|
| **Jules Control Tower** | Main dispatcher for periodic tasks. | Hourly / Event-driven |
| **Jules PR Compiler** | Consolidates multiple small Jules PRs. | Daily 3:30 AM UTC |
| **Jules Assessment Generator** | Audits codebase for technical debt/bugs. | Daily Midnight PST |
| **Jules Tech Debt Assessor** | High-level architectural review. | Weekly |
| **Jules Issue Resolver** | Attempts to fix detected issues automatically. | Daily |

## Maintenance & Cleanup
- **Comment-to-Issue-Converter**: Automatically creates issues from PR review comments.
- **Stale Cleanup**: Manages stale issues and PRs.
- **Jules Cleaner**: Cleans up temporary artifacts.

---
*Last Updated: 2026-01-21*
*Automated part of the Golf Modeling Suite Maintenance Plan.*
