# UpstreamDrift Documentation

This index is the canonical inventory for the repository documentation tree.
Every top-level directory under `docs/` must be represented in the catalog
below with an owner and stability tag so new documentation has a clear home.

## Canonical User Documentation

The rendered documentation surface is
[upstream-drift.readthedocs.io](https://upstream-drift.readthedocs.io), backed
by the Sphinx project in `docs/sphinx/`. Repository Markdown remains useful for
development notes, governance records, and source-adjacent references, but user
navigation should start with the rendered documentation URL.

## Directory Catalog

| Directory               | Owner                 | Stability | Description                                                                                                                         |
| ----------------------- | --------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `adr/`                  | @architecture-team    | stable    | Architecture decision records and templates for durable design decisions.                                                           |
| `ai_implementation/`    | @automation-team      | draft     | AI-assisted implementation notes and operational agent guidance.                                                                    |
| `api/`                  | @api-team             | stable    | REST API architecture, endpoint references, and integration guidance.                                                               |
| `architecture/`         | @architecture-team    | stable    | System architecture diagrams, dependency boundaries, and design overviews.                                                          |
| `assessments/`          | @quality-team         | archived  | Generated repository health assessments retained for historical comparison.                                                         |
| `audit_reports/`        | @quality-team         | archived  | Audit outputs and review evidence from repository-wide inspections.                                                                 |
| `audits/`               | @quality-team         | archived  | Legacy audit notes preserved alongside `audit_reports/` for historical reference.                                                   |
| `bunkershot3d/`         | @physics-team         | draft     | Granular bunker-shot backend comparison notes (Project Chrono, LIGGGHTS, MuJoCo MPM).                                               |
| `code-quality/`         | @quality-team         | stable    | Coding standards, quality gates, and maintainability guidance.                                                                      |
| `codemap/`              | @docs-team            | stable    | Code-map indexer (chat + MCP) integration notes, agent setup, and MCP wiring guidance.                                              |
| `competitive_analysis/` | @product-team         | draft     | Market and ecosystem comparisons used for planning context.                                                                         |
| `config/`               | @platform-team        | draft     | Configuration and settings documentation (e.g. the pydantic-settings migration guide).                                              |
| `deployment/`           | @platform-team        | stable    | Deployment procedures, packaging notes, and release environment guidance.                                                           |
| `design/`               | @architecture-team    | draft     | Feature design sketches and deeper design rationale before ADR promotion.                                                           |
| `development/`          | @engineering-team     | stable    | Developer workflow notes, implementation reports, and local contribution guidance.                                                  |
| `engineering/`          | @engineering-team     | stable    | Engineering practices and cross-cutting technical standards.                                                                        |
| `engines/`              | @physics-team         | stable    | Physics engine support tiers, capabilities, and backend-specific documentation.                                                     |
| `examples/`             | @developer-experience | stable    | Example workflows and sample usage for common simulation tasks.                                                                     |
| `golf-model/`           | @physics-team         | draft     | Golf-model investigation notes and motion-matching diagnostics.                                                                     |
| `governance/`           | @maintainers          | stable    | Repository governance policies, documentation rules, and maintenance process.                                                       |
| `help/`                 | @support-team         | stable    | User support material and task-oriented help pages.                                                                                 |
| `historical/`           | @maintainers          | archived  | Historical records preserved for context but not current guidance.                                                                  |
| `installation/`         | @developer-experience | stable    | Installation instructions and environment setup guidance.                                                                           |
| `issues/`               | @maintainers          | archived  | Issue-derived notes and local tracking artifacts retained under docs.                                                               |
| `legal/`                | @maintainers          | stable    | License, compliance, and legal reference material.                                                                                  |
| `motion_capture/`       | @research-team        | draft     | Motion capture intake notes and source-format reference material.                                                                   |
| `motion_matching/`      | @research-team        | stable    | Motion-matching system documentation including surrogate training and cross-option leaderboards.                                    |
| `motion_pipeline/`      | @research-team        | stable    | User-facing motion pipeline workflow guide, format matrix, troubleshooting, and backend compatibility tables.                       |
| `motion_training/`      | @research-team        | draft     | Motion training research notes and prototype workflow documentation.                                                                |
| `operations/`           | @platform-team        | stable    | Operational runbooks, observability notes, and production maintenance guidance.                                                     |
| `physics/`              | @physics-team         | stable    | Physics assumptions, validation sources, and biomechanical modeling references.                                                     |
| `plans/`                | @product-team         | draft     | Roadmaps, implementation plans, and active planning documents.                                                                      |
| `portfolio/`            | @developer-experience | stable    | Reviewer-facing demonstrations and concise project showcase material.                                                               |
| `proposals/`            | @product-team         | draft     | Proposed changes and design alternatives pending acceptance or archival.                                                            |
| `references/`           | @research-team        | stable    | External references, source maps, and supporting research material.                                                                 |
| `review_archive/`       | @quality-team         | archived  | Older review records retained until consolidated into `reviews/archive/`.                                                           |
| `reviews/`              | @quality-team         | stable    | Current review records, remediation notes, and quality findings.                                                                    |
| `sg_optimizer/`         | @physics-team         | draft     | Strokes Gained Optimizer spec, data sources, and documentation.                                                                     |
| `sidekick/`             | @platform-team        | stable    | Sidekick shared-utilities docs, launcher sidebar, chat/provider integration, tools library, and integration guides.                 |
| `simulation_backends/`  | @physics-team         | stable    | Backend-agnostic golf-model simulation layer (ODE / MuJoCo CPU / MuJoCo Warp GPU): user guide, launcher tile, and cross-validation. |
| `specs/`                | @architecture-team    | stable    | Specifications that expand or support the root `SPEC.md` contract.                                                                  |
| `sphinx/`               | @docs-team            | stable    | Sphinx source and generated artifacts for the rendered documentation site.                                                          |
| `status/`               | @maintainers          | draft     | Repository status snapshots and rolling state-of-the-fleet notes.                                                                   |
| `status_quo_analysis/`  | @product-team         | archived  | Status quo analysis snapshots preserved for planning history.                                                                       |
| `strategic/`            | @product-team         | draft     | Strategic planning notes that should eventually consolidate into `plans/`.                                                          |
| `technical/`            | @engineering-team     | stable    | Technical reference pages for implementation details and subsystem behavior.                                                        |
| `technical_debt/`       | @quality-team         | draft     | Technical debt inventories, cleanup plans, and remediation tracking.                                                                |
| `testing/`              | @quality-team         | stable    | Testing strategy, validation guidance, and quality assurance references.                                                            |
| `troubleshooting/`      | @support-team         | stable    | Troubleshooting guides for installation, runtime, and development issues.                                                           |
| `tutorials/`            | @developer-experience | stable    | Step-by-step learning paths and task walkthroughs for users.                                                                        |
| `ui/`                   | @ui-team              | draft     | Launcher/UI feature parity matrix and frontend-facing notes.                                                                        |
| `user_guide/`           | @docs-team            | stable    | User-facing guides for common workflows and product capabilities.                                                                   |
| `ux/`                   | @ui-team              | draft     | UX infrastructure for epic #5968: field metadata registry, copy style, walkthrough specs, and contributor guidance.                 |
| `workflows/`            | @platform-team        | stable    | Automation workflow documentation and CI/CD process references.                                                                     |

## Governance Checks

`scripts/check_doc_catalog.py` verifies that this catalog covers every
top-level `docs/` directory and that `README.md` points readers to the rendered
documentation URL from `pyproject.toml`.

`scripts/check_doc_size_budget.py` enforces the 50 KB Markdown/Quarto budget.
Temporary exceptions must live in `scripts/config/doc_size_budget.json` with an
owner and expiration date.
