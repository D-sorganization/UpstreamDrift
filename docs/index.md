# UpstreamDrift Documentation

This index is the canonical inventory for the repository documentation tree.
Every top-level directory under `docs/` must be represented in the catalog
below with an owner and stability tag so new documentation has a clear home.

## Canonical User Documentation

User navigation starts at the [documentation hub](README.md) in this repository.
It is the canonical entry point and is kept current with the source tree.

A Sphinx project exists in `docs/sphinx/` and can be built locally, but it is not
published to a hosted documentation site. Earlier revisions of this repository
advertised `upstream-drift.readthedocs.io`; that address was never configured and
does not resolve. Do not link to it.

Repository Markdown carries development notes, governance records, and
source-adjacent references. The catalog below records where each belongs.

## Documentation Map

<!-- BEGIN GENERATED: docs-map (scripts/generate_docs_map.py) -->

Every top-level directory below is a live link. Grouping follows the
stability column of the catalog table, so archived material is visibly
separated from current guidance.

### Stable

- [`adapters/`](adapters/authoring_guide.md) - 1 page
- [`adr/`](adr/README.md) - 51 pages
- [`api/`](api/README.md) - 7 pages
- [`architecture/`](architecture/) - 19 pages
- [`audits/`](audits/) - 7 pages
- [`code-quality/`](code-quality/function-design-review.md) - 1 page
- [`codemap/`](codemap/README.md) - 3 pages
- [`conformance/`](conformance/) - no Markdown pages
- [`conventions/`](conventions/) - 3 pages
- [`deployment/`](deployment/interim_setup.md) - 1 page
- [`development/`](development/README.md) - 118 pages
- [`engineering/`](engineering/) - 2 pages
- [`engines/`](engines/README.md) - 15 pages
- [`estimation/`](estimation/synthetic-ground-truth-rig.md) - 1 page
- [`examples/`](examples/) - no Markdown pages
- [`governance/`](governance/) - 6 pages
- [`help/`](help/) - 5 pages
- [`installation/`](installation/) - 2 pages
- [`legal/`](legal/licenses.md) - 1 page
- [`motion_matching/`](motion_matching/README.md) - 2 pages
- [`motion_pipeline/`](motion_pipeline/README.md) - 6 pages
- [`operations/`](operations/) - 17 pages
- [`physics/`](physics/) - 3 pages
- [`portfolio/`](portfolio/golf_modeling_demo.md) - 1 page
- [`references/`](references/README.md) - 1 page
- [`reviews/`](reviews/) - 3 pages
- [`sidekick/`](sidekick/README.md) - 3 pages
- [`simulation_backends/`](simulation_backends/README.md) - 4 pages
- [`specs/`](specs/README.md) - 4 pages
- [`sphinx/`](sphinx/_static/VENDORED.md) - 1 page
- [`technical/`](technical/README.md) - 7 pages
- [`testing/`](testing/) - 8 pages
- [`troubleshooting/`](troubleshooting/README.md) - 6 pages
- [`tutorials/`](tutorials/README.md) - 6 pages
- [`user_guide/`](user_guide/README.md) - 30 pages
- [`validation/`](validation/cross_engine_v1.md) - 1 page
- [`workflows/`](workflows/) - 2 pages

### Draft

- [`ai_implementation/`](ai_implementation/README.md) - 5 pages
- [`bunkershot3d/`](bunkershot3d/) - 8 pages
- [`competitive_analysis/`](competitive_analysis/COMPETITOR_ANALYSIS.md) - 1 page
- [`config/`](config/pydantic-settings-migration.md) - 1 page
- [`design/`](design/) - 2 pages
- [`golf-model/`](golf-model/INPUT_POSE_INVESTIGATION.md) - 1 page
- [`model_explorer/`](model_explorer/attachment-manifests.md) - 1 page
- [`motion_capture/`](motion_capture/) - 6 pages
- [`motion_training/`](motion_training/README.md) - 1 page
- [`plans/`](plans/README.md) - 17 pages
- [`proposals/`](proposals/ROBOTICS_EXPANSION_PROPOSAL.md) - 1 page
- [`research/`](research/) - 32 pages
- [`sg_optimizer/`](sg_optimizer/README.md) - 3 pages
- [`status/`](status/) - no Markdown pages
- [`technical_debt/`](technical_debt/TODO_FIXME_REGISTER.md) - 1 page
- [`ui/`](ui/FEATURE_PARITY_MATRIX.md) - 1 page
- [`ux/`](ux/field_metadata.md) - 1 page

### Archived

- [`assessments/`](assessments/README.md) - 294 pages
- [`audit_reports/`](audit_reports/induced_acceleration_audit.md) - 1 page
- [`historical/`](historical/README.md) - 4 pages
- [`issues/`](issues/README.md) - 56 pages
- [`review_archive/`](review_archive/) - 24 pages
- [`status_quo_analysis/`](status_quo_analysis/running_log.md) - 1 page

<!-- END GENERATED: docs-map -->

## Calculation and Derivation References

The calculation sheets below carry the derivations behind the simulation and
analysis code. They are catalogued here explicitly because a 2026-08-21 review
(issue #8850) found several of them had zero inbound references anywhere in the
repository, which made them effectively unreachable.

### Physics and Ball Flight

- [Ball flight model documentation](physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md)
  -- drag, lift, and spin-decay formulation used by the flight solvers.
- [Golf ball flight and impact source map](physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md)
  -- maps each published coefficient and equation to its literature source.
- [Putting kinematics and kinetics review](physics/PUTTING_KINEMATICS_KINETICS_REVIEW.md)
  -- the putting-stroke kinematics/kinetics derivation behind
  `src/tools/putting_green_gui/`.
- [Reference papers](references/README.md) -- the primary literature the
  formulations above cite, with local PDFs under `references/papers/`.

### Estimation and Validation

- [Synthetic ground-truth rig](estimation/synthetic-ground-truth-rig.md) --
  how synthetic fixtures are generated and what identifiability they support.
- [Perturbation analysis parity guidelines](technical/perturbation_analysis_parity_guidelines.md)
  -- tolerance rules for cross-engine perturbation comparisons.
- [Control strategies summary](technical/control-strategies-summary.md) --
  the control formulations compared across engines.

### Research Derivations

- [Proximal-distal energy transfer reviewer workbench](research/proximal_distal_energy_transfer/REVIEWER_WORKBENCH.md)
  -- entry point to the interaction-force, two-hand, and distributed-shaft
  derivations, their figures, and their claim-adjudication evidence.

### Engineering Rules the Calculations Depend On

- [Dependency direction rules](engineering/dependency-direction-rules.md) --
  which layers may import which, so calculation modules stay reusable.
- [Logging policy](engineering/logging-policy.md) -- required logging shape for
  numerical code paths.

## Directory Catalog

| Directory               | Owner                 | Stability | Description                                                                                                                         |
| ----------------------- | --------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `adapters/`             | @physics-team         | stable    | Adapter authoring guidance for engine contracts, canonical state remaps, capabilities, and conformance gates.                       |
| `adr/`                  | @architecture-team    | stable    | Architecture decision records and templates for durable design decisions.                                                           |
| `ai_implementation/`    | @automation-team      | draft     | AI-assisted implementation notes and operational agent guidance.                                                                    |
| `api/`                  | @api-team             | stable    | REST API architecture, endpoint references, and integration guidance.                                                               |
| `architecture/`         | @architecture-team    | stable    | System architecture diagrams, dependency boundaries, and design overviews.                                                          |
| `assessments/`          | @quality-team         | archived  | Generated repository health assessments retained for historical comparison.                                                         |
| `audit_reports/`        | @quality-team         | archived  | Audit outputs and review evidence from repository-wide inspections.                                                                 |
| `audits/`               | @quality-team         | stable    | Dated audit records, including the current adversarial and hardening reviews.                                                       |
| `bunkershot3d/`         | @physics-team         | draft     | Granular bunker-shot backend comparison notes (Project Chrono, LIGGGHTS, MuJoCo MPM).                                               |
| `code-quality/`         | @quality-team         | stable    | Coding standards, quality gates, and maintainability guidance.                                                                      |
| `codemap/`              | @docs-team            | stable    | Code-map indexer (chat + MCP) integration notes, agent setup, and MCP wiring guidance.                                              |
| `competitive_analysis/` | @product-team         | draft     | Market and ecosystem comparisons used for planning context.                                                                         |
| `config/`               | @platform-team        | draft     | Configuration and settings documentation (e.g. the pydantic-settings migration guide).                                              |
| `conformance/`          | @physics-team         | stable    | Cross-engine conformance notes, divergence ledgers, and canonical-core parity records.                                              |
| `conventions/`          | @architecture-team    | stable    | Cross-subsystem data contracts and naming conventions, including canonical pose/state interchange.                                  |
| `deployment/`           | @platform-team        | stable    | Deployment procedures, packaging notes, and release environment guidance.                                                           |
| `design/`               | @architecture-team    | draft     | Feature design sketches and deeper design rationale before ADR promotion.                                                           |
| `development/`          | @engineering-team     | stable    | Developer workflow notes, implementation reports, and local contribution guidance.                                                  |
| `engineering/`          | @engineering-team     | stable    | Engineering practices and cross-cutting technical standards.                                                                        |
| `engines/`              | @physics-team         | stable    | Physics engine support tiers, capabilities, and backend-specific documentation.                                                     |
| `estimation/`           | @engineering-team     | stable    | Estimation validation, synthetic fixtures, identifiability probes, and offline estimator readiness notes.                           |
| `examples/`             | @engineering-team     | stable    | Runnable example scripts (mock-engine sim, kinematics estimation, synthetic motion matching) and their index.                       |
| `golf-model/`           | @physics-team         | draft     | Golf-model investigation notes and motion-matching diagnostics.                                                                     |
| `governance/`           | @maintainers          | stable    | Repository governance policies, documentation rules, and maintenance process.                                                       |
| `help/`                 | @support-team         | stable    | User support material and task-oriented help pages.                                                                                 |
| `historical/`           | @maintainers          | archived  | Historical records preserved for context but not current guidance.                                                                  |
| `installation/`         | @developer-experience | stable    | Installation instructions and environment setup guidance.                                                                           |
| `issues/`               | @maintainers          | archived  | Issue-derived notes and local tracking artifacts retained under docs.                                                               |
| `legal/`                | @maintainers          | stable    | License, compliance, and legal reference material.                                                                                  |
| `model_explorer/`       | @ui-team              | draft     | Model Explorer attachment manifest documentation and related UI workflow notes.                                                     |
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
| `research/`             | @research-team        | draft     | Long-form research articles (Quarto/LaTeX/PDF) produced from repository analyses, with verified bibliographies.                     |
| `review_archive/`       | @quality-team         | archived  | Older review records retained in place; see the consolidation decisions below.                                                      |
| `reviews/`              | @quality-team         | stable    | Current review records, remediation notes, and quality findings.                                                                    |
| `sg_optimizer/`         | @physics-team         | draft     | Strokes Gained Optimizer spec, data sources, and documentation.                                                                     |
| `shared_tools/`         | @platform-team        | stable    | UpstreamDrift <-> Tools seam: divergence inventory, per-package rulings, drift-gate docs (UD #9406).                                |
| `sidekick/`             | @platform-team        | stable    | Sidekick shared-utilities docs, launcher sidebar, chat/provider integration, tools library, and integration guides.                 |
| `simulation_backends/`  | @physics-team         | stable    | Backend-agnostic golf-model simulation layer (ODE / MuJoCo CPU / MuJoCo Warp GPU): user guide, launcher tile, and cross-validation. |
| `specs/`                | @architecture-team    | stable    | Specifications that expand or support the root `SPEC.md` contract.                                                                  |
| `sphinx/`               | @docs-team            | stable    | Sphinx source and generated artifacts for the rendered documentation site.                                                          |
| `status/`               | @maintainers          | draft     | Repository status snapshots and rolling state-of-the-fleet notes.                                                                   |
| `status_quo_analysis/`  | @product-team         | archived  | Status quo analysis snapshots preserved for planning history.                                                                       |
| `technical/`            | @engineering-team     | stable    | Technical reference pages for implementation details and subsystem behavior.                                                        |
| `technical_debt/`       | @quality-team         | draft     | Technical debt inventories, cleanup plans, and remediation tracking.                                                                |
| `testing/`              | @quality-team         | stable    | Testing strategy, validation guidance, and quality assurance references.                                                            |
| `troubleshooting/`      | @support-team         | stable    | Troubleshooting guides for installation, runtime, and development issues.                                                           |
| `tutorials/`            | @developer-experience | stable    | Step-by-step learning paths and task walkthroughs for users.                                                                        |
| `ui/`                   | @ui-team              | draft     | Launcher/UI feature parity matrix and frontend-facing notes.                                                                        |
| `user_guide/`           | @docs-team            | stable    | User-facing guides for common workflows and product capabilities.                                                                   |
| `ux/`                   | @ui-team              | draft     | UX infrastructure for epic #5968: field metadata registry, copy style, walkthrough specs, and contributor guidance.                 |
| `validation/`           | @quality-team         | stable    | Cross-engine validation artifacts, differential reports, and machine-readable evidence snapshots.                                   |
| `workflows/`            | @platform-team        | stable    | Automation workflow documentation and CI/CD process references.                                                                     |

## Consolidation Decisions

Issue #8840 recorded four consolidations that this catalog described as pending
rather than performing. Each is now decided. Dated entries, newest first; a
decision recorded here is final unless a later dated entry supersedes it.

### 2026-09-03 -- `strategic/` Into `plans/`: DONE

`docs/strategic/` held two draft planning documents that belonged with the rest
of the roadmap material. Both moved to `plans/`
([feature expansion roadmap](plans/FEATURE_EXPANSION_ROADMAP.md),
[ideas](plans/ideas.md)) and the `strategic/` catalog row was removed. No
redirect stub was left: the only inbound references were this catalog and two
dated historical notes, which are preserved as written.

### 2026-09-03 -- `audits/` Alongside `audit_reports/`: NO MERGE, Catalog Corrected

The premise was wrong. `audits/` was tagged `archived` and described as "legacy
notes", but it holds the _newest_ audit records in the repository, including
`2026-08-21-adversarial-integration-review.md` (the review that produced this
very issue) and `consolidation_hardening_audit_2026-09.md`. `audit_reports/`
holds a single generated report. The directories are therefore not a duplicate
pair to be merged; the defect was the catalog metadata. `audits/` is now tagged
`stable` and described as dated audit records. No files move.

### 2026-09-03 -- `review_archive/` Into `reviews/archive/`: DEFERRED

Deliberately not performed here. `docs/review_archive/` is referenced by path
from `.github/workflows/Nightly-Doc-Organizer.yml` and
`.github/workflows/Comment-to-Issue-Converter.yml`, so the move would require
coupled workflow edits. Workflow changes in this repository go through the
governed workflow campaign, not an ad-hoc docs pull request. Until that
campaign runs, `review_archive/` stays where it is and keeps its `archived`
stability tag, which already gives readers the navigational precedence the
issue asked for. The "retained until consolidated" wording has been removed so
the catalog no longer advertises pending work.

### 2026-09-03 -- 72 Dated `assessments/` Entries: NO RESTRUCTURE

`assessments/` carries 294 Markdown pages in dated batches and is already
tagged `archived`, which is the signal a reader needs to tell superseded
material from current guidance. Re-nesting it by date would rewrite several
hundred paths, invalidate inbound references from dated audit and review
records, and buy no navigational precedence beyond what the `archived` tag and
the grouped documentation map above already provide. Declined as
disproportionate; revisit only if `assessments/` starts receiving new material,
in which case the correct fix is a fresh directory rather than a reorganisation
of the archive.

## Governance Checks

`scripts/check_doc_catalog.py` verifies that this catalog covers every
top-level `docs/` directory and that `README.md` points readers to the rendered
documentation URL from `pyproject.toml`.

`scripts/generate_docs_map.py` regenerates the documentation map above and the
structure block in `README.md` from the real `docs/` tree; run it with
`--check` to confirm neither block has drifted.

`scripts/check_doc_size_budget.py` enforces the 50 KB Markdown/Quarto budget.
Temporary exceptions must live in `scripts/config/doc_size_budget.json` with an
owner and expiration date.
