# UpstreamDrift Documentation Hub

> Updated: 2026-04-24 | Closes #3073

This index is the single starting point for all UpstreamDrift documentation.
Start here, then follow the links to the area you need.

---

## Quick navigation

| I want to…                         | Go to                            |
| ---------------------------------- | -------------------------------- |
| Install and run the project        | [installation/](installation/)   |
| Understand the system architecture | [architecture/](architecture/)   |
| Read per-engine reference docs     | [engines/](engines/)             |
| Follow a hands-on tutorial         | [tutorials/](tutorials/)         |
| Browse the REST API reference      | [api/](api/)                     |
| Read Architecture Decision Records | [adr/](adr/)                     |
| Contribute / develop               | [development/](development/)     |
| Security policies                  | [../SECURITY.md](../SECURITY.md) |
| Understand the full spec           | [../SPEC.md](../SPEC.md)         |

---

## Directory reference

### Active documentation

| Directory                                    | Contents                                                                                    |
| -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `adr/`                                       | Architecture Decision Records (ADR-0001 through ADR-0005)                                   |
| `ai_implementation/`                         | Notes on AI/ML integration patterns                                                         |
| `api/`                                       | REST API architecture and endpoint reference                                                |
| `architecture/`                              | System diagrams, project map, orthogonality review, data pipeline                           |
| `deployment/`                                | Docker, GPU, and production deployment guides                                               |
| `design/`                                    | Design guidelines and proposals                                                             |
| `development/`                               | Contributing guide, getting started, configuration reference, external provider onboarding  |
| `engines/`                                   | Per-engine reference docs (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite) and selection guide |
| `examples/`                                  | Runnable example scripts and notebooks                                                      |
| `governance/`                                | Documentation governance policies                                                           |
| `help/`                                      | User-facing help and FAQ                                                                    |
| `installation/`                              | Installation instructions for all platforms                                                 |
| `legal/`                                     | License and compliance notes                                                                |
| `motion_training/`                           | Motion capture and training data documentation                                              |
| `operations/`                                | Operational runbooks                                                                        |
| `perturbation_analysis_parity_guidelines.md` | Guidelines for perturbation-analysis parity across engines                                  |
| `physics/`                                   | Physics modeling reference                                                                  |
| `proposals/`                                 | Feature proposals under consideration                                                       |
| `references/`                                | External references and bibliography                                                        |
| `specs/`                                     | Additional spec fragments                                                                   |
| `technical/`                                 | Technical deep-dives                                                                        |
| `technical_debt/`                            | Tracked technical debt items                                                                |
| `testing/`                                   | Test strategy and coverage guidance                                                         |
| `troubleshooting/`                           | Common issues and resolution steps                                                          |
| `tutorials/`                                 | Step-by-step tutorials (humanoid, golf, pendulum, choose-your-engine)                       |
| `user_guide/`                                | Task-oriented user guide                                                                    |
| `workflows/`                                 | CI/CD and development workflow documentation                                                |
| `engine_selection_guide.md`                  | How to choose the right physics engine                                                      |
| `docker-gpu.md`                              | GPU Docker setup                                                                            |
| `index.md`                                   | Legacy index (superseded by this file)                                                      |
| `UPSTREAM_DRIFT_USER_MANUAL.md`              | Full user manual                                                                            |
| `USER_MANUAL.md`                             | Condensed user manual                                                                       |
| `BUILD_INFRASTRUCTURE_REVIEW.md`             | Build infrastructure review notes                                                           |
| `CONFIG_ISSUES_QUICK_REFERENCE.md`           | Quick reference for common config issues                                                    |
| `IDEAS.md`                                   | Feature ideas parking lot                                                                   |
| `INFRASTRUCTURE_REVIEW_INDEX.md`             | Index of infrastructure review documents                                                    |
| `project_design_guidelines.qmd`              | Quarto project design guidelines                                                            |

### [ARCHIVED] — historical / superseded content

The directories below contain documents from January–February 2026 planning
sessions that have been superseded by GitHub issues under the #3045 umbrella.
They are retained for historical reference only and should not be used as
authoritative guidance.

| Directory               | Reason archived                                                                 |
| ----------------------- | ------------------------------------------------------------------------------- |
| `plans/`                | 13 "master plan" documents (Jan 2026) — open items migrated to GitHub issues    |
| `assessments/`          | Point-in-time codebase assessments (Jan–Apr 2026) — superseded by issue tracker |
| `audit_reports/`        | Audit snapshots — superseded by ongoing CI and issue tracker                    |
| `reviews/`              | Ad-hoc review documents — findings tracked in issues                            |
| `review_archive/`       | Older review documents — archived                                               |
| `status_quo_analysis/`  | Status snapshots — superseded by current SPEC.md and README                     |
| `historical/`           | Previously archived documents                                                   |
| `competitive_analysis/` | Competitive landscape analysis (Jan 2026)                                       |
| `strategic/`            | Strategic planning documents (Jan 2026)                                         |
| `engineering/`          | Engineering process notes (Jan 2026) — superseded by `development/`             |
| `code-quality/`         | Code quality reports — superseded by CI and issue tracker                       |
| `issues/`               | Issue staging area — use GitHub Issues instead                                  |

### Sphinx API docs

`sphinx/` contains a Sphinx configuration (`conf.py`) and pre-generated HTML.
The Sphinx build is not currently wired into CI. Until `docs-ci.yml` publishes
it automatically, treat the HTML as a best-effort snapshot and refer to the
inline docstrings in `src/` as the authoritative API reference.

---

## A new contributor's path (3 clicks)

1. **[installation/](installation/)** — get the project running locally
2. **[tutorials/](tutorials/)** — follow the `choose_your_engine` tutorial
3. **[development/getting_started.md](development/getting_started.md)** — understand the contribution workflow

---

_This file is maintained by the UpstreamDrift team. If you find a broken link
or a missing directory, please open an issue referencing #3073._
