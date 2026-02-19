# Assessment Framework v2.0

## Canonical Current Quality Status

- Canonical status file: `docs/assessments/CANONICAL_QUALITY_STATUS.md`
- Archive index: `docs/assessments/archive/INDEX.md`

When assessments are superseded, move old artifacts into `docs/assessments/archive/` and update the archive index.

---

## Canonical Assessment Taxonomy

- Active framework index: `docs/assessments/README.md` (this file)
- Archived/historical assessments: `docs/archive/` and `docs/assessments/archive/`
- Generated debt inventory: `docs/technical_debt/TODO_FIXME_REGISTER.md`

When adding or moving assessment artifacts, update this README in the same PR.

---

## Overview

This document defines the comprehensive 16-point assessment framework (A-O + Highlight) for evaluating repository health, user experience, and production readiness.

**Version**: 2.0 (January 2026)
**Based On**: Fleet-wide adversarial review analysis identifying strategic gaps

---

## Assessment Categories (A-O)

### Core Technical (A-C)

| ID    | Name                          | Focus                                  | Weight |
| ----- | ----------------------------- | -------------------------------------- | ------ |
| **A** | Architecture & Implementation | Code structure, patterns, completeness | 2x     |
| **B** | Code Quality & Hygiene        | Linting, formatting, type safety       | 1.5x   |
| **C** | Documentation & Comments      | Code docs, API docs, inline comments   | 1x     |

### User-Facing (D-F)

| ID    | Name                                | Focus                                       | Weight |
| ----- | ----------------------------------- | ------------------------------------------- | ------ |
| **D** | User Experience & Developer Journey | Time-to-value, onboarding, friction points  | 2x     |
| **E** | Performance & Scalability           | Computational efficiency, memory, profiling | 1.5x   |
| **F** | Installation & Deployment           | pip/conda, cross-platform, CI/CD            | 1.5x   |

### Reliability & Safety (G-I)

| ID    | Name                        | Focus                                           | Weight |
| ----- | --------------------------- | ----------------------------------------------- | ------ |
| **G** | Testing & Validation        | Unit tests, integration tests, coverage         | 2x     |
| **H** | Error Handling & Debugging  | Error messages, stack traces, recovery          | 1.5x   |
| **I** | Security & Input Validation | Injection, sanitization, vulnerability scanning | 1.5x   |

### Sustainability (J-L)

| ID    | Name                                | Focus                                        | Weight |
| ----- | ----------------------------------- | -------------------------------------------- | ------ |
| **J** | Extensibility & Plugin Architecture | Adding new features, API stability           | 1x     |
| **K** | Reproducibility & Provenance        | Determinism, versioning, experiment tracking | 1.5x   |
| **L** | Long-Term Maintainability           | Tech debt, dependency aging, bus factor      | 1x     |

### Communication (M-O)

| ID    | Name                              | Focus                                          | Weight |
| ----- | --------------------------------- | ---------------------------------------------- | ------ |
| **M** | Educational Resources & Tutorials | Learning curve, examples, video guides         | 1x     |
| **N** | Visualization & Export            | Plot quality, accessibility, publication-ready | 1x     |
| **O** | CI/CD & DevOps                    | Pipeline health, automation, release process   | 1x     |

### Summary

| ID            | Name              | Focus                                         | Weight |
| ------------- | ----------------- | --------------------------------------------- | ------ |
| **Highlight** | Executive Summary | Top risks, overall score, remediation roadmap | N/A    |

---

## Specialized Audits

In addition to the standard framework, specialized audits monitor specific quality aspects.

### Completist Audits

**Focus**: Implementation completeness, identifying critical gaps, feature requests (TODOs), and technical debt (FIXMEs).

| Report                                                        | Date       | Focus                                                             |
| ------------------------------------------------------------- | ---------- | ----------------------------------------------------------------- |
| `docs/assessments/completist/Completist_Report_2026-02-19.md` | 2026-02-19 | Critical Controller Connectivity, Feature Gaps, Technical Debt    |
| `docs/assessments/completist/Completist_Report_2026-02-15.md` | 2026-02-15 | Critical backend gaps (PyVista/Unreal) & False Positive Filtering |

---

## Assessment Prompt Index

| Assessment    | Prompt File                      | Results Template                         |
| ------------- | -------------------------------- | ---------------------------------------- |
| A             | `Assessment_Prompt_A.md`         | `Assessment_A_Results_YYYY-MM-DD.md`     |
| B             | `Assessment_Prompt_B.md`         | `Assessment_B_Results_YYYY-MM-DD.md`     |
| C             | `Assessment_Prompt_C.md`         | `Assessment_C_Results_YYYY-MM-DD.md`     |
| D             | `Assessment_Prompt_D.md`         | `Assessment_D_Results_YYYY-MM-DD.md`     |
| E             | `Assessment_Prompt_E.md`         | `Assessment_E_Results_YYYY-MM-DD.md`     |
| F             | `Assessment_Prompt_F.md`         | `Assessment_F_Results_YYYY-MM-DD.md`     |
| G             | `Assessment_Prompt_G.md`         | `Assessment_G_Results_YYYY-MM-DD.md`     |
| H             | `Assessment_Prompt_H.md`         | `Assessment_H_Results_YYYY-MM-DD.md`     |
| I             | `Assessment_Prompt_I.md`         | `Assessment_I_Results_YYYY-MM-DD.md`     |
| J             | `Assessment_Prompt_J.md`         | `Assessment_J_Results_YYYY-MM-DD.md`     |
| K             | `Assessment_Prompt_K.md`         | `Assessment_K_Results_YYYY-MM-DD.md`     |
| L             | `Assessment_Prompt_L.md`         | `Assessment_L_Results_YYYY-MM-DD.md`     |
| M             | `Assessment_Prompt_M.md`         | `Assessment_M_Results_YYYY-MM-DD.md`     |
| N             | `Assessment_Prompt_N.md`         | `Assessment_N_Results_YYYY-MM-DD.md`     |
| O             | `Assessment_Prompt_O.md`         | `Assessment_O_Results_YYYY-MM-DD.md`     |
| Highlight     | `Assessment_Prompt_Highlight.md` | `Assessment_Highlight_YYYY-MM-DD.md`     |
| Comprehensive | N/A (multi-framework)            | `comprehensive_assessment_YYYY-MM-DD.md` |

### Specialized Assessments

| Date       | Topic                                    | Report File                                            |
| ---------- | ---------------------------------------- | ------------------------------------------------------ |
| 2025-02-24 | Physics Audit                            | `docs/assessments/physics/Physics_Audit_2025-02-24.md` |
| 2026-02-17 | Implementation Gaps                      | `docs/assessments/implementation_gaps_report.md`       |
| 2026-02-17 | Cross-Repo A-O + Pragmatic + DbC/DRY/TDD | `docs/assessments/Cross_Repo_Assessment_2026-02-17.md` |

---

## Severity Definitions (All Assessments)

| Severity     | Definition                                           | Response Time |
| ------------ | ---------------------------------------------------- | ------------- |
| **BLOCKER**  | Prevents core functionality, must fix before release | 24 hours      |
| **CRITICAL** | High user impact, significant risk                   | 48 hours      |
| **MAJOR**    | Deviation from standards, quality issue              | 2 weeks       |
| **MINOR**    | Low risk improvement                                 | 6 weeks       |
| **NIT**      | Style/preference only if systemic                    | Backlog       |

---

## Assessment Execution Schedule

### Rolling Assessment Cycle

| Week | Assessments | Focus                    |
| ---- | ----------- | ------------------------ |
| 1    | A, D, G     | Core implementation & UX |
| 2    | B, E, H     | Quality & error handling |
| 3    | C, F, I     | Documentation & security |
| 4    | J, K, L     | Sustainability           |
| 5    | M, N, O     | Communication & CI/CD    |
| 6    | Highlight   | Executive summary        |

### Trigger-Based Assessments

| Trigger           | Assessments              |
| ----------------- | ------------------------ |
| Pre-release       | A, D, F, G, I, Highlight |
| Post-incident     | G, H, I                  |
| Dependency update | F, I, L                  |
| New feature       | A, D, G, M               |

---

## Success Criteria (All Repositories)

### Minimum Viable Quality

- [ ] All BLOCKER issues resolved
- [ ] All CRITICAL issues have mitigation plan
- [ ] > 80% test coverage on core modules
- [ ] <15 minute installation (P90)
- [ ] <30 minute first plot/result (P90)
- [ ] > 80% of errors have actionable messages

### Target Quality (FLAGSHIP Status)

- [ ] All MAJOR issues resolved
- [ ] > 90% overall assessment score
- [ ] Full documentation coverage
- [ ] Video tutorials available
- [ ] Extension API documented
- [ ] 3+ community contributions

---

## Version History

| Version | Date    | Changes                                        |
| ------- | ------- | ---------------------------------------------- |
| 1.0     | 2025-12 | Initial 15-point framework                     |
| 2.0     | 2026-01 | Reorganized based on UX/strategic gap analysis |
| 2.1     | 2026-02 | Added comprehensive multi-framework assessment |
| 2.2     | 2026-02 | Added Physics Audit (Specialized Assessment)   |
| 2.3     | 2026-02 | Added Injury Risk Scoring issue (ISSUE-016)    |
| 2.4     | 2026-02 | Added Implementation Gaps Report               |

---

_See individual Assessment_Prompt_X.md files for detailed prompts._
