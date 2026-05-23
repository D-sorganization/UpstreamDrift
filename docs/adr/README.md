# Architecture Decision Records (ADRs)

This directory tracks architecture-impacting decisions for UpstreamDrift.

## Policy

- Use `ADR_TEMPLATE.md` for every new ADR.
- Filename format: `NNNN-short-title.md`.
- Every ADR must include Status, Date, and validation notes.
- Superseded ADRs must link to the replacing ADR.

## Index

| ADR                                                | Title                                                           | Status   | Date       |
| -------------------------------------------------- | --------------------------------------------------------------- | -------- | ---------- |
| [0001](0001-fastapi-local-first-api.md)            | FastAPI for Local-First API Design                              | Accepted | 2026-02-18 |
| [0002](0002-physics-engine-plugin-architecture.md) | Physics Engine Plugin Architecture                              | Accepted | 2026-02-18 |
| [0003](0003-websocket-realtime-simulation.md)      | WebSocket Protocol for Real-Time Simulation                     | Accepted | 2026-02-18 |
| [0004](0004-launcher-provider-migration.md)        | Launcher Provider Migration Modes and Legacy Deprecation Policy | Accepted | 2026-04-08 |
| [0005](0005-mission-drift-calculators.md)          | Mission-Drift Calculators                                       | Accepted | 2026-05-08 |
| [0006](0006-canonical-urdf-subsystem.md)           | Canonical URDF Subsystem                                        | Proposed | 2026-05-08 |
| [0007](0007-canonical-urdf-subsystem.md)           | Canonical URDF Subsystem (duplicate — see ADR-0006)             | Proposed | 2026-05-08 |
| [0008](0008-body-part-viz-toolkit.md)              | Body-Part Visualisation Toolkit                                 | Accepted | 2026-05-08 |
| [0009](0009-anthropometrics-pipeline.md)           | Anthropometrics Pipeline                                        | Accepted | 2026-05-09 |
| [0010](0010-anthropometrics-pipeline.md)           | Anthropometrics Pipeline v2 (supersedes 0009)                   | Accepted | 2026-05-09 |
| [0011](0011-plot-style-toolkit.md)                 | Plot Style Toolkit                                              | Accepted | 2026-05-09 |
| [0012](0012-canonical-pose-interchange.md)         | Canonical Pose Interchange                                      | Accepted | 2026-05-09 |
| [0013](0013-launcher-composability.md)             | Launcher Composability — Embeddable-tool contract and IPC layer | Accepted | 2026-05-09 |
| [0014](0014-shared-biomech-models.md)              | Shared Biomechanical Models                                     | Accepted | 2026-05-09 |
| [0015](0015-rust-python-callback-pattern.md)       | Rust-Python Callback Pattern                                    | Accepted | 2026-05-09 |
| [0016](0016-error-handling-discipline.md)          | Error-handling discipline and the ratchet pattern               | Accepted | 2026-05-21 |
| [0017](0017-rust-tools-core-git-dependency.md)     | Pin `tools-core` as a Git Dependency (formerly ADR-0005)        | Accepted | 2026-04-23 |
| [0018](0018-multi-source-motion-targets.md)        | Multi-Source Motion Targets (formerly ADR-0006)                 | Accepted | 2026-05-08 |
| [0019](0019-motion-pipeline-architecture.md)       | Motion Pipeline Architecture — CIR (formerly ADR-0007)          | Proposed | 2026-05-08 |

## ADR Backlog

1. Engine adapter boundary ownership and contract lifecycle.
2. UI/API orchestration boundaries and dependency direction.
3. CI quality gate scope and blocking policy.
