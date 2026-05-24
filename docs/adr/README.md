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
| [0005](0005-rust-tools-core-git-dependency.md)     | Pin `tools-core` as a Git Dependency                            | Accepted | 2026-04-23 |
| [0006](0006-multi-source-motion-targets.md)        | Multi-Source Motion Targets                                     | Accepted | 2026-05-08 |
| [0007](0007-motion-pipeline-architecture.md)       | Motion Pipeline Architecture (CIR)                              | Proposed | 2026-05-08 |
| [0012](0012-canonical-pose-interchange.md)         | Canonical Pose Interchange                                      | Accepted | 2026-05-09 |
| [0013](0013-launcher-composability.md)             | Launcher Composability — Embeddable-tool contract and IPC layer | Accepted | 2026-05-09 |
| [0017](0017-sidekick-agentic-action-layer.md)      | Sidekick agentic action layer                                   | Accepted | 2026-05-22 |
| [0018](0018-standalone-sidekick.md)                | Standalone Sidekick Application                                 | Accepted | 2026-05-23 |
| [0019](0019-mission-drift-calculators.md)          | Mission-Drift Calculators                                       | Accepted | 2026-04-25 |
| [0020](0020-canonical-urdf-subsystem.md)           | Canonical URDF subsystem                                        | Proposed | 2026-05-08 |
| [0021](0021-container-strategy.md)                 | Root container surface policy                                   | Accepted | 2026-05-24 |

## ADR Backlog

1. Engine adapter boundary ownership and contract lifecycle.
2. UI/API orchestration boundaries and dependency direction.
3. CI quality gate scope and blocking policy.
