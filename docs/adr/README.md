# Architecture Decision Records (ADRs)

This directory tracks architecture-impacting decisions for UpstreamDrift.

## Policy

- Use `ADR_TEMPLATE.md` for every new ADR.
- Filename format: `NNNN-short-title.md`.
- Every ADR must include Status, Date, and validation notes.
- Superseded ADRs must link to the replacing ADR.

## Index

| ADR                                                           | Title                                                                             | Status   | Date       |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------- | ---------- |
| [0001](0001-fastapi-local-first-api.md)                       | FastAPI for Local-First API Design                                                | Accepted | 2026-02-18 |
| [0002](0002-physics-engine-plugin-architecture.md)            | Physics Engine Plugin Architecture                                                | Accepted | 2026-02-18 |
| [0003](0003-websocket-realtime-simulation.md)                 | WebSocket Protocol for Real-Time Simulation                                       | Accepted | 2026-02-18 |
| [0004](0004-launcher-provider-migration.md)                   | Launcher Provider Migration Modes and Legacy Deprecation Policy                   | Accepted | 2026-04-08 |
| [0005](0005-rust-tools-core-git-dependency.md)                | Pin `tools-core` as a Git Dependency                                              | Accepted | 2026-04-23 |
| [0006](0006-multi-source-motion-targets.md)                   | Multi-Source Motion Targets                                                       | Accepted | 2026-05-08 |
| [0007](0007-motion-pipeline-architecture.md)                  | Motion Pipeline Architecture (CIR)                                                | Proposed | 2026-05-08 |
| [0012](0012-canonical-pose-interchange.md)                    | Canonical Pose Interchange                                                        | Accepted | 2026-05-09 |
| [0013](0013-launcher-composability.md)                        | Launcher Composability — Embeddable-tool contract and IPC layer                   | Accepted | 2026-05-09 |
| [0017](0017-sidekick-agentic-action-layer.md)                 | Sidekick agentic action layer                                                     | Accepted | 2026-05-22 |
| [0018](0018-standalone-sidekick.md)                           | Standalone Sidekick Application                                                   | Accepted | 2026-05-23 |
| [0019](0019-mission-drift-calculators.md)                     | Mission-Drift Calculators                                                         | Accepted | 2026-04-25 |
| [0020](0020-canonical-urdf-subsystem.md)                      | Canonical URDF subsystem                                                          | Accepted | 2026-05-08 |
| [0021](0021-container-strategy.md)                            | Container Strategy — Three-Dockerfile Policy                                      | Accepted | 2026-05-25 |
| [0022](0022-chat-sidekick-boundary.md)                        | Chat Sidekick Boundary                                                            | Accepted | 2026-06-12 |
| [0023](0023-mujoco-warp-backend.md)                           | MuJoCo Warp GPU + MuJoCo CPU backends behind one Protocol                         | Accepted | 2026-05-29 |
| [0024](0024-differentiable-backend.md)                        | Differentiable backend — MJX (JAX) vs custom Warp kernels                         | Accepted | 2026-05-29 |
| [0025](0025-jaxsim-backend-home.md)                           | JaxSim Backend Home                                                               | Accepted | 2026-05-30 |
| [0026](0026-canonical-dynamic-state-v2.md)                    | Canonical Dynamic State v2                                                        | Accepted | 2026-05-31 |
| [0027](0027-canonical-viewport-backend.md)                    | Canonical 3D Viewport Backend (Rerun export follow-up executed 2026-08-08, #8405) | Accepted | 2026-05-31 |
| [0028](0028-react-tauri-launcher-parity.md)                   | React/Tauri launcher parity model                                                 | Accepted | 2026-06-10 |
| [0030](0030-c3d-viewer-renderer-backend.md)                   | C3D Viewer Renderer Backend                                                       | Accepted | 2026-06-10 |
| [0031](0031-launch-monitor-canonical-shot-schema.md)          | Canonical Launch Monitor Shot Schema                                              | Accepted | 2026-08-04 |
| [0032](0032-bunkershot3d-club-design-architecture.md)         | BunkerShot3D as a Multi-Fidelity Club-Design Tool                                 | Accepted | 2026-08-13 |
| [0033](0033-bunkershot3d-sand-field-tier.md)                  | Sand-Field Visualization Tier for BunkerShot3D                                    | Proposed | 2026-08-16 |
| [0034](0034-launch-monitor-analysis-contract-v2.md)           | Launch Monitor Analysis Contract V2                                               | Accepted | 2026-08-19 |
| [0035](0035-source-backed-strokes-gained-contract.md)         | Source-Backed Strokes-Gained Contract                                             | Accepted | 2026-08-20 |
| [0036](0036-launch-monitor-identity-boundaries.md)            | Launch Monitor Player, Session, and Order Identity Boundaries                     | Accepted | 2026-08-20 |
| [0037](0037-immutable-launch-monitor-dataset-jobs.md)         | Immutable Launch-Monitor Dataset Jobs                                             | Accepted | 2026-08-20 |
| [0038](0038-launch-monitor-player-covariation-contract.md)    | Canonical Launch Monitor Player Covariation Contract                              | Accepted | 2026-08-20 |
| [0039](0039-attested-launch-monitor-longitudinal-sessions.md) | Attested Launch Monitor Longitudinal Sessions                                     | Accepted | 2026-08-20 |
| [0040](0040-data-free-launch-monitor-conformance-bundle.md)   | Data-Free Launch-Monitor Conformance Bundle                                       | Accepted | 2026-08-21 |
| [0041](0041-markerless-mocap-consumer-authority.md)           | Markerless Mocap Consumer Authority                                               | Accepted | 2026-08-25 |
| [0042](0042-engineering-design-manual-authority.md)           | Engineering Design Manual Authority and Release Boundary                          | Accepted | 2026-08-25 |
| [0043](0043-companion-manifest-provider-authority.md)         | Companion Manifest Provider Authority                                             | Accepted | 2026-08-28 |
| [0044](0044-out-of-plane-fidelity-for-bunkershot3d.md)        | Out-of-Plane Fidelity for BunkerShot3D                                            | Proposed | 2026-08-29 |

Note: ADR 0013 was amended on 2026-05-31 to document the CC-32
canonical-core app-shell registry reuse of the embeddable-tool contract.

## Recent Amendments

- **2026-08-29:** ADR-0044 keeps BunkerShot3D in-plane: F1's
  `RefusedQuantity.OUT_OF_PLANE`, the `EXTRUDED` slice labelling, and the
  ball-as-cylinder caveat are recorded as the tool's durable position rather
  than a stopgap, with a concrete, cheap reopening trigger (fix #9247, then
  run the existing Sobol'/Morris study over `WedgeGeometry`'s heel/toe and
  rocker parameters) instead of a permanent close.
- **2026-08-29:** ADR-0043 adds the #9192 exact-commit publication boundary:
  one shared bundle command, ephemeral protected-main evidence, draft-first
  immutable release acquisition, attestation, and non-fabricated schema
  compatibility history.
- **2026-08-16:** ADR-0033 amends ADR-0032's fidelity-tier table: F1 is
  narrowed from "reduced-order / 2-D plane-strain continuum" to a 2-D
  plane-strain **MPM** solver and becomes the sand-field visualization tier,
  and the F3 MuJoCo proxy is recorded as non-functional rather than merely
  low-fidelity.
- **2026-08-11:** ADR-0019 supersedes its original universal 1 MB PDF ceiling
  with signature validation, a 50,000,000-byte warning, and GitHub's
  100,000,000-byte hard file boundary so in-scope scientific publications can
  retain publication-quality figures.
- **2026-05-31:** ADR-0017 now records the CC-38 canonical-core Sidekick tool
  adapter and its fixed `canonical.*` action allowlist.

## ADR Backlog

1. Engine adapter boundary ownership and contract lifecycle.
2. UI/API orchestration boundaries and dependency direction.
3. CI quality gate scope and blocking policy.
