# Architecture Decision Records (ADRs)

This directory tracks architecture-impacting decisions for UpstreamDrift.

## Policy

- Use `ADR_TEMPLATE.md` for every new ADR.
- Filename format: `NNNN-short-title.md`.
- Every ADR must include Status, Date, and validation notes.
- Superseded ADRs must link to the replacing ADR.

## Index

| ADR                                                | Title                                       | Status   | Date       |
| -------------------------------------------------- | ------------------------------------------- | -------- | ---------- |
| [0001](0001-fastapi-local-first-api.md)            | FastAPI for Local-First API Design          | Accepted | 2026-02-18 |
| [0002](0002-physics-engine-plugin-architecture.md) | Physics Engine Plugin Architecture          | Accepted | 2026-02-18 |
| [0003](0003-websocket-realtime-simulation.md)      | WebSocket Protocol for Real-Time Simulation | Accepted | 2026-02-18 |

## ADR Backlog

1. Engine adapter boundary ownership and contract lifecycle.
2. UI/API orchestration boundaries and dependency direction.
3. CI quality gate scope and blocking policy.
