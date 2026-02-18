# ADR-0002: Physics Engine Plugin Architecture

- Status: Accepted
- Date: 2026-02-18
- Decision Makers: Core maintainers
- Related Issues/PRs: #1486

## Context

The suite supports 5 physics engines (MuJoCo, Drake, Pinocchio, OpenSim,
MyoSuite). Each engine has different capabilities, dependencies, and APIs. Users
may only have a subset of engines installed depending on their use case and
platform. We need a way to add, remove, or swap engines without modifying core
application code.

Key constraints:

- Engines have heavy binary dependencies that are difficult to install together
- Each engine exposes a different Python API with different conventions
- Not all engines support all capabilities (e.g., muscle modeling, trajectory optimization)
- New engines may be added in the future

## Decision

Use a Protocol-based plugin architecture:

- Define `PhysicsEngine` protocol in `src/shared/python/engine_core/`
- Each engine implements the protocol independently in `src/engines/physics_engines/<name>/`
- Engine availability detected at runtime via `importlib`
- `EngineManager` provides unified access layer with capability queries
- Engine capabilities are declared via a structured capability manifest

## Alternatives Considered

1. **Abstract base class hierarchy**: Requires all engines to inherit from a
   common base. Tight coupling makes it hard to add engines in separate packages.
2. **Configuration-driven factory**: Register engines via config files. Loses
   type safety and makes capability discovery harder.
3. **Monolithic multi-engine module**: All engines in one module with
   conditional imports. Grows unwieldy as engines are added.

## Consequences

- **Positive**: Engines can be installed independently; missing engines degrade gracefully
- **Positive**: New engines added without modifying existing code (open/closed principle)
- **Positive**: Capability queries let the UI adapt to available engines at runtime
- **Negative**: Engine-specific features require protocol extensions or escape hatches
- **Negative**: Runtime detection adds startup overhead (mitigated by caching)
- **Follow-ups**: Define formal engine capability taxonomy as more engines are added

## Validation

- Unit tests mock engine imports to verify graceful degradation when engines are missing
- Integration tests in `tests/integration/cross_engine/` validate protocol compliance
- CI matrix tests each engine in isolation and in combination
