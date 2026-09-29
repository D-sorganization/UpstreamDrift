---
title: "CRITICAL: Law of Demeter violations — 150+ six-level-deep import chains and private attr access"
labels: ["critical", "technical-debt", "lod", "coupling"]
---

## Severity: Critical

## Summary

The codebase exhibits **150+ instances** of 6+ level deep attribute/module chaining and pervasive private attribute access with `# type: ignore[attr-defined]`. This creates brittle coupling where client code depends on internal implementation details.

## Evidence

### Deep Module Path Anti-Pattern

```python
# src/launchers/pinocchio_dashboard.py
from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
    PinocchioPhysicsEngine,
)
```

Repeated identically across `mujoco_dashboard.py`, `drake_dashboard.py`, `cross_engine_dashboard.py`, etc.

### Private Attribute Access with Type Suppression

```python
# src/api/routes/physics.py
engine_manager._speed_factor = request.speed_factor  # type: ignore[attr-defined]
engine_manager._is_recording = True                  # type: ignore[attr-defined]
engine_manager._recorded_frames = []                 # type: ignore[attr-defined]
```

### String-Based Module Paths (Not Type-Checkable)

```python
# src/shared/python/perturbation/cross_engine_runner.py
"src.engines.physics_engines.pinocchio.python.perturbation.analyzer|PinocchioPerturbationAnalyzer"
```

## Root Cause

- No unified engine factory/registry abstraction
- API routes directly mutate physics engine internals
- Import paths encode physical directory structure rather than logical interfaces

## Remediation Plan

### Phase 1: Immediate (Week 1)

- [ ] Create `EngineRegistry` in `src/engines/registry.py` to replace per-launcher imports
- [ ] Extract `TrajectoryRecorder` protocol from `physics.py` route to formalize recording interface
- [ ] Add `__all__` exports to engine modules to shorten import paths

### Phase 2: Short-term (Month 1)

- [ ] Replace string-based module paths with enum-based engine identifiers
- [ ] Introduce `PhysicsEngineFacade` to shield API routes from engine internals
- [ ] Add architecture test: ban `type: ignore[attr-defined]` in `src/api/`

### Phase 3: Long-term (Quarter)

- [ ] All engine access goes through registry/factory (zero direct imports in launchers)
- [ ] Deep imports restricted to `< 4` levels via linter rule
- [ ] Private attribute access in `src/api/` reduced to zero

## Acceptance Criteria

- [ ] No import path deeper than 4 levels in `src/launchers/`
- [ ] Zero `# type: ignore[attr-defined]` in `src/api/routes/`
- [ ] All engine instantiation uses `EngineRegistry.get(name)` API
