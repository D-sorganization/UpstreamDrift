---
title: "HIGH: DRY violations — 150+ deep imports, 458 no-op pass statements, monolithic contracts.py"
labels: ["high", "technical-debt", "dry", "refactoring"]
---

## Severity: High

## Summary

The codebase repeats import patterns across every launcher, uses `pass` as a structural filler 458 times, and houses a 708-line `contracts.py` module that admits its own `ARCHITECTURE_DEBT`.

## Evidence

### Per-Launcher Import Duplication

```python
# src/launchers/mujoco_dashboard.py
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import MuJoCoPhysicsEngine

# src/launchers/drake_dashboard.py
from src.engines.physics_engines.drake.python.drake_physics_engine import DrakePhysicsEngine

# src/launchers/pinocchio_dashboard.py
from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import PinocchioPhysicsEngine
```

### No-Op `pass` as Structural Filler

```python
# 458 instances of bare `pass` across the codebase
# Used in empty blocks, stub classes, and exception handlers
```

### `contracts.py` Monolith

```python
# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
```

708 lines containing: decorators, exceptions, predicates, validation helpers, domain checks, module identity guards, and backward-compatibility shims.

## Root Cause

- No factory/registry abstraction for engines
- `pass` used instead of `raise NotImplementedError` or `...` for stubs
- Module-level responsibilities never refactored into subpackages

## Remediation Plan

### Phase 1: Immediate (Week 1)

- [ ] Create `EngineRegistry` class to centralize engine loading
- [ ] Replace 10 most frequent `pass` fillers with `raise NotImplementedError` or `...`
- [ ] Extract `contracts.exceptions`, `contracts.predicates`, `contracts.validators` from monolith

### Phase 2: Short-term (Month 1)

- [ ] All launchers use `EngineRegistry` (zero direct engine imports)
- [ ] Add ruff rule to flag bare `pass` in non-exception blocks
- [ ] `contracts.py` split into 4+ focused modules

### Phase 3: Long-term (Quarter)

- [ ] All engine paths go through registry
- [ ] `pass` count < 100 (only in legitimate exception handlers)
- [ ] Zero modules > 300 lines without `ARCHITECTURE_DEBT` admission

## Acceptance Criteria

- [ ] `EngineRegistry` used by all launchers
- [ ] `contracts.py` < 200 lines or split into submodules
- [ ] `pass` in non-exception blocks flagged by linter
