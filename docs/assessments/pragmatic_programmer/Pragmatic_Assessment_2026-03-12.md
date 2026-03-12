# Pragmatic Programmer Assessment: 2026-03-12

**Assessor**: Claude (automated)
**Date**: 2026-03-12
**Framework**: The Pragmatic Programmer (Hunt & Thomas) principles

---

## Summary

This assessment applies the Pragmatic Programmer principles to UpstreamDrift and identifies
areas where the codebase adheres to or violates the core tenets:
DRY, DbC, TDD, Orthogonality, Reversibility, and Broken Windows detection.

---

## DRY (Don't Repeat Yourself)

### Violations Found

**1. Flight model derivative computation (MODERATE)**

`WaterlooPennerModel.simulate()` and `MacDonaldHanzelyModel.simulate()` and `ConstantCoefficientModel.simulate()`
all share nearly identical ODE derivative function structure:

- Same `if speed < MIN_SPEED_THRESHOLD: return [v, 0, 0, -g]` guard
- Same `acc[2] -= launch.gravity` gravity term
- Same `np.array([v_val[0], v_val[1], v_val[2], ...])` return format

The `_run_ode_simulation()` consolidation is excellent, but the derivative body is still partially duplicated.

**Location**: `src/shared/python/physics/flight_models.py:287-390`

**2. TopographyData nested loop sampling (MODERATE)**

`to_heightmap()` and `sample_uniform()` both iterate `for i in range(ny): for j in range(nx)`.
These could be vectorized with `np.vectorize` or meshgrid.

**Location**: `src/shared/python/physics/topography.py:304-322, 560-578`

**3. Launcher UI pass duplication (MINOR)**

`src/launchers/unified_launcher.py` has 7 bare `pass` and several try/except patterns
that repeat across `launcher_dialogs.py`, `settings_dialog.py`, `golf_launcher.py`.

---

## DbC (Design by Contract)

### Current State

The codebase has a working DbC system in `src/shared/python/core/contracts/`:

- `@precondition` decorator with lambda validators
- `PreconditionError`, `PostconditionError` exceptions
- GitHub issue #1755 tracking fleet-wide adoption target (20% preconditions by Q2 2026)

### Coverage Analysis

**Well-Covered (DbC applied):**

- `SecurityManager.hash_password()` — precondition: non-empty string
- `SecurityManager.verify_password()` — two preconditions
- `SecurityManager.create_access_token()` — data must contain 'sub'
- `SecurityManager.verify_token()` — token non-empty, token_type valid
- `UsageTracker.check_quota()` — resource_type must be valid enum
- `compute_prefix_hash()` — prefix must be non-empty string

**Missing Contracts (HIGH PRIORITY):**

- `TopographyData.get_elevation_at()` — no precondition on position shape
- `TopographyData.set_heightmap()` — no precondition on heightmap dimensions
- `AerodynamicsEngine.compute_forces()` — no precondition on velocity/spin shapes
- `AerodynamicsEngine.compute_acceleration()` — no precondition that mass > 0
- `FlightModelRegistry.get_model()` — no postcondition that returned model is not None
- `ImpactModel.solve()` — no preconditions on pre_state validity

**Example Missing Contract:**

```python
# Current (no contract):
def compute_acceleration(self, velocity, spin, mass, ...):
    forces = self.compute_forces(velocity, spin, ...)
    return forces["total"] / mass  # Division by zero if mass=0!

# Should be:
@precondition(lambda self, velocity, spin, mass, ...: mass > 0, "mass must be positive")
def compute_acceleration(self, velocity, spin, mass, ...):
    ...
```

---

## TDD (Test-Driven Development)

### Coverage Gaps

The completist report shows 416 critical stubs. Testing analysis shows:

**Tested and passing:**

- `test_aerodynamics.py`: 65 tests PASS
- `test_flight_models.py`: 21 tests PASS
- `test_impact_model.py`: 41 tests PASS
- `test_terrain.py`: 59 tests PASS
- `test_terrain_engine.py`: 26 tests PASS
- `test_flexible_shaft.py`: 43 tests PASS

**Missing Test Coverage:**

- `topography.py` — `TopographyProvider` Protocol is untested
  - `get_elevation_at()` with out-of-bounds positions
  - `get_gradient_at()` numerical accuracy
  - `from_file()` with various formats
- `plot_engine/protocols.py` — 5 stub methods untested
- `pose_estimation/interface.py` — 3 stub methods untested
- `model_generation/plugins/` — 4 stubs untested
- `model_generation/library/repository.py` — 4 stubs untested

**TDD Principle Violation:**
The completist report lists `flight_models.py:160,166,172,177` as stubs, but these are actually
the `BallFlightModel` ABC abstract method docstrings — Protocol stubs by design, not implementation
gaps. The completist scanner is false-positiving on `...` in abstract method bodies.
Tests for concrete implementations DO exist and pass.

---

## Orthogonality

### Well-Orthogonal Areas

1. **Force models**: `DragModel`, `LiftModel`, `MagnusModel` are independent components
   with no shared state. Each calculates one force type. Excellent orthogonality.

2. **Engine sub-protocols**: `Loadable`, `Steppable`, `Queryable`, `DynamicsComputable`,
   `CounterfactualComputable` are composable sub-protocols. Consumers can depend on only
   what they need.

3. **TerrainMixin**: Properly uses mixin pattern to add terrain to any engine without
   modifying base class.

### Coupling Issues

1. **`AuthCache` + `SecurityManager` coupling**: `auth_cache` is a module-level global
   in `security.py`. Any code importing `security.py` implicitly gets the cache instance.
   This makes unit testing difficult.

2. **`FlightModelRegistry._models` class variable**: Shared mutable class state means
   test pollution — if one test mutates the registry, all subsequent tests see the mutation.

3. **`TopographyData._interpolator`**: The interpolator type changes based on data source
   (RegularGridInterpolator for heightmaps, RBFInterpolator for contours). This requires
   callers to know which type was used. The `if self._heightmap is not None:` branch in
   `get_elevation_at()` is a code smell.

---

## Reversibility

### Well-Reversible Areas

1. **`AerodynamicsConfig.with_changes()`**: Returns a new config — immutable pattern.
2. **Frozen dataclasses**: `AerodynamicsConfig`, `WindConfig`, `ConstantCoefficientSpec`
   all use `frozen=True`.
3. **Engine toggles**: All aerodynamic effects can be toggled on/off at runtime.

### Irreversibility Issues

1. **`TopographyData.set_heightmap()`** with `smooth=True` applies Gaussian filter in-place
   (stores filtered copy). Original data is lost.

2. **`AuthCache.set()`** uses `self._cache.clear()` on overflow — loses all cached entries
   atomically rather than LRU eviction.

---

## Broken Windows

### Detected Broken Windows

1. **Silent exception handlers** (CRITICAL BROKEN WINDOW):

   ```python
   try:
       something()
   except Exception:
       pass  # Silent failure
   ```

   Found in: `sim_widget.py`, `drake_gui_viz.py`, `sim_rendering_mixin.py`, `biomechanics.py`

   These teach contributors that silencing errors is acceptable.

2. **`pass` in abstract-looking class body** (`model_card.py:pass`, `base.py:pass`):
   Empty class bodies suggest unfinished work.

3. **`launcher_diagnostics.py:pass`** at class body level — unimplemented diagnostic class.

4. **Completist report inflation**: The scanner counts Protocol/ABC `...` as "stubs",
   which inflates the gap count. This makes the report less actionable — contributors
   can't tell real gaps from intentional Protocol definitions.

---

## Tracer Bullets

### Incomplete Feature Traces

1. **Gear Effect in Impact Model**: `ImpactParameters.gear_effect_factor` defined but
   the `_compute_friction_spin()` method has a comment indicating the gear effect
   calculation is not implemented.

2. **Flexible shaft in engines**: `PhysicsEngine.set_shaft_properties()` returns `False`
   by default. No concrete engine implementation found that returns `True`.

3. **ZTCF/ZVCF counterfactuals**: Defined in `PhysicsEngine` Protocol with full docs.
   MockEngine implements them. Unknown if MuJoCo/Drake/Pinocchio implementations pass parity tests.

---

## Recommended Priority Actions

### Immediate (BLOCKER)

- None — no true blockers found in fresh read.

### High Priority (CRITICAL)

1. Add `test_topography.py` with tests for `TopographyData` (missing test coverage)
2. Add `@precondition(lambda ...: mass > 0)` to `compute_acceleration()` in `AerodynamicsEngine`
3. Fix silent `except: pass` patterns in MuJoCo/Drake GUI code — at minimum log the error
4. Add `@precondition` to `AerodynamicsEngine.compute_forces()` for velocity/spin shape

### Medium Priority (MAJOR)

5. Vectorize `TopographyData.to_heightmap()` and `sample_uniform()` — replace nested loops
6. Make `FlightModelRegistry._models` an instance variable or use `functools.lru_cache`
7. Move `auth_cache` into a dependency injection pattern

### Low Priority (MINOR)

8. Consolidate derivative computation template in flight models
9. Document why `TopographyProvider` Protocol has `...` bodies (it's intentional — Protocol)
10. Add postcondition to `FlightModelRegistry.get_model()`: returned model is never None

---

_Generated by automated Pragmatic Programmer assessment on 2026-03-12._
