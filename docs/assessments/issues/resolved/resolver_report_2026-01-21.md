# Issue Resolution Report - 2026-01-21

## Summary
Focused on resolving critical stability issues in cross-engine integration tests, specifically addressing module reload corruption.

## Issues Addressed

### 1. Fix module reload corruption in cross-engine tests (#496)
*   **Priority:** 🔴 Critical
*   **Status:** Resolved
*   **Problem:** Tests were using `importlib.reload()` to refresh module state (particularly `mujoco` and physics engine modules) to handle mocking. This caused corruption in C-extension modules and potential flakiness or crashes, especially when running the full suite.
*   **Fix:**
    *   Refactored `tests/integration/test_mujoco_protocol.py` to remove `importlib.reload`. Used `sys.modules.pop(module_name, None)` before import to force a fresh module load without the side effects of in-place reload on extension modules.
    *   Refactored `tests/integration/test_real_engine_loading.py` to remove `importlib.reload(mujoco)` and `importlib.reload(physics_engine)`. Replaced with `sys.modules.pop(...)` + `import` pattern.
    *   Verified tests pass with the new approach.

## Changes Made
*   Modified `tests/integration/test_mujoco_protocol.py`: Replaced `reload` with `sys.modules.pop` + `import`.
*   Modified `tests/integration/test_real_engine_loading.py`: Replaced `reload` with `sys.modules.pop` + `import` and fixed explicit `mujoco` reload.

## Issues Deferred
*   **#124 Phase 2.2: Archive & Legacy Cleanup**: Lower priority than critical test stability.
*   **#120 Phase 1.2: Consolidate Dependency Management**: `pyproject.toml` seems well structured, further consolidation wasn't blocking critical path.

## New Issues Discovered
*   `tests/unit/test_api_security.py` also uses `importlib.reload(security)`. While less critical (pure Python), it should eventually be refactored to avoid side effects on global configuration state.
