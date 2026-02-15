# Critical Incomplete: PyVista Backend

**Status:** Critical
**Type:** Missing Implementation
**File:** `src/unreal_integration/viewer_backends.py`
**Line:** 743

## Description
The `create_viewer` factory function raises `NotImplementedError` when `BackendType.PYVISTA` is requested. This blocks users from using the PyVista-based visualization backend, which is planned as a core feature.

## Impact
- **Severity**: 5/5
- **User Impact**: Users cannot access desktop-native visualization.
- **Blocking**: Yes, blocks `pyvista` mode.

## Proposed Solution
Implement `PyVistaBackend` class inheriting from `ViewerBackend` in `src/unreal_integration/viewer_backends.py`. This class should use `pyvista` library to render meshes and handle scene updates.

## Verification
- Can successfully call `create_viewer("pyvista")`.
- `PyVistaBackend` implements all abstract methods of `ViewerBackend`.
