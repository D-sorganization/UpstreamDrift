# Critical Incomplete Implementation in plotting_core.py

**Status**: Open
**Priority**: Critical
**Created**: 2026-01-29
**Module**: Shared Utility
**File**: `./src/shared/python/plotting_core.py`

## Description
The following critical incomplete implementations (stubs, placeholders, or NotImplementedError) were identified in `plotting_core.py`. These items block core functionality or represent significant technical debt in non-test code.

## Findings

| Line | Function/Context | Issue Type |
|------|------------------|------------|
| 78 | `get_time_series` | Stub |
| 89 | `get_induced_acceleration_series` | Stub |
| 102 | `get_counterfactual_series` | Stub |

## Impact
- **Blocking**: Features relying on these functions will fail or return incomplete data.
- **Stability**: Potential for runtime errors (NotImplementedError) or silent failures (pass).

## Recommended Action
1. Review the listed stubs.
2. Implement the missing logic or remove the function if unused.
3. If the function is an interface definition, ensure it is marked as abstract or uses `Protocol`.

## References
- Completist Audit Report ({today})
