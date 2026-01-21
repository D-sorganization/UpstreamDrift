---
title: "Critical: Fix Test Suite & Environment"
labels: ["jules:assessment", "needs-attention", "priority:high"]
assignees: ["jules-test-generator"]
---

## Problem
The current test coverage is **0.92%**, far below the 10% threshold. Most tests fail to collect or run due to `ImportError` and `ModuleNotFoundError`. The test suite is tightly coupled to heavy physics engines (`mujoco`, `drake`) which are difficult to install in standard CI environments.

## Evidence
- `pytest` collection fails for `shared/python/tests/test_common_utils.py` and many others.
- `coverage` report shows < 1% coverage.

## Proposed Solution
1. **Mock Everything**: Create a `tests/conftest.py` that mocks `mujoco`, `drake`, `pinocchio`, and other heavy libs if they are not present.
2. **Isolate Unit Tests**: Ensure `shared/python` tests do not import engine-specific code unless necessary.
3. **Fix Imports**: Address all `ImportError` issues to ensure 100% test collection.
