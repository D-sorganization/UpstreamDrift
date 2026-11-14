---
issue: 15108
summary: "Scope tests/unit/conftest.py engine MagicMocks to tests/unit and guard OpenSim dynamics parity (#15108)"
---

- Scope `tests/unit/conftest.py` optional-dependency `MagicMock` injection (`pydrake`, `pydrake.all`, `casadi`, `pinocchio`, `pinocchio.casadi`) to `tests/unit/` items via `pytest_collection_finish`, `pytest_runtest_setup`, and `pytest_runtest_teardown`, resetting the `engine_availability` cache on transitions (#15108).
- Harden `require_real_module` in `tests/integration/cross_engine/loop_nullspace_conformance_support.py` to require a string `__file__` attribute in addition to rejecting `unittest.mock` stand-ins and return the verified module.
- Update `tests/integration/cross_engine/test_opensim_engine_dynamics_parity.py` to remove module-top-level `pytest.importorskip("mujoco")`, split `_MuJoCoReference` and `_PinocchioReference` fixtures using `require_real_module`, and apply `@pytest.mark.requires_pinocchio` / `@pytest.mark.requires_mujoco` markers on the cross-engine parity tests.
- Add regression tests in `tests/unit/ci/test_unit_conftest_mock_scope.py`.
