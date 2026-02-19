import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Save original modules so we can restore them after mocking.
# This prevents pollution of sys.modules for other test modules.
_saved_modules = {}
for _key in ["casadi", "pinocchio", "pinocchio.casadi"]:
    if _key in sys.modules:
        _saved_modules[_key] = sys.modules[_key]

# Mock dependencies temporarily to allow importing optimize_arm
sys.modules["casadi"] = MagicMock()
sys.modules["pinocchio"] = MagicMock()
sys.modules["pinocchio.casadi"] = MagicMock()

import casadi as ca  # noqa: E402
import pinocchio as pin  # noqa: E402
import pinocchio.casadi as cpin  # noqa: E402

# Use sys.modules.pop instead of reload to avoid C-extension corruption
sys.modules.pop("src.shared.python.optimization.examples.optimize_arm", None)
from src.shared.python.optimization.examples.optimize_arm import main  # noqa: E402

# Restore original modules IMMEDIATELY to prevent polluting other test modules.
# The module-level code above runs at pytest collection time, so without this
# restore, sys.modules["pinocchio"] would remain a MagicMock during the
# entire collection phase, breaking any test that imports pinocchio afterward.
for _key in ["casadi", "pinocchio", "pinocchio.casadi"]:
    if _key in _saved_modules:
        sys.modules[_key] = _saved_modules[_key]
    elif _key in sys.modules:
        del sys.modules[_key]


def setup_module(module):
    """Re-install mocks for test execution in this module."""
    for key in ["casadi", "pinocchio", "pinocchio.casadi"]:
        if key in sys.modules:
            _saved_modules.setdefault(key, sys.modules[key])
    sys.modules["casadi"] = ca
    sys.modules["pinocchio"] = pin
    sys.modules["pinocchio.casadi"] = cpin


def teardown_module(module):
    """Clean up sys.modules pollution by restoring original modules."""
    for key in ["casadi", "pinocchio", "pinocchio.casadi"]:
        if key in _saved_modules:
            sys.modules[key] = _saved_modules[key]
        elif key in sys.modules:
            del sys.modules[key]


@pytest.fixture
def mock_casadi():
    opti = MagicMock()
    # Mock variable creation
    mock_var = MagicMock()
    # Ensure it looks like an array for numpy broadcasting if needed
    mock_var.__getitem__.return_value = MagicMock()
    mock_var.__len__.return_value = 2

    # Mock comparisons
    mock_var.__eq__ = MagicMock()  # type: ignore[method-assign]
    mock_var.__eq__.return_value = MagicMock()  # type: ignore[attr-defined]

    # Slicing returns
    mock_slice_return = MagicMock()
    mock_slice_return.__len__.return_value = 2
    mock_slice_return.__eq__ = MagicMock()  # type: ignore[method-assign]
    mock_slice_return.__eq__.return_value = MagicMock()  # type: ignore[attr-defined]
    mock_var.__getitem__.return_value = mock_slice_return

    opti.variable.return_value = mock_var

    # Mock bounds and constraints
    opti.bounded.return_value = MagicMock()
    opti.subject_to.return_value = MagicMock()
    opti.minimize.return_value = MagicMock()
    opti.solver.return_value = MagicMock()

    # Mock solve
    sol = MagicMock()

    # Set up value side effect to return appropriate mock data
    call_count = 0

    def value_side_effect(arg):
        nonlocal call_count
        call_count += 1
        # Return data based on call order: Q, V, U, cost
        if call_count == 1:
            return np.zeros((2, 41))  # Q matrix
        if call_count == 2:
            return np.zeros((2, 41))  # V matrix
        if call_count == 3:
            return np.zeros((2, 40))  # U matrix
        return 0.1234  # Cost (scalar)

    sol.value.side_effect = value_side_effect

    opti.solve.return_value = sol

    ca.Opti.return_value = opti

    # Also mock sumsqr since it's used
    ca.sumsqr.return_value = MagicMock()

    return opti


@pytest.fixture
def mock_pinocchio():
    # Mock model
    model = MagicMock()
    model.nq = 2
    model.nv = 2
    model.nu = 2
    pin.buildModelFromUrdf.return_value = model

    # Mock casadi model
    cmodel = MagicMock()
    cpin.Model.return_value = cmodel
    cpin.Model.createData.return_value = MagicMock()

    # Mock ABA
    cpin.aba.return_value = MagicMock()  # Symbolic result

    return model


def test_main_execution(mock_casadi, mock_pinocchio):
    with (
        patch("os.path.exists", return_value=True),
        patch(
            "src.shared.python.optimization.examples.optimize_arm.np.savetxt"
        ) as mock_save,
    ):
        main()

        # Verify solver called
        mock_casadi.solver.assert_called_with(
            "ipopt", {"expand": True}, {"max_iter": 1000, "print_level": 5}
        )
        mock_casadi.solve.assert_called()

        # Verify output saved
        assert mock_save.call_count == 3


def test_main_missing_dependencies():
    with (
        patch(
            "src.shared.python.optimization.examples.optimize_arm.DEPENDENCIES_AVAILABLE",
            False,
        ),
        patch(
            "src.shared.python.optimization.examples.optimize_arm.MISSING_DEP_ERROR",
            "Test Error",
            create=True,
        ),
        patch(
            "src.shared.python.optimization.examples.optimize_arm.logger"
        ) as mock_logger,
    ):
        main()
        mock_logger.error.assert_any_call(
            "Skipping optimize_arm.py due to missing dependencies: Test Error"
        )


def test_urdf_not_found():
    with patch("os.path.exists", return_value=False), pytest.raises(SystemExit):
        main()


def test_optimization_failure(mock_casadi, mock_pinocchio):
    mock_casadi.solve.side_effect = Exception("Infeasible")

    with patch("os.path.exists", return_value=True), pytest.raises(SystemExit):
        main()
