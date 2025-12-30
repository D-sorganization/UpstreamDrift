import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock dependencies
sys.modules["casadi"] = MagicMock()
sys.modules["pinocchio"] = MagicMock()
sys.modules["pinocchio.casadi"] = MagicMock()

import casadi as ca  # noqa: E402
import pinocchio as pin  # noqa: E402
import pinocchio.casadi as cpin  # noqa: E402

from shared.python.optimization.examples.optimize_arm import main  # noqa: E402


@pytest.fixture
def mock_casadi():
    opti = MagicMock()
    # Mock variable creation
    mock_var = MagicMock()
    # Ensure it looks like an array for numpy broadcasting if needed
    mock_var.__getitem__.return_value = MagicMock()
    mock_var.__len__.return_value = 2

    # Mock comparisons
    mock_var.__eq__.return_value = MagicMock()

    # Slicing returns
    mock_slice_return = MagicMock()
    mock_slice_return.__len__.return_value = 2
    mock_slice_return.__eq__.return_value = MagicMock()
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
        elif call_count == 2:
            return np.zeros((2, 41))  # V matrix
        elif call_count == 3:
            return np.zeros((2, 40))  # U matrix
        else:
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
        patch("numpy.savetxt") as mock_save,
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
            "shared.python.optimization.examples.optimize_arm.DEPENDENCIES_AVAILABLE",
            False,
        ),
        patch(
            "shared.python.optimization.examples.optimize_arm.MISSING_DEP_ERROR",
            "Test Error",
            create=True,
        ),
    ):
        with patch("builtins.print") as mock_print:
            main()
            mock_print.assert_any_call(
                "Skipping optimize_arm.py due to missing dependencies: Test Error"
            )


def test_urdf_not_found():
    with patch("os.path.exists", return_value=False):
        with pytest.raises(SystemExit):
            main()


def test_optimization_failure(mock_casadi, mock_pinocchio):
    # Reset side effect to ensure we don't run out of iterator items if main() retries stuff
    # But main() calls solve() then exits if exception.
    mock_casadi.solve.side_effect = Exception("Infeasible")

    with patch("os.path.exists", return_value=True):
        with pytest.raises(SystemExit):
            main()
