import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock dependencies
sys.modules["casadi"] = MagicMock()
sys.modules["pinocchio"] = MagicMock()
sys.modules["pinocchio.casadi"] = MagicMock()

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

from shared.python.optimization.examples.optimize_arm import main


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

    # Mock solve
    sol = MagicMock()

    # Make value return different things based on input
    # Since we can't easily check identity of input mock objects created inside main(),
    # we can try to infer based on call count or just make it return a float sometimes?
    # Or better, make sol.value return a MagicMock that acts as both a float and an array?
    # No, that's hard.
    # Let's inspect call args.
    def value_side_effect(arg):
        # Heuristic: if it looks like the cost (scalar), return float
        # But we don't know what 'arg' is exactly.
        # However, cost comes last in the sequence of calls in main: q, v, u, cost
        # But order isn't guaranteed by dict iteration if that were used, but here it's procedural.
        # q, v, u are arrays. cost is scalar.
        return np.zeros((2, 41))  # Default to array

    # We can use side_effect with an iterator if the order is deterministic
    sol.value.side_effect = [
        np.zeros((2, 41)),  # Q
        np.zeros((2, 41)),  # V
        np.zeros((2, 40)),  # U
        0.1234,  # Cost (scalar float)
    ]

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
