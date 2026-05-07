# Import mocked modules for use in fixtures below
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock dependencies at test-level via conftest.pytest_configure() and fixtures.
# Mocking is now handled by @patch.dict decorators at test function level,
# preventing module-level sys.modules pollution that can affect other tests.
from src.shared.python.optimization.examples.optimize_arm import main  # noqa: E402

ca = sys.modules.get("casadi", MagicMock())
pin = sys.modules.get("pinocchio", MagicMock())
cpin = sys.modules.get("pinocchio.casadi", MagicMock())


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
    opti.method.return_value = MagicMock()

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


@patch.dict(
    "sys.modules",
    {"casadi": MagicMock(), "pinocchio": MagicMock(), "pinocchio.casadi": MagicMock()},
)
def test_main_execution(mock_casadi, mock_pinocchio):
    with (
        patch("os.path.exists", return_value=True),
        patch(
            "src.shared.python.optimization.examples.optimize_arm.np.savetxt"
        ) as mock_save,
    ):
        main()

        # Verify solver called
        mock_casadi.method.assert_called_with(
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
    mock_casadi.solve.side_effect = RuntimeError("Infeasible")

    with patch("os.path.exists", return_value=True), pytest.raises(SystemExit):
        main()
