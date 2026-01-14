import sys
from unittest.mock import MagicMock, patch

import pytest
import numpy as np


@pytest.fixture
def mock_pinocchio_env():
    """Mock pinocchio and pink dependencies."""
    mock_mods = {
        "pink": MagicMock(),
        "pink.tasks": MagicMock(),
        "pinocchio": MagicMock(),
    }
    with patch.dict(sys.modules, mock_mods):
        # Clean up module under test to ensure it imports mocks
        if "engines.physics_engines.pinocchio.python.dtack.ik.tasks" in sys.modules:
            del sys.modules["engines.physics_engines.pinocchio.python.dtack.ik.tasks"]
        yield


def test_create_joint_coupling_task(mock_pinocchio_env):
    """Verify that create_joint_coupling_task works as expected."""
    import pinocchio as pin  # This is the mocked pinocchio
    from engines.physics_engines.pinocchio.python.dtack.ik.tasks import (
        create_joint_coupling_task,
    )

    # Setup mock model
    mock_model = pin.Model()
    mock_model.nv = 10
    mock_model.existJointName.return_value = True

    # Mock getJointId and corresponding joint info
    mock_model.getJointId.return_value = 1
    mock_joint = MagicMock()
    mock_joint.idx_v = 5
    mock_model.joints = [None, mock_joint] # index 1 access

    # Call the function
    joint_names = ["joint1"]
    ratios = [1.0]
    task = create_joint_coupling_task(mock_model, joint_names, ratios)

    # Verify the task was created with correct A matrix
    # The implementation calls pink.tasks.LinearHolonomicTask(A, b, cost=cost)
    # We can check the arguments passed to the mock
    import pink.tasks

    args, kwargs = pink.tasks.LinearHolonomicTask.call_args
    A_matrix = args[0]
    b_vector = args[1]

    assert A_matrix.shape == (1, 10)
    assert A_matrix[0, 5] == 1.0
    assert np.all(b_vector == 0)
    assert kwargs['cost'] == 100.0
