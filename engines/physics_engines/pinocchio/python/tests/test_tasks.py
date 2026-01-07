import sys
from unittest.mock import MagicMock, patch

import pytest


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


def test_create_joint_coupling_task_not_implemented(mock_pinocchio_env):
    """Verify that create_joint_coupling_task raises NotImplementedError."""
    from engines.physics_engines.pinocchio.python.dtack.ik.tasks import (
        create_joint_coupling_task,
    )

    msg = "Joint coupling task requires mapping names to joint indices dynamically"
    with pytest.raises(NotImplementedError, match=msg):
        create_joint_coupling_task(["joint1"], [1.0])
