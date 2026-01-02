import sys
from unittest.mock import MagicMock, patch

import numpy as np


# Use a patch for the import since mujoco might not be installed
@patch.dict("sys.modules", {"mujoco": MagicMock()})
def test_mujoco_iaa_logic():
    # Add the mujoco python directory to sys.path so we can import mujoco_humanoid_golf
    from pathlib import Path

    root_path = Path.cwd()
    target_path = root_path / "engines" / "physics_engines" / "mujoco" / "python"

    if str(target_path) not in sys.path:
        sys.path.append(str(target_path))

    # Now we can import the module
    from mujoco_humanoid_golf.rigid_body_dynamics.induced_acceleration import (
        MuJoCoInducedAccelerationAnalyzer,
    )
    import mujoco

    # Setup Mocks
    mock_model = MagicMock()
    mock_model.nv = 2

    mock_data = MagicMock()
    # Initial state
    mock_data.qvel = np.array([1.0, 2.0])
    mock_data.qM = np.eye(2).flatten()  # Simplified logic for qM access
    mock_data.qfrc_bias = np.array([10.0, 20.0])  # Term C+G normally

    # Analyzer relies on calling mj_fullM
    def side_effect_fullM(model, M, qM):
        M[:] = np.eye(2)  # Identity mass matrix
    mujoco.mj_fullM.side_effect = side_effect_fullM

    # We need to simulate mj_forward changing qfrc_bias based on qvel
    # When qvel is 0, qfrc_bias = G. Let's say G = [0, 9.8]
    # When qvel is valid, qfrc_bias = C+G = [10, 20]

    def side_effect_forward(model, data):
        if np.all(data.qvel == 0):
            data.qfrc_bias = np.array([0.0, 5.0])  # G term
        else:
            data.qfrc_bias = np.array([10.0, 20.0])  # C+G term

    mujoco.mj_forward.side_effect = side_effect_forward

    # Run Test
    analyzer = MuJoCoInducedAccelerationAnalyzer(mock_model, mock_data)
    results = analyzer.compute_components(tau_app=np.array([2.0, 0.0]))

    # Check logic
    # M = I
    # G = [0, 5]
    # C+G = [10, 20] => C = [10, 15]
    # acc_g = -G = [0, -5]
    # acc_c = -C = [-10, -15]
    # acc_t = tau = [2, 0]

    np.testing.assert_allclose(results["gravity"], np.array([0.0, -5.0]))
    np.testing.assert_allclose(results["velocity"], np.array([-10.0, -15.0]))
    np.testing.assert_allclose(results["control"], np.array([2.0, 0.0]))
    np.testing.assert_allclose(results["total"], np.array([-8.0, -20.0]))

    # Verify qvel was restored
    np.testing.assert_array_equal(mock_data.qvel, np.array([1.0, 2.0]))

    # Verify mj_forward called exactly twice (once for G, once for restore)
    # Actually wait. mj_forward called inside compute_components 2 times (try + finally).
    # But compute_components logic:
    # 1. try: mj_forward (for G) -> 1 call
    # 2. finally: mj_forward (for restore) -> 1 call
    # Total 2 calls.
    # What about the initialization? No usage there.
    # What about line 50 mj_fullM? Not mj_forward.
    
    # Wait, in logic 2. Compute G(q)
    # mujoco.mj_forward(self.model, self.data)
    # finally:
    # mujoco.mj_forward(self.model, self.data)
    
    # But we run test flow manually, calling compute_components once. 
    # Yes, 2 calls.
    
    assert mujoco.mj_forward.call_count == 2
