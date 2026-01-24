from unittest.mock import MagicMock, patch

import numpy as np

from src.shared.python.path_utils import setup_import_paths

# Import paths configured at test runner level and with setup_import_paths
setup_import_paths()


# Use a patch for the import since mujoco might not be installed
@patch.dict("sys.modules", {"mujoco": MagicMock()})
def test_mujoco_iaa_logic():

    # Now we can import the module
    import mujoco
    from mujoco_humanoid_golf.rigid_body_dynamics.induced_acceleration import (
        MuJoCoInducedAccelerationAnalyzer,
    )

    # Setup Mocks
    mock_model = MagicMock()
    mock_model.nv = 2

    mock_data = MagicMock()
    # Initial state
    mock_data.qvel = np.array([1.0, 2.0])
    mock_data.qM = np.eye(2).flatten()  # Simplified logic for qM access
    mock_data.qfrc_bias = np.array([10.0, 20.0])  # Term C+G normally
    mock_data.qfrc_constraint = np.array([0.0, 0.0])  # Constraint forces (nv=2)
    mock_data.qfrc_actuator = np.array([0.0, 0.0])  # Actuator forces (nv=2)

    # Analyzer relies on calling mj_fullM
    def side_effect_fullM(model, M, qM):
        M[:] = np.eye(2)  # Identity mass matrix

    mujoco.mj_fullM.side_effect = side_effect_fullM

    # We need to simulate mj_forward changing qfrc_bias based on qvel
    # When qvel is 0, qfrc_bias = G. Let's say G = [0, GRAVITY_M_S2]
    # When qvel is valid, qfrc_bias = C+G = [10, 20]

    def side_effect_forward(model, data):
        if np.all(data.qvel == 0):
            data.qfrc_bias = np.array([0.0, 5.0])  # G term
        else:
            data.qfrc_bias = np.array([10.0, 20.0])  # C+G term

    mujoco.mj_forward.side_effect = side_effect_forward

    # Mock mj_rne for optimized G calculation
    def side_effect_rne(model, data, flg_acc, result):
        # mj_rne with flg_acc=0 computes G if qvel=0.
        # In test, we expect G = [0, 5]
        # Verify qvel is 0
        if np.all(data.qvel == 0):
            result[:] = np.array([0.0, 5.0])
        else:
            # Just in case, but we shouldn't hit this path with new logic
            result[:] = np.array([10.0, 20.0])

    mujoco.mj_rne.side_effect = side_effect_rne

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

    # Verify mj_forward called exactly once (for restore)
    # The optimization replaces one mj_forward with mj_rne
    assert mujoco.mj_forward.call_count == 1
    # Verify mj_rne called once
    assert mujoco.mj_rne.call_count == 1
