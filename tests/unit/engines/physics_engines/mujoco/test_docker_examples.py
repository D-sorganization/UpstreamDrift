"""Unit tests for Docker examples and utilities."""

import sys
from unittest.mock import MagicMock, patch


# Mock dm_control and imageio before importing modules
mock_dm_control = MagicMock()
mock_imageio = MagicMock()

with patch.dict(
    sys.modules,
    {
        "dm_control": mock_dm_control,
        "dm_control.suite": mock_dm_control.suite,
        "dm_control.mjcf": mock_dm_control.mjcf,
        "imageio": mock_imageio,
    },
):
    from src.engines.physics_engines.mujoco.docker import dump_names
    from src.engines.physics_engines.mujoco.docker import example_dynamic_stance


def test_dump_names_main():
    """Test the dump_names main script."""
    with patch.dict(
        sys.modules,
        {"dm_control": mock_dm_control, "dm_control.suite": mock_dm_control.suite},
    ):
        mock_env = MagicMock()
        mock_env.physics.model.ngeom = 2
        mock_env.physics.model.nbody = 2
        mock_env.physics.model.id2name.side_effect = lambda i, t: f"{t}_{i}"
        mock_dm_control.suite.load.return_value = mock_env

        # Test main execution
        dump_names.main()

        # Verify it loads the correct suite
        mock_dm_control.suite.load.assert_called_once_with(
            domain_name="humanoid_CMU", task_name="stand"
        )
        # Verify id2name is called
        assert mock_env.physics.model.id2name.call_count == 4


def test_example_dynamic_stance_get_cmu_xml_path():
    """Test get_cmu_xml_path in example_dynamic_stance."""
    with patch.dict(
        sys.modules,
        {"dm_control": mock_dm_control, "dm_control.suite": mock_dm_control.suite},
    ):
        mock_dm_control.suite.__file__ = "/mock/path/to/suite/__init__.py"
        path = example_dynamic_stance.get_cmu_xml_path()
        assert "humanoid_CMU.xml" in path


def test_example_dynamic_stance_pd_control():
    """Test pd_control logic."""
    mock_physics = MagicMock()

    def mock_qpos_get(name):
        return 0.0

    def mock_qvel_get(name):
        return 0.0

    mock_physics.named.data.qpos.__getitem__.side_effect = mock_qpos_get
    mock_physics.named.data.qvel.__getitem__.side_effect = mock_qvel_get
    mock_physics.model.nu = 5

    target_pose = {"joint1": 1.0}
    actuators = {"joint1": 2}

    action = example_dynamic_stance.pd_control(
        mock_physics, target_pose, actuators, kp=10.0, kd=1.0
    )

    assert action.shape == (5,)
    assert action[2] == 10.0  # (10.0 * (1.0 - 0.0)) - (1.0 * 0.0) = 10.0


def test_example_dynamic_stance_customize_model():
    """Test customize_model geometric coloring."""
    mock_physics = MagicMock()
    mock_physics.model.ngeom = 3

    def mock_id2name(i, t):
        names = ["left_eye", "right_golf_club", "torso"]
        return names[i]

    mock_physics.model.id2name = mock_id2name
    mock_physics.model.geom_rgba = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    example_dynamic_stance.customize_model(mock_physics)

    assert mock_physics.model.geom_rgba[0] == [1.0, 1.0, 1.0, 1.0]  # White eye
    assert mock_physics.model.geom_rgba[1] == [0.8, 0.8, 0.8, 1.0]  # Silver club
    assert mock_physics.model.geom_rgba[2] == [0.6, 0.6, 0.6, 1.0]  # Grey shirt


@patch.object(example_dynamic_stance.Path, "exists")
@patch("builtins.open")
def test_example_dynamic_stance_load_and_patch_xml(mock_open, mock_exists):
    """Test XML loading and patching."""
    with patch.dict(
        sys.modules,
        {"dm_control": mock_dm_control, "dm_control.mjcf": mock_dm_control.mjcf},
    ):
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.read.return_value = '<mujoco><compiler class="main"/></mujoco>'
        mock_open.return_value.__enter__.return_value = mock_file

        mock_root = MagicMock()
        mock_dm_control.mjcf.from_xml_string.return_value = mock_root

        physics = example_dynamic_stance._load_and_patch_xml("/fake/path.xml")

        mock_dm_control.mjcf.from_xml_string.assert_called_once()
        mock_dm_control.mjcf.Physics.from_mjcf_model.assert_called_once_with(mock_root)


def test_example_dynamic_stance_main():
    """Test the main execution loop of dynamic stance."""
    with (
        patch.object(example_dynamic_stance, "get_cmu_xml_path") as mock_get_path,
        patch.object(example_dynamic_stance, "_setup_physics") as mock_setup,
        patch.object(example_dynamic_stance, "customize_model") as mock_customize,
        patch.object(example_dynamic_stance, "_set_initial_pose") as mock_set_pose,
        patch.object(example_dynamic_stance, "_run_simulation_loop") as mock_run,
    ):
        mock_physics = MagicMock()
        mock_physics.model.nu = 2
        mock_physics.model.id2name.side_effect = lambda i, t: f"act_{i}"
        mock_setup.return_value = mock_physics

        example_dynamic_stance.main()

        mock_get_path.assert_called_once()
        mock_setup.assert_called_once()
        mock_customize.assert_called_once()
        mock_set_pose.assert_called_once()
        mock_run.assert_called_once()
