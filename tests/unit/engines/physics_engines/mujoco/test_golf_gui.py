"""Tests for the MuJoCo Golf GUI module."""

from unittest.mock import MagicMock, patch


def test_style_mixin():
    """Test StyleMixin methods."""
    with patch(
        "src.engines.physics_engines.mujoco.docker.gui.golf_gui_styles.ttk.Style"
    ) as mock_style:
        from src.engines.physics_engines.mujoco.docker.gui.golf_gui_styles import (
            StyleMixin,
        )

        class TestGUI(StyleMixin):
            pass

        gui = TestGUI()
        gui.setup_styles()

        mock_style.return_value.theme_use.assert_called_once_with("clam")
        mock_style.return_value.configure.assert_called()
        mock_style.return_value.map.assert_called()


def test_docker_mixin_get_docker_cmd():
    """Test _get_docker_cmd from DockerMixin."""
    from src.engines.physics_engines.mujoco.docker.gui.golf_gui_docker import (
        DockerMixin,
    )

    with patch("shutil.which") as mock_which, patch("os.name", "posix"):
        mock_which.return_value = True
        mixin = DockerMixin()
        assert mixin._get_docker_cmd() == ["docker"]

    with patch("shutil.which") as mock_which, patch("os.name", "nt"):
        # if docker is not found but wsl is
        mock_which.side_effect = lambda x: x == "wsl"
        mixin = DockerMixin()
        assert mixin._get_docker_cmd() == ["wsl", "docker"]


def test_docker_mixin_generate_update_dockerfile():
    """Test _generate_update_dockerfile."""
    from src.engines.physics_engines.mujoco.docker.gui.golf_gui_docker import (
        DockerMixin,
    )

    mixin = DockerMixin()
    content = mixin._generate_update_dockerfile()
    assert "FROM upstream-drift:engine" in content
    assert "defusedxml>=0.7.1" in content


class MockDockerProtocolHost:
    def __init__(self):
        self.root = MagicMock()
        self.is_windows = False
        self.wsl_path = "/wsl"
        self.repo_path = "/repo"
        self.live_view_var = MagicMock()
        self.live_view_var.get.return_value = False
        self.stop_event = MagicMock()
        self.stop_event.is_set.return_value = False
        self.process = MagicMock()
        self.btn_run = MagicMock()
        self.btn_stop = MagicMock()
        self.btn_rebuild = MagicMock()
        self.btn_open_video = MagicMock()
        self.btn_open_data = MagicMock()
        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def on_sim_success(self):
        pass


def test_docker_mixin_build_docker_command():
    """Test building docker command."""
    from src.engines.physics_engines.mujoco.docker.gui.golf_gui_docker import (
        DockerMixin,
    )

    class TestMixin(DockerMixin, MockDockerProtocolHost):
        pass

    host = TestMixin()

    # Test Linux without live view
    host.is_windows = False
    host.live_view_var.get.return_value = False
    cmd = host._build_docker_command()
    assert cmd[0] == "docker"
    assert "MUJOCO_GL=osmesa" in cmd

    # Test Linux with live view
    host.live_view_var.get.return_value = True
    cmd = host._build_docker_command()
    assert "MUJOCO_GL=glfw" in cmd

    # Test Windows without live view
    host.is_windows = True
    host.live_view_var.get.return_value = False
    cmd = host._build_docker_command()
    assert cmd[0] == "wsl"
    assert "MUJOCO_GL=osmesa" in cmd


def test_golf_simulation_gui_init():
    """Test the main GUI initialization with mocked tkinter."""
    import sys

    # Create mock for tkinter
    mock_tk = MagicMock()
    mock_ttk = MagicMock()
    mock_messagebox = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "tkinter": mock_tk,
            "tkinter.ttk": mock_ttk,
            "tkinter.messagebox": mock_messagebox,
        },
    ):
        from src.engines.physics_engines.mujoco.docker.gui.deepmind_control_suite_MuJoCo_GUI import (
            GolfSimulationGUI,
        )

        root_mock = MagicMock()
        gui = GolfSimulationGUI(root_mock)

        assert gui.root == root_mock
        root_mock.title.assert_called()

        # Verify it loads config or uses defaults
        assert hasattr(gui, "colors")
        assert hasattr(gui, "height_var")
