"""Tests for ``src.tools.golf_simulation_suite.__main__``.

The orchestration module pulls in PyQt6, PyVista, and the heavy
``pyvistaqt.QtInteractor`` (which requires a working OpenGL context). To keep
this suite FAST and headless-portable we patch the GUI dependencies with
``MagicMock`` collaborators and exercise the launcher logic directly. We never
construct a real ``QApplication``.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

MODULE_PATH = "src.tools.golf_simulation_suite.__main__"


def _install_fake_modules() -> dict[str, ModuleType]:
    """Build a mapping of stub modules for the GUI/visualization imports."""

    fakes: dict[str, ModuleType] = {}

    # PyQt6 / PyQt6.QtWidgets
    pyqt6 = ModuleType("PyQt6")
    qt_widgets = ModuleType("PyQt6.QtWidgets")

    class _FakeQMainWindow:
        def __init__(self, *args, **kwargs) -> None:
            self._title: str | None = None
            self._geometry: tuple[int, int, int, int] | None = None
            self._central: object | None = None

        def setWindowTitle(self, title: str) -> None:
            self._title = title

        def setGeometry(self, x: int, y: int, w: int, h: int) -> None:
            self._geometry = (x, y, w, h)

        def setCentralWidget(self, widget: object) -> None:
            self._central = widget

        def show(self) -> None:  # pragma: no cover - trivial
            pass

    class _FakeQWidget:
        def __init__(self, *args, **kwargs) -> None:
            self.layout = None

    class _FakeQVBoxLayout:
        def __init__(self, parent=None) -> None:
            self.parent = parent
            self.widgets: list[object] = []

        def addWidget(self, widget: object) -> None:
            self.widgets.append(widget)

    class _FakeSignal:
        def __init__(self) -> None:
            self.callbacks: list[object] = []

        def connect(self, callback) -> None:
            self.callbacks.append(callback)

        def emit(self) -> None:
            for cb in list(self.callbacks):
                cb()

    class _FakeQPushButton:
        def __init__(self, label: str = "") -> None:
            self.label = label
            self.clicked = _FakeSignal()

    class _FakeQApplication:
        _instance: _FakeQApplication | None = None
        last_argv: list[str] | None = None
        exec_calls: int = 0

        def __init__(self, argv) -> None:
            type(self).last_argv = list(argv)
            type(self)._instance = self

        @classmethod
        def instance(cls) -> _FakeQApplication | None:
            return cls._instance

        def exec(self) -> int:
            type(self).exec_calls += 1
            return 0

    qt_widgets.QApplication = _FakeQApplication
    qt_widgets.QMainWindow = _FakeQMainWindow
    qt_widgets.QVBoxLayout = _FakeQVBoxLayout
    qt_widgets.QWidget = _FakeQWidget
    qt_widgets.QPushButton = _FakeQPushButton
    pyqt6.QtWidgets = qt_widgets

    fakes["PyQt6"] = pyqt6
    fakes["PyQt6.QtWidgets"] = qt_widgets

    # pyvista stub
    pyvista = ModuleType("pyvista")
    pyvista.Sphere = MagicMock(name="Sphere", return_value="sphere-mesh")
    pyvista.MultipleLines = MagicMock(name="MultipleLines", return_value="lines-mesh")
    pyvista.Plane = MagicMock(name="Plane", return_value="plane-mesh")
    pyvista.Cylinder = MagicMock(name="Cylinder", return_value="cylinder-mesh")
    fakes["pyvista"] = pyvista

    # pyvistaqt stub
    pyvistaqt = ModuleType("pyvistaqt")

    class _FakeQtInteractor:
        def __init__(self, parent=None) -> None:
            self.parent = parent
            self.interactor = MagicMock(name="interactor")
            self.clear = MagicMock(name="clear")
            self.add_mesh = MagicMock(name="add_mesh")
            self.add_axes = MagicMock(name="add_axes")
            self.reset_camera = MagicMock(name="reset_camera")

    pyvistaqt.QtInteractor = _FakeQtInteractor
    fakes["pyvistaqt"] = pyvistaqt

    # Heavy physics import — stub the simulator
    shared_pkg = ModuleType("src.shared.python.physics.ball_enhanced_simulator")
    shared_pkg.EnhancedBallFlightSimulator = MagicMock(
        name="EnhancedBallFlightSimulator",
        return_value=MagicMock(name="ball-sim-instance"),
    )
    fakes["src.shared.python.physics.ball_enhanced_simulator"] = shared_pkg

    return fakes


@pytest.fixture
def patched_module():
    """Reload the orchestration module with stubbed GUI/physics deps."""

    fakes = _install_fake_modules()
    # Drop any previously loaded copy so the patched imports take effect.
    sys.modules.pop(MODULE_PATH, None)
    with patch.dict(sys.modules, fakes):
        module = importlib.import_module(MODULE_PATH)
        try:
            yield module
        finally:
            sys.modules.pop(MODULE_PATH, None)


def test_module_exposes_expected_public_api(patched_module) -> None:
    assert hasattr(patched_module, "GolfSimulationWindow")
    assert hasattr(patched_module, "get_dockable_ui")
    assert hasattr(patched_module, "main")


def test_window_initialization_sets_title_and_geometry(patched_module) -> None:
    window = patched_module.GolfSimulationWindow()

    assert window._title == "Golf Simulation Suite"
    assert window._geometry == (100, 100, 800, 600)
    assert window._central is not None


def test_window_creates_two_action_buttons_with_signals(patched_module) -> None:
    window = patched_module.GolfSimulationWindow()

    assert window.btn_simulate.label == "Simulate Ball Flight"
    assert window.btn_putting.label == "Putting Green Mode"
    # Each button has exactly one connected handler.
    assert len(window.btn_simulate.clicked.callbacks) == 1
    assert len(window.btn_putting.clicked.callbacks) == 1


def test_window_initializes_ball_simulator(patched_module) -> None:
    sim_factory = sys.modules[
        "src.shared.python.physics.ball_enhanced_simulator"
    ].EnhancedBallFlightSimulator
    sim_factory.reset_mock()

    window = patched_module.GolfSimulationWindow()

    sim_factory.assert_called_once_with()
    assert window.ball_sim is sim_factory.return_value


def test_run_simulation_clears_plotter_and_draws_ball_trajectory(
    patched_module,
) -> None:
    window = patched_module.GolfSimulationWindow()

    window.run_simulation()

    plotter = window.plotter
    plotter.clear.assert_called_once_with()
    plotter.add_axes.assert_called_once_with()
    plotter.reset_camera.assert_called_once_with()
    # Two meshes added: the sphere and the trajectory polyline.
    assert plotter.add_mesh.call_count == 2
    mesh_args = [call.args[0] for call in plotter.add_mesh.call_args_list]
    assert "sphere-mesh" in mesh_args
    assert "lines-mesh" in mesh_args


def test_run_simulation_uses_expected_geometry_params(patched_module) -> None:
    pv = sys.modules["pyvista"]
    pv.Sphere.reset_mock()
    pv.MultipleLines.reset_mock()
    window = patched_module.GolfSimulationWindow()

    window.run_simulation()

    pv.Sphere.assert_called_once_with(radius=0.02, center=(0, 0, 0))
    pv.MultipleLines.assert_called_once()
    points = pv.MultipleLines.call_args.kwargs["points"]
    assert points == [[0, 0, 0], [50, 0, 20], [100, 0, 0]]


def test_run_putting_green_draws_plane_hole_and_ball(patched_module) -> None:
    pv = sys.modules["pyvista"]
    pv.Plane.reset_mock()
    pv.Cylinder.reset_mock()
    pv.Sphere.reset_mock()
    window = patched_module.GolfSimulationWindow()

    window.run_putting_green()

    plotter = window.plotter
    plotter.clear.assert_called_once_with()
    plotter.add_axes.assert_called_once_with()
    plotter.reset_camera.assert_called_once_with()
    assert plotter.add_mesh.call_count == 3
    pv.Plane.assert_called_once_with(
        center=(0, 0, 0), direction=(0, 0, 1), i_size=10, j_size=10
    )
    pv.Cylinder.assert_called_once_with(
        center=(3, 3, -0.05), direction=(0, 0, 1), radius=0.05, height=0.1
    )
    pv.Sphere.assert_called_once_with(radius=0.02, center=(-3, -3, 0.02))


def test_button_click_triggers_simulation(patched_module) -> None:
    window = patched_module.GolfSimulationWindow()

    window.btn_simulate.clicked.emit()

    assert window.plotter.clear.call_count == 1
    assert window.plotter.add_mesh.call_count == 2


def test_button_click_triggers_putting_green(patched_module) -> None:
    window = patched_module.GolfSimulationWindow()

    window.btn_putting.clicked.emit()

    assert window.plotter.clear.call_count == 1
    assert window.plotter.add_mesh.call_count == 3


def test_get_dockable_ui_returns_fresh_window(patched_module) -> None:
    window_a = patched_module.get_dockable_ui()
    window_b = patched_module.get_dockable_ui()

    assert isinstance(window_a, patched_module.GolfSimulationWindow)
    assert isinstance(window_b, patched_module.GolfSimulationWindow)
    assert window_a is not window_b


def test_main_reuses_existing_qapplication_instance(patched_module) -> None:
    qapp_cls = sys.modules["PyQt6.QtWidgets"].QApplication
    qapp_cls._instance = qapp_cls.__new__(qapp_cls)
    qapp_cls._instance.exec_calls = 0
    qapp_cls.exec_calls = 0
    existing = qapp_cls._instance
    qapp_cls.last_argv = None

    with patch.object(patched_module.sys, "exit") as mock_exit:
        patched_module.main()

    # Existing instance reused, so argv was not re-read.
    assert qapp_cls.last_argv is None
    assert qapp_cls.instance() is existing
    mock_exit.assert_called_once_with(0)


def test_main_creates_qapplication_when_none_exists(patched_module) -> None:
    qapp_cls = sys.modules["PyQt6.QtWidgets"].QApplication
    qapp_cls._instance = None
    qapp_cls.last_argv = None
    qapp_cls.exec_calls = 0

    with patch.object(patched_module.sys, "exit") as mock_exit:
        patched_module.main()

    assert qapp_cls.last_argv is not None
    mock_exit.assert_called_once_with(0)


def test_main_configures_logging_at_info_level(patched_module) -> None:
    with (
        patch.object(patched_module.logging, "basicConfig") as mock_cfg,
        patch.object(patched_module.sys, "exit"),
    ):
        patched_module.main()

    mock_cfg.assert_called_once()
    assert mock_cfg.call_args.kwargs["level"] == patched_module.logging.INFO


def test_run_simulation_logs_info_message(patched_module, caplog) -> None:
    window = patched_module.GolfSimulationWindow()
    with caplog.at_level("INFO", logger=patched_module.logger.name):
        window.run_simulation()

    assert any("golf ball simulation" in r.message for r in caplog.records)


def test_run_putting_green_logs_info_message(patched_module, caplog) -> None:
    window = patched_module.GolfSimulationWindow()
    with caplog.at_level("INFO", logger=patched_module.logger.name):
        window.run_putting_green()

    assert any("putting green" in r.message for r in caplog.records)
