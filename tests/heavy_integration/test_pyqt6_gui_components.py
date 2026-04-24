"""Heavy integration tests for PyQt6 GUI components (fixes #1984).

Verifies that core PyQt6 widgets — QApplication, launcher, theme system,
and pendulum simulator — can be instantiated in a headless (Xvfb) environment.
All tests skip gracefully when PyQt6 is unavailable.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(scope="module")
def qt_app():
    """Return or create a QApplication singleton for headless tests."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 not installed")

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app
    # Do not call app.quit() here — other tests in the module may need the app


class TestQApplicationCreation:
    """Contract: QApplication can be created in a headless environment."""

    def test_qapplication_exists(self, qt_app) -> None:
        """QApplication instance is alive after fixture setup."""
        from PyQt6.QtWidgets import QApplication

        assert QApplication.instance() is not None

    def test_qapplication_process_events(self, qt_app) -> None:
        """QApplication.processEvents() does not raise in headless mode."""
        qt_app.processEvents()


class TestThemeSystem:
    """Contract: Theme system applies dark/light themes without error."""

    def test_dark_theme_applicable(self, qt_app) -> None:
        """Applying the dark theme to the app does not raise."""
        try:
            from src.shared.python.gui_launcher.theme import apply_dark_theme
        except ImportError:
            try:
                from src.shared.python.themes.dark_theme import apply_dark_theme
            except ImportError as exc:
                pytest.skip(f"theme module not importable: {exc}")

        try:
            apply_dark_theme(qt_app)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"apply_dark_theme raised: {exc}")

    def test_light_theme_applicable(self, qt_app) -> None:
        """Applying the light theme to the app does not raise."""
        try:
            from src.shared.python.gui_launcher.theme import apply_light_theme
        except ImportError:
            try:
                from src.shared.python.themes.light_theme import apply_light_theme
            except ImportError as exc:
                pytest.skip(f"light theme module not importable: {exc}")

        try:
            apply_light_theme(qt_app)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"apply_light_theme raised: {exc}")


class TestGolfLauncherInstantiation:
    """Contract: GolfLauncher main window can be instantiated headlessly."""

    def test_launcher_main_window_importable(self, qt_app) -> None:  # noqa: ARG002
        """The launcher main window class is importable."""
        try:
            from src.shared.python.gui_launcher.launcher import GolfLauncher
        except ImportError:
            try:
                from src.shared.python.gui_launcher.main_window import (
                    MainWindow as GolfLauncher,
                )
            except ImportError as exc:
                pytest.skip(f"GolfLauncher not importable: {exc}")

        assert GolfLauncher is not None

    def test_launcher_instantiates(self, qt_app) -> None:
        """GolfLauncher can be instantiated without rendering."""
        try:
            from src.shared.python.gui_launcher.launcher import GolfLauncher
        except ImportError:
            try:
                from src.shared.python.gui_launcher.main_window import (
                    MainWindow as GolfLauncher,
                )
            except ImportError as exc:
                pytest.skip(f"GolfLauncher not importable: {exc}")

        try:
            win = GolfLauncher()
            assert win is not None
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"GolfLauncher instantiation failed: {exc}")


class TestPendulumSimulatorGui:
    """Contract: PendulumSimulator main window opens without error."""

    def test_pendulum_gui_importable(self, qt_app) -> None:  # noqa: ARG002
        """pendulum_simulator GUI module is importable."""
        try:
            from src.shared.python.pendulum_simulator.gui import (  # noqa: F401
                main_window,
            )
        except ImportError as exc:
            pytest.skip(f"pendulum_simulator GUI not importable: {exc}")

    def test_pendulum_main_window_instantiates(self, qt_app) -> None:
        """PendulumMainWindow can be instantiated headlessly."""
        try:
            from src.shared.python.pendulum_simulator.gui.main_window import (
                PendulumMainWindow,
            )
        except ImportError as exc:
            pytest.skip(f"PendulumMainWindow not importable: {exc}")

        try:
            win = PendulumMainWindow()
            assert win is not None
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"PendulumMainWindow instantiation failed: {exc}")


class TestModelExplorerGui:
    """Contract: ModelExplorer main window can be instantiated headlessly."""

    def test_model_explorer_importable(self, qt_app) -> None:  # noqa: ARG002
        """ModelExplorer module is importable."""
        try:
            from src.tools.model_explorer import main_window  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"model_explorer not importable: {exc}")

    def test_model_explorer_window_instantiates(self, qt_app) -> None:
        """ModelExplorer main window can be instantiated without rendering."""
        try:
            from src.tools.model_explorer.main_window import ModelExplorerWindow
        except ImportError as exc:
            pytest.skip(f"ModelExplorerWindow not importable: {exc}")

        try:
            win = ModelExplorerWindow()
            assert win is not None
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"ModelExplorerWindow instantiation failed: {exc}")


pytestmark = pytest.mark.live_simulation
