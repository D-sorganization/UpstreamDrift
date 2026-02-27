"""Core launcher implementation for GUI applications."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from importlib.util import find_spec
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GUIType(Enum):
    """Supported GUI framework types."""

    PYQT6 = "pyqt6"
    REACT = "react"
    TKINTER = "tkinter"
    BROWSER = "browser"


@dataclass
class LaunchConfig:
    """Configuration for launching a GUI application.

    Fields for subprocess-based launch:
        module_path, entry_point, web_path, working_dir, port, env_vars,
        auto_open_browser

    Fields for in-process PyQt6 launch (used by launch_pyqt6_app):
        class_name, title, settings_app, min_size, organization
    """

    tool_name: str
    gui_type: GUIType
    module_path: str | None = None
    class_name: str | None = None
    web_path: str | None = None
    entry_point: str | None = None
    dependencies: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    working_dir: str | None = None
    port: int = 3000
    auto_open_browser: bool = True
    title: str | None = None
    settings_app: str | None = None
    min_size: tuple[int, int] | None = None
    organization: str = "D-sorganization"


@dataclass
class DependencyStatus:
    """Result of dependency check."""

    ok: bool
    missing: list[str]
    guidance: dict[str, str]


def check_python_dependencies(
    packages: list[str],
    spec_finder: Callable[[str], object] = find_spec,
) -> DependencyStatus:
    """Check if required Python packages are available.

    Args:
        packages: List of package names to check
        spec_finder: Function to check module availability (for testing)

    Returns:
        DependencyStatus with results
    """
    install_hints = {
        "PyQt6": "pip install PyQt6",
        "numpy": "pip install numpy",
        "pandas": "pip install pandas",
        "matplotlib": "pip install matplotlib",
        "scipy": "pip install scipy",
    }

    missing = [pkg for pkg in packages if spec_finder(pkg) is None]

    return DependencyStatus(
        ok=not missing,
        missing=missing,
        guidance={pkg: install_hints.get(pkg, f"pip install {pkg}") for pkg in missing},
    )


def check_node_dependencies(web_path: Path) -> DependencyStatus:
    """Check if Node.js dependencies are installed.

    Args:
        web_path: Path to the web project directory

    Returns:
        DependencyStatus with results
    """
    node_modules = web_path / "node_modules"
    package_json = web_path / "package.json"

    missing = []
    guidance = {}

    if not package_json.exists():
        missing.append("package.json")
        guidance["package.json"] = "Web project not properly configured"

    if not node_modules.exists():
        missing.append("node_modules")
        guidance["node_modules"] = f"Run: cd {web_path} && npm install"

    return DependencyStatus(
        ok=not missing,
        missing=missing,
        guidance=guidance,
    )


class GUILauncher:
    """Unified launcher for PyQt6 and React GUI applications."""

    def __init__(
        self,
        config: LaunchConfig | None = None,
        *,
        tool_name: str = "",
        gui_type: GUIType = GUIType.PYQT6,
        **kwargs: Any,
    ) -> None:
        """Initialize the launcher.

        Args:
            config: Full launch configuration
            tool_name: Name of the tool (if not using config)
            gui_type: Type of GUI to launch (if not using config)
            **kwargs: Additional configuration options
        """
        if config:
            self.config = config
        else:
            self.config = LaunchConfig(
                tool_name=tool_name,
                gui_type=gui_type,
                **kwargs,
            )

        self._process: subprocess.Popen | None = None

    def check_dependencies(self) -> DependencyStatus:
        """Check all required dependencies.

        Returns:
            DependencyStatus indicating what's missing
        """
        if self.config.gui_type == GUIType.PYQT6:
            base_deps = ["PyQt6"] + self.config.dependencies
            return check_python_dependencies(base_deps)

        if self.config.gui_type == GUIType.REACT:
            if self.config.web_path:
                web_path = Path(self.config.web_path)
                return check_node_dependencies(web_path)
            return DependencyStatus(ok=False, missing=["web_path"], guidance={})

        if self.config.gui_type == GUIType.TKINTER:
            return check_python_dependencies(["tkinter"] + self.config.dependencies)

        return DependencyStatus(ok=True, missing=[], guidance={})

    def launch(self) -> int:
        """Launch the GUI application.

        Returns:
            Exit code (0 for success)
        """
        status = self.check_dependencies()
        if not status.ok:
            self._print_missing_deps(status)
            return 1

        try:
            if self.config.gui_type == GUIType.PYQT6:
                return self._launch_pyqt6()
            if self.config.gui_type == GUIType.REACT:
                return self._launch_react()
            if self.config.gui_type == GUIType.TKINTER:
                return self._launch_tkinter()
            # GUIType.BROWSER
            return self._launch_browser()
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to launch {self.config.tool_name}: {e}")
            return 1

    def _launch_pyqt6(self) -> int:
        """Launch a PyQt6 application."""
        if self.config.module_path:
            cmd = [sys.executable, "-m", self.config.module_path]
        elif self.config.entry_point:
            cmd = [sys.executable, self.config.entry_point]
        else:
            logger.error("No module_path or entry_point specified for PyQt6 app")
            return 1

        env = os.environ.copy()
        env.update(self.config.env_vars)

        logger.info(f"Launching PyQt6 app: {self.config.tool_name}")
        return subprocess.call(cmd, env=env)

    def _launch_react(self) -> int:
        """Launch a React/Vite web application."""
        if not self.config.web_path:
            logger.error("No web_path specified for React app")
            return 1

        web_path = Path(self.config.web_path)

        # Find the npm executable (npm.cmd on Windows, npm elsewhere)
        npm_path = shutil.which("npm")
        if npm_path is None:
            logger.error("npm is not installed or not in PATH")
            return 1

        # Verify npm works
        try:
            subprocess.run([npm_path, "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("npm is not installed or not in PATH")
            return 1

        logger.info(f"Starting React dev server for: {self.config.tool_name}")
        logger.info(f"  Directory: {web_path}")
        logger.info(f"  Port: {self.config.port}")

        env = os.environ.copy()
        env.update(self.config.env_vars)
        env["PORT"] = str(self.config.port)

        # Start the dev server without shell=True (security hardening).
        # Using the resolved npm_path avoids the need for shell resolution.
        cmd = [npm_path, "run", "dev"]

        self._process = subprocess.Popen(
            cmd,
            cwd=web_path,
            env=env,
        )

        if self.config.auto_open_browser:
            import time

            time.sleep(2)  # Give the server time to start
            webbrowser.open(f"http://localhost:{self.config.port}")

        try:
            return self._process.wait()
        except KeyboardInterrupt:
            self._process.terminate()
            return 0

    def _launch_tkinter(self) -> int:
        """Launch a Tkinter application."""
        if self.config.entry_point:
            cmd = [sys.executable, self.config.entry_point]
        elif self.config.module_path:
            cmd = [sys.executable, "-m", self.config.module_path]
        else:
            logger.error("No entry_point or module_path specified for Tkinter app")
            return 1

        logger.info(f"Launching Tkinter app: {self.config.tool_name}")
        return subprocess.call(cmd)

    def _launch_browser(self) -> int:
        """Launch a static HTML file in the browser."""
        if not self.config.web_path:
            logger.error("No web_path specified for browser app")
            return 1

        file_path = Path(self.config.web_path)
        if not file_path.exists():
            logger.error(f"HTML file not found: {file_path}")
            return 1

        logger.info(f"Opening in browser: {file_path}")
        webbrowser.open(f"file://{file_path.absolute()}")
        return 0

    def _print_missing_deps(self, status: DependencyStatus) -> None:
        """Print missing dependency information."""
        logger.info("Missing dependencies detected:")
        for pkg in status.missing:
            hint = status.guidance.get(pkg, "")
            logger.info(f"  - {pkg}: {hint}")
        logger.info("\nInstall the missing packages and try again.")

    def stop(self) -> None:
        """Stop the running application (for React dev servers)."""
        if self._process:
            self._process.terminate()
            self._process = None


def create_launcher(
    tool_name: str,
    gui_type: GUIType,
    **kwargs: Any,
) -> GUILauncher:
    """Factory function to create a launcher with common configuration.

    Args:
        tool_name: Name of the tool
        gui_type: Type of GUI framework
        **kwargs: Additional configuration options

    Returns:
        Configured GUILauncher instance
    """
    config = LaunchConfig(tool_name=tool_name, gui_type=gui_type, **kwargs)
    return GUILauncher(config=config)


def launch_pyqt6_app(config: LaunchConfig) -> int:
    """Launch a PyQt6 application in-process with theme support.

    This is the consolidated launcher that eliminates boilerplate from
    individual launch_pyqt6.py scripts. It handles:
    - Dependency checking
    - QApplication creation and configuration
    - Dynamic import of the window/widget class
    - Theme system setup via setup_themed_app()
    - Window display and event loop

    Args:
        config: LaunchConfig with at least module_path and class_name set.
            Optional: title, settings_app, min_size, dependencies.

    Returns:
        Application exit code (0 for success, 1 for error).
    """
    # Check dependencies first (before importing PyQt6)
    all_deps = list(config.dependencies)
    if "PyQt6" not in all_deps:
        all_deps.insert(0, "PyQt6")
    status = check_python_dependencies(all_deps)
    if not status.ok:
        logger.info("Missing required packages:")
        for pkg in status.missing:
            hint = status.guidance.get(pkg, f"pip install {pkg}")
            logger.info(f"  - {pkg}: {hint}")
        logger.info("\nInstall the missing packages and try again.")
        return 1

    if not config.module_path or not config.class_name:
        logger.error(
            "launch_pyqt6_app requires both module_path and class_name in config"
        )
        return 1

    try:
        import importlib

        from PyQt6.QtWidgets import QApplication, QMainWindow

        # Dynamically import the window/widget class
        module = importlib.import_module(config.module_path)
        window_class = getattr(module, config.class_name)

        # Create the application
        app = QApplication(sys.argv)
        display_name = config.title or config.tool_name
        app.setApplicationName(display_name)
        app.setOrganizationName(config.organization)

        # Create the window
        window_obj = window_class()

        # If the class is a QMainWindow, use it directly.
        # If it's a QWidget, wrap it in a QMainWindow.
        if isinstance(window_obj, QMainWindow):
            window = window_obj
        else:
            window = QMainWindow()
            window.setCentralWidget(window_obj)

        window.setWindowTitle(display_name)
        if config.min_size:
            window.setMinimumSize(*config.min_size)

        # Apply theme system
        try:
            from shared.python.theme import setup_themed_app

            settings_app = config.settings_app or config.tool_name.replace(
                " ", ""
            ).replace("_", "")
            setup_themed_app(app, window, settings_app=settings_app)
        except ImportError:
            logger.warning("Theme system not available, launching without theme")

        window.show()
        return app.exec()

    except ImportError as e:
        logger.error("Failed to import GUI components: %s", e)
        logger.error(f"Error importing GUI components: {e}")
        logger.info("\nMake sure the package is installed correctly.")
        return 1


def launch_from_gui_info(gui_info: dict[str, Any]) -> int:
    """Launch a PyQt6 app directly from a GUI_INFO dict.

    This is the simplest way to launch a tool from its gui_registration.py.
    Each launch_pyqt6.py script can be reduced to::

        from my_tool.gui_registration import GUI_INFO
        from gui_launcher import launch_from_gui_info
        sys.exit(launch_from_gui_info(GUI_INFO))

    Args:
        gui_info: The GUI_INFO dictionary from gui_registration.py

    Returns:
        Application exit code.
    """
    pyqt6 = gui_info.get("pyqt6")
    if not pyqt6:
        logger.info(
            f"No pyqt6 config found in GUI_INFO for {gui_info.get('name', '?')}"
        )
        return 1

    min_size = pyqt6.get("min_size")
    config = LaunchConfig(
        tool_name=gui_info.get("tool_name", ""),
        gui_type=GUIType.PYQT6,
        module_path=pyqt6.get("module"),
        class_name=pyqt6.get("class"),
        dependencies=pyqt6.get("dependencies", []),
        title=gui_info.get("name"),
        settings_app=pyqt6.get("settings_app"),
        min_size=tuple(min_size) if min_size else None,
    )
    return launch_pyqt6_app(config)


def launch_web_app(
    tool_name: str,
    web_dir: Path,
    port: int = 5173,
    auto_open_browser: bool = True,
    npm_args: list[str] | None = None,
) -> int:
    """Launch a React/Vite web application dev server.

    This is the consolidated web launcher that eliminates boilerplate from
    individual launch_web.py scripts. It handles:
    - Node.js and npm availability checks
    - npm dependency installation
    - Dev server startup
    - Optional browser opening

    Args:
        tool_name: Human-readable name for log messages.
        web_dir: Path to the web project directory (containing package.json).
        port: Port number for the dev server.
        auto_open_browser: Whether to open the browser automatically.
        npm_args: Additional arguments to pass to ``npm run dev``.

    Returns:
        Process exit code (0 for success, 1 for error).
    """
    # Check Node.js / npm
    for cmd_name in ("node", "npm"):
        try:
            subprocess.run(
                [cmd_name, "--version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(f"Error: {cmd_name} is not installed or not in PATH")
            if cmd_name == "node":
                logger.info("Install Node.js from https://nodejs.org/")
            return 1

    if not web_dir.exists():
        logger.error(f"Error: Web directory not found at {web_dir}")
        return 1

    # Install dependencies if needed
    if not (web_dir / "node_modules").exists():
        logger.info("Installing dependencies...")
        install_result = subprocess.run(
            ["npm", "install"],
            cwd=str(web_dir),
            shell=False,
        )
        if install_result.returncode != 0:
            logger.error("Error: Failed to install npm dependencies")
            return 1

    # Build dev server command
    dev_cmd: list[str] = ["npm", "run", "dev"]
    if npm_args:
        dev_cmd.extend(npm_args)

    env = os.environ.copy()
    env["PORT"] = str(port)

    logger.info(f"Starting {tool_name} web application on http://localhost:{port}")

    process = subprocess.Popen(
        dev_cmd,
        cwd=str(web_dir),
        env=env,
        shell=False,
    )

    if auto_open_browser:
        import threading
        import time

        def _open_browser() -> None:
            time.sleep(2)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=_open_browser, daemon=True).start()

    try:
        return process.wait()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        process.terminate()
        return 0


def launch_web_from_gui_info(gui_info: dict[str, Any], caller_file: str) -> int:
    """Launch a React web app from a GUI_INFO dict.

    Each launch_web.py script can be reduced to::

        from my_tool.gui_registration import GUI_INFO
        from gui_launcher import launch_web_from_gui_info
        sys.exit(launch_web_from_gui_info(GUI_INFO, __file__))

    Args:
        gui_info: The GUI_INFO dictionary from gui_registration.py.
        caller_file: ``__file__`` of the calling launch_web.py script,
            used to locate the ``web/`` directory relative to the caller.

    Returns:
        Application exit code.
    """
    web_cfg = gui_info.get("web", {})
    tool_name = gui_info.get("name", gui_info.get("tool_name", "Unknown"))

    web_dir = Path(caller_file).parent / "web"
    port = web_cfg.get("port", 5173)
    auto_open = web_cfg.get("auto_open_browser", True)

    return launch_web_app(
        tool_name=tool_name,
        web_dir=web_dir,
        port=port,
        auto_open_browser=auto_open,
    )


def launch_tool_by_name(tool_name: str) -> int:
    """Launch a tool by its registered name.

    Looks up the tool in the global GUI registry and launches the
    PyQt6 variant using launch_pyqt6_app().

    Args:
        tool_name: The tool_name used during registration.

    Returns:
        Application exit code.
    """
    from .registry import get_registry

    registry = get_registry()
    registration = registry.get(tool_name)
    if registration is None:
        logger.info(f"Tool '{tool_name}' not found in registry.")
        available = registry.list_tools()
        if available:
            logger.info("\nAvailable tools:")
            for reg in available:
                logger.info(f"  - {reg.tool_name} ({reg.display_name})")
        return 1

    config = registration.gui_configs.get(GUIType.PYQT6)
    if config is None:
        logger.info(f"Tool '{tool_name}' has no PyQt6 configuration.")
        return 1

    return launch_pyqt6_app(config)


def make_pyqt6_launcher(gui_info_module: str) -> int:
    """Generic launcher factory that eliminates duplicate launch_pyqt6.py files.

    Instead of every tool having its own 17-line launch_pyqt6.py that only
    differs in the GUI_INFO import path, each tool can now use a thin wrapper::

        #!/usr/bin/env python3
        from _bootstrap import bootstrap
        bootstrap(__file__)
        import sys
        from gui_launcher import make_pyqt6_launcher
        if __name__ == "__main__":
            sys.exit(make_pyqt6_launcher("my_tool.gui_registration"))

    Args:
        gui_info_module: Dotted module path to the gui_registration module
            that contains a ``GUI_INFO`` dict
            (e.g. ``"acid_gas_dewpoint.gui_registration"``).

    Returns:
        Application exit code (0 for success, 1 for error).
    """
    import importlib

    try:
        mod = importlib.import_module(gui_info_module)
    except ImportError as exc:
        logger.error(
            "Failed to import GUI registration module %r: %s",
            gui_info_module,
            exc,
        )
        return 1

    gui_info = getattr(mod, "GUI_INFO", None)
    if gui_info is None:
        logger.error("Module %r does not define a GUI_INFO dict", gui_info_module)
        return 1

    return launch_from_gui_info(gui_info)
