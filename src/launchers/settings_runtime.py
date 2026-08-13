"""Runtime dependency checks and WSL setup UI for launcher settings."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_RUNTIME_DEPENDENCIES = {
    "numpy": ">=1.26.4",
    "scipy": ">=1.13.1",
    "mujoco": ">=3.6.0",
    "pydrake": ">=1.22.0",
    "pinocchio": ">=2.6.0",
    "opensim": ">=4.4.0",
    "myosuite": ">=2.0.0",
}
_RUNTIME_DISPLAY_NAMES = {
    "numpy": "NumPy",
    "scipy": "SciPy",
    "mujoco": "MuJoCo",
    "pydrake": "Drake (PyDrake)",
    "pinocchio": "Pinocchio",
    "opensim": "OpenSim",
    "myosuite": "MyoSuite",
}
_RUNTIME_OPTIONAL_DEPS = {"pydrake", "pinocchio", "opensim", "myosuite"}
_DEPENDENCY_PROBE_SCRIPT = (
    "import importlib.metadata\n"
    "import sys\n"
    "res = []\n"
    f"for p in {list(_RUNTIME_DEPENDENCIES)!r}:\n"
    "    try:\n"
    "        __import__(p)\n"
    "        try:\n"
    "            v = importlib.metadata.version(p)\n"
    "        except Exception:\n"
    "            v = getattr(sys.modules[p], '__version__', 'Unknown')\n"
    "        res.append(f'{p}:{v}')\n"
    "    except ImportError:\n"
    "        res.append(f'{p}:Missing')\n"
    "print(','.join(res))"
)


@dataclass(frozen=True)
class RuntimeDependencyReport:
    """Structured result from a background runtime dependency check."""

    dialog_title: str
    table_title: str
    environment_name: str
    check_results: list[dict[str, Any]]


class RuntimeDependencyCheckFailure(RuntimeError):
    """A dependency check completed with a user-facing warning or error."""

    def __init__(self, severity: str, title: str, html: str) -> None:
        super().__init__(html)
        self.severity = severity
        self.title = title
        self.html = html


class RuntimeDependencyCheckWorker(QThread):
    """Run Docker or WSL dependency checks without blocking the GUI thread."""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str, str)

    def __init__(
        self,
        check_fn: Callable[[], RuntimeDependencyReport],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._check_fn = check_fn

    def run(self) -> None:
        try:
            self.succeeded.emit(self._check_fn())
        except RuntimeDependencyCheckFailure as exc:
            self.failed.emit(exc.severity, exc.title, exc.html)
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
            self.failed.emit(
                "critical",
                "Dependency Check",
                f"<h3>Runtime Dependency Check Failed</h3><p>{exc}</p>",
            )


def _numeric_version(version: str) -> list[int]:
    """Return the leading numeric component of each dotted version part."""
    parts: list[int] = []
    for component in version.split("."):
        digits: list[str] = []
        for character in component:
            if not character.isdigit():
                break
            digits.append(character)
        parts.append(int("".join(digits)) if digits else 0)
    return parts


def compare_version_strings(installed: str, required_spec: str) -> bool:
    """Return whether an installed version satisfies a simple >= or == spec."""
    if installed == "Unknown":
        return True
    if installed == "Missing":
        return False

    operator = ">="
    required = required_spec
    if required_spec.startswith((">=", "==")):
        operator = required_spec[:2]
        required = required_spec[2:]

    try:
        installed_parts = _numeric_version(installed)
        required_parts = _numeric_version(required)
        width = max(len(installed_parts), len(required_parts))
        installed_parts.extend([0] * (width - len(installed_parts)))
        required_parts.extend([0] * (width - len(required_parts)))
        if operator == "==":
            return installed_parts == required_parts
        return installed_parts >= required_parts
    except (ValueError, TypeError, IndexError):
        return True


def parse_dependency_output(output: str) -> dict[str, str]:
    """Parse the dependency probe's comma-separated package/version output."""
    parsed: dict[str, str] = {}
    for item in output.split(","):
        if ":" in item:
            key, value = item.split(":", 1)
            parsed[key] = value
    return parsed


def build_dependency_check_results(
    parsed_dependencies: dict[str, str],
) -> list[dict[str, Any]]:
    """Build display records for every required and optional runtime package."""
    results: list[dict[str, Any]] = []
    for package_name, requirement in _RUNTIME_DEPENDENCIES.items():
        installed = parsed_dependencies.get(package_name, "Missing")
        if installed == "Missing":
            status = "warn" if package_name in _RUNTIME_OPTIONAL_DEPS else "error"
        else:
            status = (
                "ok" if compare_version_strings(installed, requirement) else "error"
            )
        results.append(
            {
                "name": _RUNTIME_DISPLAY_NAMES[package_name],
                "required": requirement,
                "installed": installed,
                "status": status,
            }
        )
    return results


def check_docker_dependencies_report() -> RuntimeDependencyReport:
    """Inspect the engine image and probe its installed Python dependencies."""
    from src.launchers.docker_manager import get_docker_cmd

    docker_command = get_docker_cmd()
    try:
        inspection = subprocess.run(
            docker_command + ["image", "inspect", "upstream-drift:engine"],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except subprocess.TimeoutExpired as exc:
        html = (
            "<h3>Docker Connection Timeout</h3>"
            "<p>Failed to communicate with the Docker daemon within the timeout limit (10.0s):</p>"
            f"<pre style='color:#f87171;'>{exc}</pre>"
            "<p>This typically occurs if the WSL2 subsystem or Docker Desktop is starting up, frozen, or not running.</p>"
            "<p>Please verify that:</p><ol>"
            "<li><b>Docker Desktop</b> is running.</li>"
            "<li><b>WSL integration</b> is enabled for your distribution in Docker Desktop settings.</li>"
            "<li>You can run <code>wsl docker ps</code> in a terminal without hanging.</li>"
            "</ol>"
        )
        raise RuntimeDependencyCheckFailure(
            "critical", "Docker Dependency Check", html
        ) from exc

    if inspection.returncode != 0:
        html = (
            "<h3>Docker Environment Status: Missing Image</h3>"
            "<p>The <b>upstream-drift:engine</b> image has not been built yet.</p>"
            "<p>Please click the <b>Build Image</b> button below to build the container first.</p>"
        )
        raise RuntimeDependencyCheckFailure("warning", "Docker Dependency Check", html)

    try:
        probe = subprocess.run(
            docker_command
            + [
                "run",
                "--rm",
                "upstream-drift:engine",
                "python3",
                "-c",
                _DEPENDENCY_PROBE_SCRIPT,
            ],
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except subprocess.TimeoutExpired as exc:
        html = (
            "<h3>Docker Environment Status: Timeout</h3>"
            "<p>Docker container dependency check timed out.</p>"
            "<p>The Docker image exists, but a cold container start did not finish within 60 seconds.</p>"
        )
        raise RuntimeDependencyCheckFailure(
            "warning", "Docker Dependency Check", html
        ) from exc

    if probe.returncode != 0:
        html = (
            "<h3>Docker Environment Status: Degraded</h3>"
            f"<p>The image is built, but running probe failed (Exit Code {probe.returncode}).</p>"
            f"<pre style='color:#f87171;'>{probe.stderr.strip()}</pre>"
            "<p>Verify Docker is running and has access to execute containers in WSL.</p>"
        )
        raise RuntimeDependencyCheckFailure("warning", "Docker Dependency Check", html)

    return RuntimeDependencyReport(
        dialog_title="Docker Dependency Check",
        table_title="Docker Container Environment",
        environment_name="Docker Container",
        check_results=build_dependency_check_results(
            parse_dependency_output(probe.stdout.strip())
        ),
    )


def check_wsl_dependencies_report() -> RuntimeDependencyReport:
    """Probe dependencies in the repository's WSL virtual environment."""
    from src.shared.python.data_io.path_utils import get_repo_root

    virtual_environment = get_repo_root() / ".venv-wsl"
    python_executable = (
        "./.venv-wsl/bin/python" if virtual_environment.exists() else "python3"
    )
    probe = subprocess.run(
        ["wsl", python_executable, "-c", _DEPENDENCY_PROBE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if probe.returncode != 0:
        html = (
            "<h3>WSL Environment Status: Not Set Up</h3>"
            "<p>WSL environment could not be checked or has not been initialized.</p>"
            f"<p>Details: Exit Code {probe.returncode}</p>"
            f"<pre style='color:#f87171;'>{probe.stderr.strip()}</pre>"
            "<p>Please click the <b>WSL Setup</b> button to view the setup script and run it in WSL.</p>"
        )
        raise RuntimeDependencyCheckFailure("warning", "WSL Dependency Check", html)

    return RuntimeDependencyReport(
        dialog_title="WSL Dependency Check",
        table_title="WSL2 Environment Status",
        environment_name=f"WSL ({python_executable})",
        check_results=build_dependency_check_results(
            parse_dependency_output(probe.stdout.strip())
        ),
    )


class WslScriptDialog(QDialog):
    """Inspect, copy, and launch the WSL dependency installation script."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("WSL Dependency Setup Script")
        self.resize(700, 550)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        description = QLabel(
            "<h3>WSL2 Dependency Setup</h3>"
            "<p>This script installs all system and Python dependencies needed to run "
            "UpstreamDrift physics engines (including Drake, Pinocchio, MuJoCo, and OpenSim) inside WSL2 Ubuntu.</p>"
        )
        description.setWordWrap(True)
        description.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(description)
        layout.addWidget(self._create_commands_group())

        self.btn_run_terminal = QPushButton(
            "Run script in interactive WSL Terminal window"
        )
        self.btn_run_terminal.setToolTip(
            "Open a new Windows Command Prompt that runs the script inside WSL. "
            "Recommended if sudo password prompt is required."
        )
        self.btn_run_terminal.clicked.connect(self._run_in_terminal)
        self.btn_run_terminal.setStyleSheet(
            "QPushButton { background-color: #0366d6; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 4px; } "
            "QPushButton:hover { background-color: #0056b3; }"
        )
        layout.addWidget(self.btn_run_terminal)
        layout.addWidget(self._create_content_group())

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _create_commands_group(self) -> QGroupBox:
        group = QGroupBox("Run Commands")
        layout = QVBoxLayout(group)
        self.lbl_wsl_cmd = QLabel(
            "<code>bash scripts/install_wsl_dependencies.sh</code>"
        )
        self.lbl_win_cmd = QLabel(
            "<code>wsl bash scripts/install_wsl_dependencies.sh</code>"
        )
        layout.addLayout(
            self._create_command_row(
                "Inside WSL:",
                self.lbl_wsl_cmd,
                "bash scripts/install_wsl_dependencies.sh",
                "WSL shell command",
            )
        )
        layout.addLayout(
            self._create_command_row(
                "From Windows:",
                self.lbl_win_cmd,
                "wsl bash scripts/install_wsl_dependencies.sh",
                "Windows run command",
            )
        )
        return group

    def _create_command_row(
        self, label: str, command_label: QLabel, command: str, name: str
    ) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(f"<b>{label}</b>"))
        command_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        row.addWidget(command_label)
        row.addStretch()
        copy_button = QPushButton("Copy")
        copy_button.setFixedWidth(60)
        copy_button.clicked.connect(lambda: self._copy_text(command, name))
        row.addWidget(copy_button)
        return row

    def _create_content_group(self) -> QGroupBox:
        group = QGroupBox("Script Contents (scripts/install_wsl_dependencies.sh)")
        layout = QVBoxLayout(group)
        self.txt_content = QTextEdit()
        self.txt_content.setReadOnly(True)
        self.txt_content.setStyleSheet(
            "font-family: 'Courier New', monospace; background-color: #1e1e1e; color: #d4d4d4;"
        )
        self.txt_content.setPlainText(self._load_script_content())
        layout.addWidget(self.txt_content)
        button_row = QHBoxLayout()
        copy_button = QPushButton("Copy Script Contents")
        copy_button.clicked.connect(
            lambda: self._copy_text(self.txt_content.toPlainText(), "Script contents")
        )
        button_row.addWidget(copy_button)
        button_row.addStretch()
        layout.addLayout(button_row)
        return group

    def _load_script_content(self) -> str:
        from src.shared.python.data_io.path_utils import get_repo_root

        script_path = get_repo_root() / "scripts" / "install_wsl_dependencies.sh"
        if script_path.exists():
            try:
                return script_path.read_text(encoding="utf-8")
            except OSError as exc:
                return f"# Error loading script contents: {exc}"
        return "# Error: scripts/install_wsl_dependencies.sh not found."

    def _copy_text(self, text: str, name: str) -> None:
        from PyQt6.QtGui import QGuiApplication
        from PyQt6.QtWidgets import QMessageBox

        clipboard = QGuiApplication.clipboard()
        if clipboard:
            clipboard.setText(text)
            QMessageBox.information(
                self, "Copied", f"<p>{name} has been copied to the clipboard.</p>"
            )

    def _run_in_terminal(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        from src.shared.python.data_io.path_utils import get_repo_root

        try:
            subprocess.Popen(
                [
                    "cmd.exe",
                    "/c",
                    "start",
                    "cmd.exe",
                    "/k",
                    "wsl bash scripts/install_wsl_dependencies.sh",
                ],
                cwd=str(get_repo_root()),
            )
            QMessageBox.information(
                self,
                "WSL Setup Script Started",
                "<p>Launched the WSL setup script in a new terminal window.</p>"
                "<p>Please check the opened window to monitor progress, enter your password if prompted, "
                "and verify installation success.</p>",
            )
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Execution Error",
                f"<p>Failed to launch terminal process: {exc}</p>",
            )
