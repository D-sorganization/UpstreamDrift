"""Windows Desktop and Start Menu shortcut manager for UpstreamDrift.

Creates and validates portable, idempotent shortcuts to the classic PyQt launcher,
ensuring consistent application identity, working directory, icon location, and
Windows AppUserModelID integration.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.launchers.app_identity import (
    APP_DISPLAY_NAME,
    APP_USER_MODEL_ID,
    SHORTCUT_FILENAME,
    resolve_canonical_icon,
    resolve_launcher_executable,
)

__all__ = [
    "InstallationResult",
    "ShortcutManager",
    "ShortcutMetadata",
    "ShortcutSpec",
    "build_default_shortcut_spec",
    "get_desktop_dir",
    "get_start_menu_dir",
    "install_desktop_and_start_menu_shortcuts",
]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShortcutSpec:
    """Specification defining shortcut target and metadata."""

    name: str
    target_path: Path
    arguments: str
    working_dir: Path
    icon_path: Path | None
    description: str

    def __post_init__(self) -> None:
        """Validate shortcut specification invariants (Design by Contract)."""
        if not self.name.endswith(".lnk"):
            raise ValueError(f"name must end with .lnk, got {self.name!r}")
        if not self.target_path.exists():
            raise ValueError(f"target_path must exist on disk: {self.target_path}")
        if not self.working_dir.exists():
            raise ValueError(f"working_dir must exist on disk: {self.working_dir}")
        if self.icon_path is not None and not self.icon_path.exists():
            raise ValueError(f"icon_path must exist when specified: {self.icon_path}")
        if not isinstance(self.arguments, str):
            raise TypeError("arguments must be a string")
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")


@dataclass(frozen=True)
class ShortcutMetadata:
    """Parsed metadata read back from an existing .lnk shortcut."""

    name: str
    path: Path
    target_path: str
    arguments: str
    working_directory: str
    icon_location: str
    description: str


@dataclass
class InstallationResult:
    """Outcome of shortcut installation across target directories."""

    desktop_shortcut: Path | None = None
    start_menu_shortcut: Path | None = None
    created: list[Path] = None  # type: ignore[assignment]
    updated: list[Path] = None  # type: ignore[assignment]
    metadata: dict[Path, ShortcutMetadata] = None  # type: ignore[assignment]
    success: bool = False

    def __post_init__(self) -> None:
        if self.created is None:
            self.created = []
        if self.updated is None:
            self.updated = []
        if self.metadata is None:
            self.metadata = {}


def get_desktop_dir() -> Path:
    """Return the current user's Desktop directory path."""
    if sys.platform == "win32":
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            desktop = Path(user_profile) / "Desktop"
            if desktop.is_dir():
                return desktop
        # Fallback to home
        return Path.home() / "Desktop"
    return Path.home() / "Desktop"


def get_start_menu_dir() -> Path:
    """Return the current user's Start Menu Programs directory path."""
    if sys.platform == "win32":
        app_data = os.environ.get("APPDATA")
        if app_data:
            programs = (
                Path(app_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
            )
            if programs.is_dir():
                return programs
    return Path.home() / "Start Menu" / "Programs"


def build_default_shortcut_spec(
    repo_root: Path | None = None, windowed: bool = True
) -> ShortcutSpec:
    """Construct the default ShortcutSpec targeting the UpstreamDrift classic launcher.

    Args:
        repo_root: Root directory of the repository. Defaults to parent of ``src/``.
        windowed: Whether to prefer ``pythonw.exe`` to suppress the console window.

    Returns:
        Configured ShortcutSpec.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent

    target_exe = resolve_launcher_executable(prefer_windowed=windowed)
    launcher_script = repo_root / "launch_upstream_drift.py"
    icon = resolve_canonical_icon(repo_root / "src" / "launchers" / "assets")

    # Use relative script path if within repo root for cleaner arguments
    try:
        script_arg = str(launcher_script.relative_to(repo_root))
    except ValueError:
        script_arg = str(launcher_script)

    arguments = f'"{script_arg}" --classic'

    return ShortcutSpec(
        name=SHORTCUT_FILENAME,
        target_path=target_exe,
        arguments=arguments,
        working_dir=repo_root,
        icon_path=icon,
        description=f"Launch {APP_DISPLAY_NAME} (Classic PyQt6 Launcher)",
    )


class ShortcutManager:
    """Manages Windows shortcut creation, updating, and inspection."""

    @staticmethod
    def _generate_powershell_script(shortcut_path: Path, spec: ShortcutSpec) -> str:
        """Generate PowerShell command to create or update a .lnk file."""
        dest_str = str(shortcut_path).replace("'", "''")
        target_str = str(spec.target_path).replace("'", "''")
        args_str = spec.arguments.replace("'", "''")
        cwd_str = str(spec.working_dir).replace("'", "''")
        desc_str = spec.description.replace("'", "''")
        icon_str = str(spec.icon_path).replace("'", "''") if spec.icon_path else ""

        ps_lines = [
            "$ErrorActionPreference = 'Stop'",
            "$WshShell = New-Object -ComObject WScript.Shell",
            f"$Shortcut = $WshShell.CreateShortcut('{dest_str}')",
            f"$Shortcut.TargetPath = '{target_str}'",
            f"$Shortcut.Arguments = '{args_str}'",
            f"$Shortcut.WorkingDirectory = '{cwd_str}'",
            f"$Shortcut.Description = '{desc_str}'",
        ]
        if icon_str:
            ps_lines.append(f"$Shortcut.IconLocation = '{icon_str}'")

        ps_lines.append("$Shortcut.Save()")
        return "\n".join(ps_lines)

    @staticmethod
    def _generate_read_script(shortcut_path: Path) -> str:
        """Generate PowerShell script to read metadata from an existing .lnk file."""
        path_escaped = str(shortcut_path).replace("'", "''")
        return f"""
$ErrorActionPreference = 'Stop'
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{path_escaped}')
Write-Output "TargetPath=$($Shortcut.TargetPath)"
Write-Output "Arguments=$($Shortcut.Arguments)"
Write-Output "WorkingDirectory=$($Shortcut.WorkingDirectory)"
Write-Output "IconLocation=$($Shortcut.IconLocation)"
Write-Output "Description=$($Shortcut.Description)"
"""

    def create_shortcut(self, destination: Path, spec: ShortcutSpec) -> bool:
        """Create or update a shortcut at the specified destination path.

        Args:
            destination: Full path of the shortcut to create (.lnk).
            spec: Shortcut specification.

        Returns:
            True if creation succeeded, False otherwise.
        """
        if not destination.name.endswith(".lnk"):
            raise ValueError(f"destination must end with .lnk, got {destination}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        ps_command = self._generate_powershell_script(destination, spec)

        try:
            self._run_powershell(ps_command)
            _logger.info("Successfully installed shortcut: %s", destination)
            return True
        except (subprocess.CalledProcessError, OSError) as exc:
            _logger.error("Failed to install shortcut at %s: %s", destination, exc)
            return False

    @staticmethod
    def _run_powershell(ps_command: str) -> subprocess.CompletedProcess[str]:
        """Execute a PowerShell command string safely."""
        return subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_command],
            check=True,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _parse_metadata(
        shortcut_path: Path, raw_output: str
    ) -> ShortcutMetadata | None:
        """Parse raw PowerShell key=value output into ShortcutMetadata."""
        props: dict[str, str] = {}
        for line in raw_output.splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                props[k.strip()] = v.strip()

        if "TargetPath" not in props:
            return None

        return ShortcutMetadata(
            name=shortcut_path.name,
            path=shortcut_path,
            target_path=props.get("TargetPath", ""),
            arguments=props.get("Arguments", ""),
            working_directory=props.get("WorkingDirectory", ""),
            icon_location=props.get("IconLocation", ""),
            description=props.get("Description", ""),
        )

    def read_shortcut(self, shortcut_path: Path) -> ShortcutMetadata | None:
        """Read and inspect metadata from an existing shortcut.

        Args:
            shortcut_path: Path to the .lnk file.

        Returns:
            ShortcutMetadata if successfully read, None otherwise.
        """
        if not shortcut_path.is_file():
            return None

        ps_command = self._generate_read_script(shortcut_path)
        try:
            result = self._run_powershell(ps_command)
            return self._parse_metadata(shortcut_path, result.stdout)
        except (subprocess.CalledProcessError, OSError) as exc:
            _logger.warning("Could not read shortcut %s: %s", shortcut_path, exc)
            return None

    @classmethod
    def read_shortcut_metadata(cls, shortcut_path: Path) -> ShortcutMetadata | None:
        """Inspect metadata from an existing shortcut."""
        return cls().read_shortcut(shortcut_path)


def install_desktop_and_start_menu_shortcuts(
    repo_root: Path | None = None,
    spec: ShortcutSpec | None = None,
    windowed: bool = True,
) -> InstallationResult:
    """Idempotently install UpstreamDrift shortcuts to Desktop and Start Menu.

    Args:
        repo_root: Repository root path. Defaults to root of current repo.
        spec: Optional custom ShortcutSpec. If omitted, built via
            :func:`build_default_shortcut_spec`.
        windowed: Whether to prefer ``pythonw.exe`` for windowed execution.

    Returns:
        InstallationResult summarizing created and updated shortcuts.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent

    if spec is None:
        spec = build_default_shortcut_spec(repo_root=repo_root, windowed=windowed)

    manager = ShortcutManager()
    result = InstallationResult()

    desktop_dir = get_desktop_dir()
    desktop_shortcut = desktop_dir / spec.name
    result.desktop_shortcut = desktop_shortcut

    start_menu_dir = get_start_menu_dir()
    start_menu_shortcut = start_menu_dir / spec.name
    result.start_menu_shortcut = start_menu_shortcut

    # 1. Desktop shortcut
    existed_desktop = desktop_shortcut.is_file()
    if manager.create_shortcut(desktop_shortcut, spec):
        if existed_desktop:
            result.updated.append(desktop_shortcut)
        else:
            result.created.append(desktop_shortcut)
        meta = manager.read_shortcut(desktop_shortcut)
        if meta:
            result.metadata[desktop_shortcut] = meta

    # 2. Start Menu shortcut
    existed_sm = start_menu_shortcut.is_file()
    if manager.create_shortcut(start_menu_shortcut, spec):
        if existed_sm:
            result.updated.append(start_menu_shortcut)
        else:
            result.created.append(start_menu_shortcut)
        meta = manager.read_shortcut(start_menu_shortcut)
        if meta:
            result.metadata[start_menu_shortcut] = meta

    has_desktop = bool(
        desktop_shortcut in result.created or desktop_shortcut in result.updated
    )
    has_start_menu = bool(
        start_menu_shortcut in result.created or start_menu_shortcut in result.updated
    )
    result.success = has_desktop and has_start_menu
    return result


def main() -> int:
    """CLI entrypoint for installing desktop and Start Menu shortcuts."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _logger.info("Installing %s shortcuts...", APP_DISPLAY_NAME)
    res = install_desktop_and_start_menu_shortcuts()
    if res.success:
        _logger.info("Shortcut installation succeeded.")
        for p in res.created:
            _logger.info("  Created: %s", p)
        for p in res.updated:
            _logger.info("  Updated: %s", p)
        return 0
    _logger.error("Shortcut installation failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
