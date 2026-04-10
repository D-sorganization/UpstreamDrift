"""Custom build hooks to bundle UI into Python package."""

import logging
import subprocess
import sys
from os import environ
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    """Return True when an environment variable is set to a truthy value."""
    value = environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


class UIBuildHook(BuildHookInterface):
    """Build the React UI and include it in the wheel."""

    def initialize(self, version: str, build_data: dict) -> None:
        """Initialize build hook."""
        if not (version):
            raise ValueError("Version parameter must not be empty")
        if not (build_data is not None):
            raise ValueError("Build data dictionary must be provided")

        ui_dir = Path(self.root) / "ui"
        dist_dir = ui_dir / "dist"

        hook_config = self.config
        force_ui_build = bool(hook_config.get("force_ui_build"))
        skip_requested = _env_flag("CI") or _env_flag("SKIP_UI_BUILD")
        dist_exists = dist_dir.exists()

        if dist_exists and not force_ui_build:
            if skip_requested:
                logger.info("Using existing UI bundle at %s", dist_dir)
            else:
                logger.info("Using existing UI build at %s", dist_dir)
            return

        if skip_requested and not force_ui_build:
            if version == "editable":
                logger.warning(
                    "Skipping UI bundle enforcement for editable build without %s",
                    dist_dir,
                )
                return
            msg = (
                f"UI bundle is missing at {dist_dir} and UI build is disabled "
                "by CI/SKIP_UI_BUILD. Build ui/dist before packaging or set "
                "force_ui_build=true to rebuild it."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if not dist_exists or force_ui_build:
            logger.info("Building UI...")

            # Check if npm is available
            npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

            try:
                # Install dependencies
                # Use --legacy-peer-deps to handle potential React version conflicts
                subprocess.run(
                    [npm_cmd, "ci", "--legacy-peer-deps"],
                    cwd=str(ui_dir),
                    check=True,
                    capture_output=True,
                    text=True,
                )

                # Build production bundle
                subprocess.run(
                    [npm_cmd, "run", "build"],
                    cwd=str(ui_dir),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info("UI built successfully to %s", dist_dir)

            except FileNotFoundError:
                msg = "npm not found. Please install Node.js to build the UI."
                logger.error("Error: %s", msg)
                raise RuntimeError(msg) from None

            except subprocess.CalledProcessError as e:
                msg = f"UI build failed: {e.stderr or e.stdout or str(e)}"
                logger.error("Error: %s", msg)
                raise RuntimeError(msg) from e

        else:
            logger.info("Using existing UI build at %s", dist_dir)
