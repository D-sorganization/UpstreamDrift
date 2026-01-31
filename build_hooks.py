"""Custom build hooks to bundle UI into Python package."""

import logging
import os
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

logger = logging.getLogger(__name__)


class UIBuildHook(BuildHookInterface):
    """Build the React UI and include it in the wheel."""

    def initialize(self, version: str, build_data: dict) -> None:
        """Initialize build hook."""
        ui_dir = Path(self.root) / "ui"
        dist_dir = ui_dir / "dist"

        # Check if we should skip UI build
        if os.environ.get("SKIP_UI_BUILD"):
            logger.warning("Skipping UI build (SKIP_UI_BUILD is set)")
            if not dist_dir.exists():
                logger.warning("Warning: UI dist directory does not exist!")
            return

        if not dist_dir.exists() or self.config.get("force_ui_build"):
            print("Building UI...")

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
                print(f"UI built successfully to {dist_dir}")

            except FileNotFoundError:
                msg = "npm not found. Please install Node.js to build the UI."
                print(f"Error: {msg}")
                raise RuntimeError(msg) from None

            except subprocess.CalledProcessError as e:
                msg = f"UI build failed: {e.stderr or e.stdout or str(e)}"
                print(f"Error: {msg}")
                raise RuntimeError(msg) from e

        else:
            print(f"Using existing UI build at {dist_dir}")
