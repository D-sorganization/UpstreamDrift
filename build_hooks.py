"""Custom build hooks to bundle UI into Python package."""

import os
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class UIBuildHook(BuildHookInterface):
    """Build the React UI and include it in the wheel."""

    def initialize(self, version: str, build_data: dict) -> None:
        """Initialize build hook."""
        ui_dir = Path(self.root) / "ui"
        dist_dir = ui_dir / "dist"

        # Target within the package structure
        # We place it in src/api/static or similar, or just allow the manifest to include it from root
        # For this setup, we'll try to just ensure it exists

        # Check if we need to build
        # We skip build if not in a git repo or if environment variable is set to skip
        if os.environ.get("SKIP_UI_BUILD"):
            return

        if not dist_dir.exists() or self.config.get("force_ui_build"):
            print("Building UI...")

            try:
                # Install dependencies
                subprocess.run(["npm", "ci"], cwd=str(ui_dir), check=True, shell=True)

                # Build production bundle
                subprocess.run(
                    ["npm", "run", "build"], cwd=str(ui_dir), check=True, shell=True
                )
                print(f"UI built successfully to {dist_dir}")
            except Exception as e:
                print(
                    f"Warning: UI build failed: {e}. continuing without fresh UI build."
                )
