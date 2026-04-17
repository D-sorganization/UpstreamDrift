"""Custom Hatchling build hook that compiles the React UI into the Python wheel.

Bundle steps
------------
1. Skipped entirely when the ``CI`` or ``SKIP_UI_BUILD`` environment variable is
   set — CI builds the wheel after a separate frontend build step.
2. If ``ui/dist/`` already exists and ``force_ui_build`` is not set in
   ``[tool.hatch.build.hooks.custom]``, the existing build is reused.
3. Otherwise ``npm ci --legacy-peer-deps`` installs exact locked dependencies,
   then ``npm run build`` compiles the Vite bundle into ``ui/dist/``.
4. On failure the hook raises ``RuntimeError`` so ``hatch build`` / ``pip install``
   surfaces a clear error rather than silently shipping without the UI.

Hatch configuration (pyproject.toml)::

    [tool.hatch.build.hooks.custom]
    path = "build_hooks.py"
    # force_ui_build = true  # uncomment to rebuild even when dist/ exists
"""

import logging
import subprocess
import sys
from os import environ
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

logger = logging.getLogger(__name__)


class UIBuildHook(BuildHookInterface):
    """Build the React UI and include it in the wheel."""

    @property
    def _ui_dir(self) -> Path:
        """Root directory of the frontend source tree."""
        return Path(self.root) / "ui"

    @property
    def _dist_dir(self) -> Path:
        """Output directory of the compiled frontend bundle."""
        return self._ui_dir / "dist"

    def _force_ui_build(self) -> bool:
        """Return True when the hook config requests a forced rebuild."""
        return bool(self.config.get("force_ui_build"))

    @staticmethod
    def _npm_error_message(e: subprocess.CalledProcessError) -> str:
        """Extract the most informative message from a CalledProcessError."""
        return e.stderr or e.stdout or str(e)

    def initialize(self, version: str, build_data: dict) -> None:
        """Initialize build hook."""
        if not (version):
            raise ValueError("Version parameter must not be empty")
        if build_data is None:
            raise ValueError("Build data dictionary must be provided")

        dist_dir = self._dist_dir

        # Always skip UI build in CI environment or if explicitly requested
        if environ.get("CI") or environ.get("SKIP_UI_BUILD"):
            logger.warning("Skipping UI build (CI environment or SKIP_UI_BUILD set)")
            if not dist_dir.exists():
                logger.warning("Warning: UI dist directory does not exist!")
            return

        if not dist_dir.exists() or self._force_ui_build():
            self._run_npm_build()
            logger.info("UI built successfully to %s", dist_dir)
        else:
            logger.info("Using existing UI build at %s", dist_dir)

    def _run_npm_build(self) -> None:
        """Run npm ci and npm run build inside the UI directory."""
        ui_dir = self._ui_dir
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        try:
            subprocess.run(
                [npm_cmd, "ci", "--legacy-peer-deps"],
                cwd=str(ui_dir),
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [npm_cmd, "run", "build"],
                cwd=str(ui_dir),
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            msg = "npm not found. Please install Node.js to build the UI."
            logger.error("Error: %s", msg)
            raise RuntimeError(msg) from None
        except subprocess.CalledProcessError as e:
            msg = f"UI build failed: {self._npm_error_message(e)}"
            logger.error("Error: %s", msg)
            raise RuntimeError(msg) from e
