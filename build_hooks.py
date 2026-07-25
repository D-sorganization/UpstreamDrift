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


def _env_flag(name: str) -> bool:
    """Return True when an environment variable is set to a truthy value."""
    value = environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    def _subprocess_error_message(e: subprocess.CalledProcessError) -> str:
        """Extract the most informative message from a CalledProcessError."""
        return e.stderr or e.stdout or str(e)

    _npm_error_message = _subprocess_error_message

    def _register_ui_bundle(self, version: str, build_data: dict) -> None:
        """Force-include the built UI bundle in the wheel payload.

        ``[tool.hatch.build] include``/``artifacts`` are inert for the wheel
        target because ``packages = ["src"]`` narrows the file selection to the
        ``src`` tree, so the compiled frontend never shipped (issue #8018).
        Registering it here rather than in a static ``force-include`` table
        keeps the ``SKIP_UI_BUILD`` escape hatch working: when the bundle is
        genuinely absent the wheel is simply built without it instead of the
        build hard-failing on a missing force-include source.

        ``ui/dist`` is placed at the install root so it matches
        ``_resolve_ui_dist_path()`` in ``src/api/local_server.py``, which looks
        three parents up from ``src/api/local_server.py``.
        """
        if version == "editable":
            # An editable install resolves ui/dist straight out of the checkout,
            # so copying it into site-packages would only create a stale mirror.
            return
        dist_dir = self._dist_dir
        if not dist_dir.is_dir():
            logger.warning("No UI bundle at %s; wheel will ship without it", dist_dir)
            return
        force_include = build_data.setdefault("force_include", {})
        force_include[str(dist_dir)] = "ui/dist"

    @property
    def _sidekick_dir(self) -> Path:
        """Source directory of the shared Sidekick package."""
        return Path(self.root) / "src" / "shared" / "python" / "sidekick"

    @staticmethod
    def _is_test_artifact(relative: Path) -> bool:
        """Return True for files that must never ship in a wheel."""
        parts = relative.parts
        if "__pycache__" in parts or "tests" in parts:
            return True
        return relative.name.startswith("test_") and relative.suffix == ".py"

    def _register_sidekick_package(self, build_data: dict) -> None:
        """Force-include the Sidekick package without its test suite.

        Hatchling's ``recurse_forced_files`` never consults ``include_path``,
        so a whole-directory ``force-include`` entry is immune to the
        ``exclude`` patterns in ``[tool.hatch.build]`` — which is why the wheel
        shipped the entire ``sidekick`` test suite (issue #8018). Enumerating
        the tree here lets the same relocation happen file by file with the
        test artefacts filtered out.
        """
        source_root = self._sidekick_dir
        if not source_root.is_dir():
            logger.warning("Sidekick package not found at %s", source_root)
            return
        force_include = build_data.setdefault("force_include", {})
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_root)
            if self._is_test_artifact(relative):
                continue
            force_include[str(path)] = "/".join(("sidekick", *relative.parts))

    def initialize(self, version: str, build_data: dict) -> None:
        """Initialize build hook."""
        if not version:
            raise ValueError("Version parameter must not be empty")
        if build_data is None:
            raise ValueError("Build data dictionary must be provided")

        self._ensure_ui_bundle(version)
        self._register_ui_bundle(version, build_data)
        self._register_sidekick_package(build_data)

    def _ensure_ui_bundle(self, version: str) -> None:
        """Build (or validate the presence of) the compiled frontend bundle."""
        dist_dir = self._dist_dir

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
            if version == "editable" or _env_flag("SKIP_UI_BUILD"):
                logger.warning(
                    "Skipping UI bundle enforcement because SKIP_UI_BUILD is set",
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
            ui_dir = self._ui_dir
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
            msg = f"UI build failed: {self._subprocess_error_message(e)}"
            logger.error("Error: %s", msg)
            raise RuntimeError(msg) from e
