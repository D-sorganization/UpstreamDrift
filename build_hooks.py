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

import importlib.util
import logging
import re
import subprocess
import sys
from collections.abc import Callable
from os import environ
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

logger = logging.getLogger(__name__)


def _load_tools_source_digest() -> Callable[[Path], str]:
    """Load the adjacent helper without assuming the source root is importable.

    PEP 517 loads custom hooks by file path in an isolated backend process. The
    repository root is therefore not guaranteed to be on ``sys.path`` even
    though the helper is present in both the checkout and source distribution.
    """

    helper = (
        Path(__file__).resolve().parent
        / "scripts"
        / "packaging"
        / "pinned_tools_provenance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_upstreamdrift_pinned_tools_provenance",
        helper,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Pinned Tools provenance helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    digest = getattr(module, "compute_tools_source_sha256", None)
    if not callable(digest):
        raise RuntimeError("Pinned Tools provenance helper is invalid")
    return digest


compute_tools_source_sha256 = _load_tools_source_digest()

_TOOLS_DIRECTORY_DESTINATIONS = {
    "shared": "shared",
    "sidekick": "sidekick",
    "chat": "chat",
    "python/src/utils": "utils",
}
_TOOLS_FILE_DESTINATIONS = {"contracts.py": "contracts.py"}
_UPSTREAM_EXTENSION_PACKAGES = ("sidekick", "chat")
_UPSTREAM_CANONICAL_PACKAGES = ("bunkershot3d",)
_TOOLS_PROVENANCE_PATHS = (
    "src/shared",
    "src/sidekick",
    "src/chat",
    "src/python/src/utils",
    "src/contracts.py",
)
_OWNERSHIP_MANIFEST_RELATIVE = Path(
    "scripts/config/shared_python_ownership_exceptions.yaml"
)


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

    @staticmethod
    def _is_test_artifact(relative: Path) -> bool:
        """Return True for files that must never ship in a wheel."""
        parts = relative.parts
        if any(part in {"__pycache__", ".pytest_cache", "tests"} for part in parts):
            return True
        if relative.suffix in {".pyc", ".pyo"}:
            return True
        return relative.name == "conftest.py" or (
            relative.name.startswith("test_") and relative.suffix == ".py"
        )

    @property
    def _canonical_tools_src_root(self) -> Path:
        """Return the pinned Tools source root used as wheel authority."""
        return Path(self.root) / "vendor" / "ud-tools" / "src"

    @property
    def _local_tools_python_root(self) -> Path:
        """Return the Upstream tree containing classified local extensions."""
        return Path(self.root) / "src" / "shared" / "python"

    @property
    def _ownership_manifest_path(self) -> Path:
        """Return the file-level ownership manifest for local extensions."""
        return Path(self.root) / _OWNERSHIP_MANIFEST_RELATIVE

    @staticmethod
    def _validated_git_sha(value: str) -> str:
        """Return a normalized Git object ID or reject malformed metadata."""
        normalized = value.strip().lower()
        valid_length = len(normalized) in {40, 64}
        valid_characters = all(
            character in "0123456789abcdef" for character in normalized
        )
        if not valid_length or not valid_characters:
            raise ValueError("Malformed Git object ID")
        return normalized

    def _validate_pinned_tools_checkout(self, version: str) -> bool:
        """Verify that the checked-out Tools commit matches the gitlink.

        Returns ``False`` only for an editable, non-CI build whose Git metadata
        is unavailable. An explicit mismatch is always fatal.
        """
        root = str(Path(self.root))
        subprocess_contract = {
            "cwd": root,
            "check": True,
            "capture_output": True,
            "text": True,
        }
        try:
            gitlink_result = subprocess.run(
                ["git", "ls-tree", "HEAD", "--", "vendor/ud-tools"],
                **subprocess_contract,
            )
            checkout_result = subprocess.run(
                ["git", "-C", "vendor/ud-tools", "rev-parse", "HEAD"],
                **subprocess_contract,
            )
            gitlink_fields = gitlink_result.stdout.strip().split(maxsplit=3)
            if (
                len(gitlink_fields) < 3
                or gitlink_fields[0] != "160000"
                or gitlink_fields[1] != "commit"
            ):
                raise ValueError("Malformed Tools gitlink metadata")
            expected_sha = self._validated_git_sha(gitlink_fields[2])
            actual_sha = self._validated_git_sha(checkout_result.stdout)
        except (
            OSError,
            subprocess.CalledProcessError,
            ValueError,
            TypeError,
            AttributeError,
        ):
            if self._validate_container_tools_provenance():
                return True
            message = "Pinned Tools checkout git metadata unavailable"
            if version != "editable" or _env_flag("CI"):
                raise RuntimeError(message) from None
            logger.warning("%s; skipping editable force-includes", message)
            return False

        if actual_sha != expected_sha:
            raise RuntimeError(
                "Pinned Tools checkout does not match superproject gitlink"
            )
        dirty_result = subprocess.run(
            [
                "git",
                "-C",
                "vendor/ud-tools",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *_TOOLS_PROVENANCE_PATHS,
            ],
            **subprocess_contract,
        )
        if dirty_result.stdout.strip():
            raise RuntimeError("Pinned Tools package sources are not clean")
        return True

    def _validate_container_tools_provenance(self) -> bool:
        """Validate a content-bound pin when Docker excludes Git metadata.

        Returns ``False`` when no container attestation was supplied. If either
        attestation field is present, both must be valid and the declared
        source digest must match the exact package roots copied into the build.
        """

        declared_pin = environ.get("UPSTREAMDRIFT_TOOLS_GITLINK_SHA")
        declared_digest = environ.get("UPSTREAMDRIFT_TOOLS_SOURCE_SHA256")
        if declared_pin is None and declared_digest is None:
            return False
        if declared_pin is None or declared_digest is None:
            raise RuntimeError("Pinned Tools container provenance is incomplete")
        try:
            self._validated_git_sha(declared_pin)
            expected_digest = self._validated_git_sha(declared_digest)
            observed_digest = compute_tools_source_sha256(
                self._canonical_tools_src_root.parent
            )
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Pinned Tools container provenance is invalid"
            ) from error
        if observed_digest != expected_digest:
            raise RuntimeError("Pinned Tools container source digest does not match")
        return True

    def _extension_owners(self) -> dict[Path, str]:
        """Return local extension ownership decisions or fail closed.

        Hatch build isolation supplies Hatchling and the standard library, not
        PyYAML. The parser below intentionally accepts only this repository's
        constrained ``paths -> owner`` manifest shape.
        """
        try:
            lines = self._ownership_manifest_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except OSError as error:
            raise RuntimeError(
                "Shared Python ownership manifest is unavailable"
            ) from error

        owners: dict[Path, str] = {}
        current_path: Path | None = None
        saw_paths = False
        for line in lines:
            if line == "paths:":
                saw_paths = True
                continue
            if not saw_paths or not line or line.lstrip().startswith("#"):
                continue
            path_match = re.fullmatch(r"  ([^:\s][^:]*\.py):", line)
            if path_match is not None:
                current_path = Path(path_match.group(1))
                if current_path in owners:
                    raise RuntimeError("Shared Python ownership manifest is malformed")
                owners[current_path] = ""
                continue
            owner_match = re.fullmatch(r"    owner: (UpstreamDrift|Unresolved)", line)
            if owner_match is not None:
                if current_path is None:
                    raise RuntimeError("Shared Python ownership manifest is malformed")
                owners[current_path] = owner_match.group(1)

        if not saw_paths or not owners or any(not owner for owner in owners.values()):
            raise RuntimeError("Shared Python ownership manifest is malformed")
        return owners

    def _register_tools_packages(self, version: str, build_data: dict) -> None:
        """Register canonical Tools packages plus non-conflicting extensions.

        The pinned Tools submodule is authoritative for every same-relative
        file. Upstream-owned files supplement a package only when the canonical
        package has no counterpart. Test artifacts from either tree never
        enter the wheel.

        Release and CI builds require both canonical package roots. Editable
        local installs may proceed without forced copies because imports
        resolve directly from their checkout.
        """
        if not version:
            raise ValueError("Version parameter must not be empty")
        if build_data is None:
            raise ValueError("Build data dictionary must be provided")

        canonical_src = self._canonical_tools_src_root
        canonical_directories = {
            source_name: canonical_src / source_name
            for source_name in _TOOLS_DIRECTORY_DESTINATIONS
        }
        missing = [
            source_name
            for source_name, source_root in canonical_directories.items()
            if not source_root.is_dir()
        ]
        missing.extend(
            source_name
            for source_name in _TOOLS_FILE_DESTINATIONS
            if not (canonical_src / source_name).is_file()
        )
        if missing:
            message = (
                "Pinned Tools package roots are required for release/CI builds; "
                f"missing: {', '.join(missing)}"
            )
            if version != "editable" or _env_flag("CI"):
                raise RuntimeError(message)
            logger.warning("%s; skipping editable force-includes", message)
            return

        if not self._validate_pinned_tools_checkout(version):
            return

        force_include = build_data.setdefault("force_include", {})
        for source_name, canonical_root in canonical_directories.items():
            destination_root = _TOOLS_DIRECTORY_DESTINATIONS[source_name]
            for path in sorted(canonical_root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(canonical_root)
                if not self._is_test_artifact(relative):
                    force_include[str(path)] = "/".join(
                        (destination_root, *relative.parts)
                    )

        for source_name, destination in _TOOLS_FILE_DESTINATIONS.items():
            force_include[str(canonical_src / source_name)] = destination

        canonical_python = canonical_src / "shared" / "python"
        local_python = self._local_tools_python_root
        extension_owners = self._extension_owners()
        for package_name in _UPSTREAM_EXTENSION_PACKAGES:
            canonical_root = canonical_python / package_name
            canonical_files = {
                path.relative_to(canonical_root)
                for path in canonical_root.rglob("*")
                if path.is_file()
            }
            local_root = local_python / package_name
            if not local_root.is_dir():
                continue
            for path in sorted(local_root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(local_root)
                if (
                    relative.suffix != ".py"
                    or relative in canonical_files
                    or self._is_test_artifact(relative)
                ):
                    continue
                extension_path = Path(package_name) / relative
                owner = extension_owners.get(extension_path)
                if owner is None:
                    raise RuntimeError(
                        "Local shared extension lacks ownership classification: "
                        f"{extension_path.as_posix()}"
                    )
                if owner != "UpstreamDrift":
                    continue
                force_include[str(path)] = "/".join(
                    ("shared", "python", package_name, *relative.parts)
                )

    def _register_upstream_packages(self, version: str, build_data: dict) -> None:
        """Expose Upstream-owned packages under their canonical import roots.

        The wheel otherwise packages the whole ``src`` directory as the literal
        top-level package ``src``.  Repository tests add ``src/`` to their import
        path, which masks that layout mismatch for packages whose public name is
        intentionally top-level.  Registering each source file explicitly keeps
        editable and wheel installs aligned without library-time ``sys.path``
        mutation or a second supported module identity.
        """
        if not version:
            raise ValueError("Version parameter must not be empty")
        if build_data is None:
            raise ValueError("Build data dictionary must be provided")

        source_root = Path(self.root) / "src"
        force_include = build_data.setdefault("force_include", {})
        for package_name in _UPSTREAM_CANONICAL_PACKAGES:
            package_root = source_root / package_name
            if not package_root.is_dir():
                raise RuntimeError(
                    f"Canonical Upstream package root is missing: {package_name}"
                )
            for path in sorted(package_root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(package_root)
                if not self._is_test_artifact(relative):
                    force_include[str(path)] = "/".join((package_name, *relative.parts))

    def initialize(self, version: str, build_data: dict) -> None:
        """Initialize build hook."""
        if not version:
            raise ValueError("Version parameter must not be empty")
        if build_data is None:
            raise ValueError("Build data dictionary must be provided")

        self._ensure_ui_bundle(version)
        self._register_ui_bundle(version, build_data)
        self._register_upstream_packages(version, build_data)
        self._register_tools_packages(version, build_data)

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
