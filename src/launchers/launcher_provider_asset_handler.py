"""Launch handler for provider-supplied model *assets*.

The launcher manifest gains child tiles from the sibling provider repos
(``MuJoCo_Models``, ``Drake_Models``, ``Pinocchio_Models``, ``OpenSim_Models``).
Their ``type`` is the asset *format* — ``mjcf``, ``urdf``, ``osim``,
``sdformat-1.8`` — not an application name, so ``ModelHandlerRegistry`` had no
handler for any of them and 28 visible tiles could not launch at all (#8087).

This handler closes that gap:

* it claims every provider asset format;
* when the engine runtime is missing it raises
  :class:`EngineRuntimeUnavailableError`, whose message names the engine and
  the exact install command — the launcher renders it verbatim and stays
  alive;
* when the runtime is present it delegates to the engine's existing viewer
  entry point with the asset path in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src.launchers.launcher_failure_messages import ENGINE_INSTALL_HINTS
from src.launchers.launcher_model_sources import (
    get_model_python_paths,
    get_model_working_directory,
    resolve_model_artifact_path,
)
from src.launchers.launcher_provider_compatibility import is_engine_runtime_available
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from src.launchers.launcher_process_manager import ProcessManager

logger = get_logger(__name__)

__all__ = [
    "PROVIDER_ASSET_TYPES",
    "EngineRuntimeUnavailableError",
    "ProviderModelAssetHandler",
]


class EngineRuntimeUnavailableError(RuntimeError):
    """Raised when a provider tile needs an engine runtime that is not installed.

    ``str(exc)`` is a complete, user-facing message: the launcher shows it as
    is rather than wrapping it in a generic traceback dialog.
    """

    #: Read by ``launcher_failure_messages.describe_launch_failure``.
    is_user_facing_message = True


#: Provider asset formats -> engine that can open them.
PROVIDER_ASSET_TYPES: Final[dict[str, str]] = {
    "mjcf": "mujoco",
    "xml": "mujoco",
    "urdf": "pinocchio",
    "osim": "opensim",
    "sdf": "drake",
    "sdformat": "drake",
    "sdformat-1.8": "drake",
    "sdformat-1.9": "drake",
}

#: Bundled model browser, used for the formats it can render.
_MODEL_EXPLORER = "src/tools/model_explorer/launch_model_explorer.py"

#: Engine -> viewer script, relative to the repository root. Every entry is
#: asserted to exist by tests/launchers/test_qa_launcher_crash_containment.py
#: so a moved viewer surfaces as a test failure, not a dead tile.
_ENGINE_VIEWERS: Final[dict[str, str]] = {
    "mujoco": _MODEL_EXPLORER,
    "pinocchio": _MODEL_EXPLORER,
    "drake": "src/engines/physics_engines/drake/python/src/drake_gui_app.py",
    "opensim": "src/engines/physics_engines/opensim/python/opensim_gui.py",
}

_ENGINE_LABELS: Final[dict[str, str]] = {
    "mujoco": "MuJoCo",
    "drake": "Drake",
    "pinocchio": "Pinocchio",
    "opensim": "OpenSim",
}


class ProviderModelAssetHandler:
    """Handler for provider-supplied model asset tiles.

    Design by Contract:
        Precondition: ``model`` carries a resolvable ``path``.
        Postcondition: either a viewer subprocess is running, or an
        :class:`EngineRuntimeUnavailableError` carrying actionable install
        guidance is raised. The handler never returns silently on failure.
    """

    MODEL_TYPES = frozenset(PROVIDER_ASSET_TYPES)

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        if model_type is None:
            raise ValueError("model_type must be provided")
        return model_type.strip().lower() in self.MODEL_TYPES

    def _engine_for(self, model: Any) -> str:
        engine_type = (getattr(model, "engine_type", None) or "").strip().lower()
        if engine_type in _ENGINE_VIEWERS:
            return engine_type
        model_type = (getattr(model, "type", "") or "").strip().lower()
        return PROVIDER_ASSET_TYPES.get(model_type, "mujoco")

    def _require_runtime(self, engine: str, model_name: str) -> None:
        if is_engine_runtime_available(engine):
            return
        label = _ENGINE_LABELS.get(engine, engine)
        install = ENGINE_INSTALL_HINTS.get(engine, f"pip install {engine}")
        raise EngineRuntimeUnavailableError(
            f"{model_name} needs the {label} runtime, which is not installed.\n\n"
            f"Install it with:\n    {install}\n\n"
            f"Then restart UpstreamDrift. The launcher is still running - "
            f"tiles for other engines are unaffected."
        )

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Open a provider model asset in its engine's viewer.

        Args:
            model: Model configuration with ``path``, ``type`` and ``engine_type``.
            repo_path: Repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True when the viewer subprocess started.

        Raises:
            ValueError: If ``repo_path`` is None.
            EngineRuntimeUnavailableError: If the engine runtime is missing, or
                the engine ships no viewer in this repository.
        """
        if repo_path is None:
            raise ValueError("repo_path must be provided")

        model_name = getattr(model, "name", None) or getattr(model, "id", "This model")
        engine = self._engine_for(model)
        self._require_runtime(engine, model_name)

        asset_path = resolve_model_artifact_path(model, repo_path)
        if not asset_path.exists():
            raise EngineRuntimeUnavailableError(
                f"{model_name} could not be opened: the model file is missing.\n\n"
                f"Expected at:\n    {asset_path}\n\n"
                f"Clone or update the provider repository that supplies it, then "
                f"restart UpstreamDrift."
            )

        viewer_rel = _ENGINE_VIEWERS.get(engine)
        viewer_path = repo_path / viewer_rel if viewer_rel else None
        if viewer_path is None or not viewer_path.exists():
            label = _ENGINE_LABELS.get(engine, engine)
            raise EngineRuntimeUnavailableError(
                f"{model_name} cannot be opened from the launcher yet: no "
                f"{label} viewer is bundled with UpstreamDrift.\n\n"
                f"The model file is available at:\n    {asset_path}\n\n"
                f"Open it with your {label} tooling directly. The launcher is "
                f"still running - other tiles are unaffected."
            )

        env = process_manager.get_subprocess_env(
            get_model_python_paths(model, repo_path)
        )
        env["UPSTREAM_DRIFT_MODEL_PATH"] = str(asset_path)
        env["UPSTREAM_DRIFT_MODEL_NAME"] = str(model_name)
        os.environ.setdefault("UPSTREAM_DRIFT_MODEL_PATH", str(asset_path))

        logger.info(
            "ProviderModelAssetHandler: opening %s (%s) with %s viewer %s",
            model_name,
            asset_path,
            engine,
            viewer_path,
        )
        process = process_manager.launch_script(
            name=str(model_name),
            script_path=viewer_path,
            cwd=get_model_working_directory(model, repo_path),
            env=env,
            extra_python_paths=get_model_python_paths(model, repo_path),
        )
        return process is not None

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """Provider asset tiles open in a separate viewer process."""
        return None
