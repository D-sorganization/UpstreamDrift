"""Model-specific launch handlers for the Golf Launcher.

This module provides specialized launch logic for different physics engines
and simulation types (MuJoCo, Drake, Pinocchio, OpenSim, etc.).

DRY refactoring: Consolidated 7 near-identical handler classes into
a data-driven ScriptHandler with a handler registry table.
"""

from __future__ import annotations

import os
import platform
import subprocess
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from src.launchers.launcher_model_sources import (
    get_model_source_root,
    get_model_python_paths,
    get_model_working_directory,
    resolve_model_artifact_path,
)
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from src.launchers.launcher_process_manager import ProcessManager

logger = get_logger(__name__)


def _package_main_module_name(
    model_path: str,
    *,
    source_root: Path,
    launcher_root: Path,
) -> str | None:
    """Return the importable module for a package ``__main__`` launcher path.

    Local ``src/tools`` tiles need the ``src`` package prefix because the
    launcher places the repository root before ``repo/src`` on ``PYTHONPATH``
    and the repository root has another ``tools`` package.  Other packages
    retain their established ``repo/src`` import style.  Sibling providers
    expose their own ``src`` directory as an extra Python path, so their
    package modules deliberately omit that prefix.
    """
    normalized_path = model_path.replace("\\", "/")
    if not normalized_path.startswith("src/") or not normalized_path.endswith(
        "/__main__.py"
    ):
        return None
    module_parts = normalized_path.removesuffix("/__main__.py").split("/")
    is_local_tool_package = (
        source_root.resolve() == launcher_root.resolve()
        and module_parts[:2] == ["src", "tools"]
    )
    if not is_local_tool_package:
        module_parts = module_parts[1:]
    if not module_parts or not all(part.isidentifier() for part in module_parts):
        return None
    return ".".join(module_parts)


def _registry_dockable_ui(model: Any) -> Any | None:
    """Resolve ``model``'s embedded UI through the ADR-0013 registry.

    The ``EMBEDDABLE_TOOL_REGISTRY`` is THE embedding contract (issue
    #8857): every handler consults it before falling back to the legacy
    import-and-probe protocol. Returns ``None`` when the tile id is not
    registered (or has no usable id).
    """
    tool_id = getattr(model, "id", "")
    if not (isinstance(tool_id, str) and tool_id):
        return None
    from src.shared.python.launcher_embed.registry import get_embeddable_tool

    tool = get_embeddable_tool(tool_id)
    if tool is None:
        return None
    return tool.create_main_widget(None)


def _warn_legacy_embed_fallback(model: Any, mechanism: str) -> None:
    """Emit a DeprecationWarning for a tile embedded via the legacy path.

    The legacy protocol (module-level ``get_dockable_ui`` probing and
    ``embed_adapter`` "mod::func" strings) is deprecated in favor of the
    ADR-0013 ``EmbeddableTool`` registry. The warning names the tile so
    the remaining users are enumerable (ratchet test in
    ``tests/launchers/test_embed_contract_convergence.py``).
    """
    tile_id = getattr(model, "id", None) or "<unknown>"
    warnings.warn(
        f"Tile {tile_id!r} resolved its embedded UI via the deprecated "
        f"legacy fallback ({mechanism}). Register an EmbeddableTool "
        "adapter per ADR-0013 and add it to embedded_tool_bootstrap "
        "instead; the legacy path will be removed.",
        DeprecationWarning,
        stacklevel=3,
    )


class ModelHandler(Protocol):
    """Protocol for model launch handlers."""

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler can handle the given model type."""
        ...

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the model."""
        ...

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """Get the dockable UI widget for the model, if supported."""
        return None


class ModuleHandler:
    """Handler that launches a Python module via process_manager.launch_module.

    DRY replacement for HumanoidMuJoCoHandler and ComprehensiveModelHandler.
    """

    def __init__(
        self, model_types: set[str], module_name: str, display_name: str
    ) -> None:
        if model_types is None:
            raise ValueError("model_types must be provided")
        self.model_types = model_types
        self.module_name = module_name
        self.display_name = display_name

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.model_types

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the module."""
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        process = process_manager.launch_module(
            name=self.display_name,
            module_name=self.module_name,
            cwd=repo_path,
            extra_python_paths=get_model_python_paths(model, repo_path),
            keep_terminal_open=True,
        )
        return process is not None

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """Try to load the module and get its dockable UI widget."""
        registry_ui = _registry_dockable_ui(model)
        if registry_ui is not None:
            return registry_ui

        import importlib
        import sys

        original_sys_path = sys.path.copy()
        success = False
        try:
            # Inject paths
            paths = get_model_python_paths(model, repo_path)
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
            for p in paths:
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))

            try:
                module = importlib.import_module(f"{self.module_name}.__main__")
            except ImportError:
                module = importlib.import_module(self.module_name)

            if hasattr(module, "get_dockable_ui"):
                ui = module.get_dockable_ui()
                if ui is not None:
                    success = True
                    _warn_legacy_embed_fallback(
                        model, f"module-level get_dockable_ui in {self.module_name}"
                    )
                    return ui
        except Exception as e:  # noqa: BLE001
            logger.debug("No dockable UI found in module %s: %s", self.module_name, e)
        finally:
            if not success:
                sys.path = original_sys_path
        return None


class ScriptHandler:
    """Handler that launches a Python script via process_manager.launch_script.

    DRY replacement for DrakeHandler, PinocchioHandler, OpenSimHandler,
    MyoSimHandler, and OpenPoseHandler.
    """

    def __init__(
        self,
        model_types: set[str],
        script_path: str,
        display_name: str,
        cwd_path: str | None = None,
    ) -> None:
        if model_types is None:
            raise ValueError("model_types must be provided")
        self.model_types = model_types
        self._script_path = script_path
        self.display_name = display_name
        self._cwd_path = cwd_path

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.model_types

    def resolve_script(self, model: Any, repo_path: Path) -> Path:
        """Resolve the script to run for ``model``.

        ``models.yaml`` is the source of truth for a tile's entry point; the
        hard-coded ``script_path`` in the handler table is only a fallback for
        model types with no declared ``path``. Preferring the table silently
        overrode ``models.yaml`` and pointed the Pinocchio, OpenSim and
        MyoSuite tiles at files that do not exist (issue #8030).
        """
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        declared = getattr(model, "path", None) if model is not None else None
        if declared:
            try:
                return resolve_model_artifact_path(model, repo_path)
            except (ValueError, OSError) as exc:
                logger.warning(
                    "ScriptHandler: could not resolve declared path %r for %r "
                    "(%s); falling back to %s",
                    declared,
                    getattr(model, "id", "unknown"),
                    exc,
                    self._script_path,
                )
        return repo_path / self._script_path

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the script."""
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        script_path = self.resolve_script(model, repo_path)
        cwd = repo_path / self._cwd_path if self._cwd_path else repo_path

        process = process_manager.launch_script(
            name=self.display_name,
            script_path=script_path,
            cwd=cwd,
            extra_python_paths=get_model_python_paths(model, repo_path),
            keep_terminal_open=True,
        )
        return process is not None

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """Try to load the script as a module and get its dockable UI widget."""
        registry_ui = _registry_dockable_ui(model)
        if registry_ui is not None:
            return registry_ui

        if repo_path is None:
            return None
        script_path = self.resolve_script(model, repo_path)
        if not script_path.exists():
            return None

        import importlib.util
        import sys

        # Add repo_path to sys.path temporarily to resolve imports
        original_sys_path = sys.path.copy()
        success = False
        try:
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
            paths = get_model_python_paths(model, repo_path)
            for p in paths:
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))

            module_name = (
                str(script_path).replace("/", "_").replace("\\", "_").replace(".py", "")
            )
            spec = importlib.util.spec_from_file_location(module_name, str(script_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "get_dockable_ui"):
                    ui = module.get_dockable_ui()
                    if ui is not None:
                        success = True
                        _warn_legacy_embed_fallback(
                            model, f"module-level get_dockable_ui in {script_path.name}"
                        )
                        return ui
        except Exception as e:  # noqa: BLE001
            logger.debug("No dockable UI found in script %s: %s", self._script_path, e)
        finally:
            if not success:
                sys.path = original_sys_path
        return None


class SpecialAppHandler:
    """Handler for launching special applications (tools, utilities).

    Handles model types: special_app
    Covers: c3d_viewer, openpose, mediapipe, model_explorer, video_analyzer,
    data_explorer, and any future tool/utility tiles.

    Design by Contract:
    Precondition: model.path must be a valid relative path to a Python entry point
    Postcondition: the entry point is launched as a subprocess
    """

    MODEL_TYPES = {"special_app"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch a special application by running its script.

        Args:
            model: Model configuration with 'path' and 'name' attrs.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        # DBC Precondition: model must have a path
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error(
                "SpecialAppHandler: model '%s' has no path",
                getattr(model, "id", "unknown"),
            )
            return False

        script_path = resolve_model_artifact_path(model, repo_path)
        model_name = getattr(model, "name", model_path)

        if not script_path.exists():
            logger.warning("SpecialAppHandler: script not found: %s", script_path)
            return False

        working_directory = get_model_working_directory(model, repo_path)
        python_paths = get_model_python_paths(model, repo_path)
        package_module = _package_main_module_name(
            model_path,
            source_root=get_model_source_root(model, repo_path),
            launcher_root=repo_path,
        )
        if package_module is not None:
            process = process_manager.launch_module(
                name=model_name,
                module_name=package_module,
                cwd=working_directory,
                extra_python_paths=python_paths,
                keep_terminal_open=True,
            )
        else:
            process = process_manager.launch_script(
                name=model_name,
                script_path=script_path,
                cwd=working_directory,
                extra_python_paths=python_paths,
                keep_terminal_open=True,
            )
        return process is not None

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """Try to load the special app script and get its dockable UI widget."""
        if repo_path is None:
            return None

        registry_ui = _registry_dockable_ui(model)
        if registry_ui is not None:
            return registry_ui

        embed_adapter = getattr(model, "embed_adapter", None)
        if embed_adapter and "::" in embed_adapter:
            mod_path, func_name = embed_adapter.split("::")
            source_root = get_model_source_root(model, repo_path)
            provider_adapter = source_root / mod_path
            local_adapter = repo_path / mod_path
            adapter_script = (
                provider_adapter if provider_adapter.exists() else local_adapter
            )
            if adapter_script.exists():
                import importlib.util
                import sys

                original_sys_path = sys.path.copy()
                success = False
                try:
                    # Inject paths
                    paths = get_model_python_paths(model, repo_path)
                    if str(repo_path) not in sys.path:
                        sys.path.insert(0, str(repo_path))
                    for p in paths:
                        if str(p) not in sys.path:
                            sys.path.insert(0, str(p))

                    spec = importlib.util.spec_from_file_location(
                        "embed_adapter", str(adapter_script)
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        if hasattr(module, func_name):
                            ui = getattr(module, func_name)()
                            if ui is not None:
                                success = True
                                _warn_legacy_embed_fallback(
                                    model,
                                    f"embed_adapter string {embed_adapter!r}",
                                )
                                return ui
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to load embed_adapter %s: %s", embed_adapter, e
                    )
                finally:
                    if not success:
                        sys.path = original_sys_path

        model_path = getattr(model, "path", None) or ""
        if not model_path:
            return None

        # A conventional package entry point executes its CLI dispatcher when
        # imported with the ``__main__`` module name.  The launcher must never
        # probe it for an embeddable widget in-process: doing so can call
        # ``sys.exit()`` and take down the launcher before the subprocess
        # launch path runs.  Such apps are launched by ``launch()`` instead.
        if Path(model_path).name == "__main__.py":
            return None

        script_path = resolve_model_artifact_path(model, repo_path)
        if not script_path.exists():
            return None

        import importlib.util
        import sys

        original_sys_path = sys.path.copy()
        success = False
        try:
            # Inject paths
            paths = get_model_python_paths(model, repo_path)
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
            for p in paths:
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))

            module_name = str(script_path.name).replace(".py", "")
            spec = importlib.util.spec_from_file_location(module_name, str(script_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "get_dockable_ui"):
                    ui = module.get_dockable_ui()
                    if ui is not None:
                        success = True
                        _warn_legacy_embed_fallback(
                            model, f"module-level get_dockable_ui in {script_path.name}"
                        )
                        return ui
        except Exception as e:  # noqa: BLE001
            logger.debug("No dockable UI found in special app %s: %s", script_path, e)
        finally:
            if not success:
                sys.path = original_sys_path
        return None


class PuttingGreenHandler:
    """Handler for launching the Putting Green simulator.

    Design by Contract:
        Precondition: model.path must point to the putting green simulator
        Postcondition: putting green simulator subprocess is running
    """

    MODEL_TYPES = {"putting_green"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the Putting Green simulation.

        Args:
            model: Model configuration with 'path' attr.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error("PuttingGreenHandler: model has no path")
            return False

        script_path = resolve_model_artifact_path(model, repo_path)
        if not script_path.exists():
            logger.warning("PuttingGreenHandler: script not found: %s", script_path)
            return False

        process = process_manager.launch_script(
            name="Putting Green Simulator",
            script_path=script_path,
            cwd=get_model_working_directory(model, repo_path, script_path.parent),
            extra_python_paths=get_model_python_paths(model, repo_path),
            keep_terminal_open=True,
        )
        return process is not None

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """Get the dockable UI widget for the putting green simulation."""
        registry_ui = _registry_dockable_ui(model)
        if registry_ui is not None:
            return registry_ui

        if repo_path is None:
            return None
        script_path = resolve_model_artifact_path(model, repo_path)
        if not script_path.exists():
            return None

        import importlib.util
        import sys

        original_sys_path = sys.path.copy()
        success = False
        try:
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
            paths = get_model_python_paths(model, repo_path)
            for p in paths:
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))

            module_name = "putting_green_gui_embed"
            spec = importlib.util.spec_from_file_location(module_name, str(script_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "get_dockable_ui"):
                    ui = module.get_dockable_ui()
                    if ui is not None:
                        success = True
                        _warn_legacy_embed_fallback(
                            model, "module-level get_dockable_ui (putting green)"
                        )
                        return ui
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to get dockable UI for Putting Green: %s", e)
        finally:
            if not success:
                sys.path = original_sys_path
        return None


class BiomechExerciseHandler:
    """Handler for launching biomechanics exercise dashboards."""

    MODEL_TYPES = {"biomech_exercise"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the biomechanics exercise dashboard."""
        if repo_path is None:
            raise ValueError("repo_path must be provided")

        exercise_name = getattr(model, "exercise", "gait")
        model_name = getattr(
            model, "name", f"Biomechanics Exercise: {exercise_name.title()}"
        )
        script_path = repo_path / "src" / "launchers" / "exercise_dashboard.py"

        # Pass the exercise as a CLI argument through an environment variable or module?
        # Actually, since process_manager.launch_script doesn't take args natively,
        # we can launch it using launch_module with the right module_name, but that doesn't take args either.
        # Let's use secure_popen manually or just set an environment variable.
        env = process_manager.get_subprocess_env(
            get_model_python_paths(model, repo_path)
        )
        env["BIOMECH_EXERCISE"] = exercise_name

        # Wait, if we use launch_script, we can't pass args.
        # But we can modify launch_script to take args, or just use the environment variable fallback.
        # Let's use secure_popen if needed, but launch_script is safer. Let's see if we can pass args?
        # Let's use launch_script and let the script read sys.argv or env.
        # I'll modify exercise_dashboard.py to read BIOMECH_EXERCISE env var if --exercise is not passed.

        process = process_manager.launch_script(
            name=model_name,
            script_path=script_path,
            cwd=repo_path,
            env=env,
            extra_python_paths=get_model_python_paths(model, repo_path),
            keep_terminal_open=True,
        )
        return process is not None

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """BiomechExercise handler does not provide a dockable UI widget."""
        return None


class ProviderExerciseHandler:
    """Launch a provider exercise card through the contained dashboard.

    Provider model-pack entries identify an exercise *directory* containing
    model-builder source.  A directory cannot be passed to the generic local
    file launch flow.  Route each supported interchange format to the shared
    exercise dashboard and preselect the engine that owns that format.
    """

    _PREFERRED_ENGINE_BY_MODEL_TYPE = {
        "mjcf": "MuJoCo_Models",
        "sdformat-1.8": "Drake_Models",
        "urdf": "Pinocchio_Models",
        "osim": "OpenSim_Models",
    }
    MODEL_TYPES = set(_PREFERRED_ENGINE_BY_MODEL_TYPE)

    def can_handle(self, model_type: str) -> bool:
        """Return whether the model format is a provider exercise format."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the model's exercise in the engine-aware dashboard."""
        if repo_path is None:
            raise ValueError("repo_path must be provided")

        model_type = getattr(model, "type", None)
        if not isinstance(model_type, str):
            logger.error("ProviderExerciseHandler: model type is missing")
            return False
        preferred_engine = self._PREFERRED_ENGINE_BY_MODEL_TYPE.get(model_type.lower())
        if preferred_engine is None:
            logger.error(
                "ProviderExerciseHandler: unsupported model type %s", model_type
            )
            return False

        model_path = getattr(model, "path", None)
        if not isinstance(model_path, str) or not model_path.strip():
            logger.error(
                "ProviderExerciseHandler: model '%s' has no exercise path",
                getattr(model, "id", "unknown"),
            )
            return False

        try:
            resolved_dir = resolve_model_artifact_path(model, repo_path)
            if not resolved_dir.is_dir():
                logger.error(
                    "ProviderExerciseHandler: model path is not a directory: %s",
                    resolved_dir,
                )
                return False
            exercise_name = resolved_dir.name
        except (ValueError, OSError) as exc:
            logger.error(
                "ProviderExerciseHandler: invalid exercise path %s: %s", model_path, exc
            )
            return False

        env = process_manager.get_subprocess_env(
            get_model_python_paths(model, repo_path)
        )
        env["BIOMECH_EXERCISE"] = exercise_name
        env["BIOMECH_ENGINE"] = preferred_engine
        process = process_manager.launch_script(
            name=getattr(model, "name", f"Biomechanics Exercise: {exercise_name}"),
            script_path=repo_path / "src" / "launchers" / "exercise_dashboard.py",
            cwd=repo_path,
            env=env,
            extra_python_paths=get_model_python_paths(model, repo_path),
            keep_terminal_open=True,
        )
        return process is not None

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """Provider exercise dashboards launch in their own process."""
        return None


class GolfSimulationSuiteHandler:
    """Handler for launching the Golf Simulation Suite.

    Design by Contract:
        Precondition: model.path must point to launch_golf_suite.py
        Postcondition: Golf Simulation Suite is launched
    """

    MODEL_TYPES = {"golf_simulation"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch the Golf Simulation Suite.

        Args:
            model: Model configuration with 'path' attr.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error("GolfSimulationSuiteHandler: model has no path")
            return False

        script_path = resolve_model_artifact_path(model, repo_path)
        if not script_path.exists():
            logger.warning(
                "GolfSimulationSuiteHandler: script not found: %s", script_path
            )
            return False

        process = process_manager.launch_script(
            name="Golf Simulation Suite",
            script_path=script_path,
            cwd=repo_path,
            extra_python_paths=get_model_python_paths(model, repo_path),
            keep_terminal_open=True,
        )
        return process is not None

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """Golf Simulation Suite handler does not provide a dockable UI widget."""
        return None


def _open_with_system_app(file_path: Path, handler_name: str) -> bool:
    """Open a file with the system default application.

    DRY helper: eliminates the duplicated platform-detection code in
    MatlabFileHandler and DocumentHandler.

    Args:
        file_path: Path to the file to open.
        handler_name: Name of the calling handler (for log messages).

    Returns:
        True if the file was opened, False otherwise.
    """
    try:
        if platform.system() == "Windows":
            os.startfile(str(file_path))  # type: ignore[attr-defined]  # noqa: S606
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(file_path)])  # noqa: S603, S607
        else:
            subprocess.Popen(["xdg-open", str(file_path)])  # noqa: S603, S607
        logger.info("%s: opened %s", handler_name, file_path.name)
        return True
    except (FileNotFoundError, PermissionError, OSError):
        logger.exception("%s: failed to open %s", handler_name, file_path)
        return False


class _SystemFileHandler:
    """Base handler for opening files with system applications.

    DRY base for MatlabFileHandler and DocumentHandler.
    """

    MODEL_TYPES: set[str] = set()
    HANDLER_NAME: str = "SystemFileHandler"

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Open a file with the system default application."""
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error(
                "%s: model '%s' has no path",
                self.HANDLER_NAME,
                getattr(model, "id", "unknown"),
            )
            return False

        file_path = resolve_model_artifact_path(model, repo_path)
        if not file_path.exists():
            logger.warning("%s: file not found: %s", self.HANDLER_NAME, file_path)
            return False

        return _open_with_system_app(file_path, self.HANDLER_NAME)

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """System file handlers do not provide a dockable UI widget."""
        return None


class SharedRepoHandler:
    """Handler for opening sibling repositories/folders.

    Handles model types: shared_repo
    Covers: MuJoCo_Models, Drake_Models, Pinocchio_Models, OpenSim_Models.
    """

    MODEL_TYPES = {"shared_repo"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Open the sibling repository directory in the default system file manager."""
        if repo_path is None:
            raise ValueError("repo_path must be provided")

        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error(
                "SharedRepoHandler: model '%s' has no path",
                getattr(model, "id", "unknown"),
            )
            return False

        # Find sibling folder
        folder_path = repo_path.parent / model_path
        if not folder_path.exists():
            logger.warning("SharedRepoHandler: directory not found: %s", folder_path)
            return False

        return _open_with_system_app(folder_path, "SharedRepoHandler")

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """Shared repository handler does not provide a dockable UI widget."""
        return None


class MatlabFileHandler(_SystemFileHandler):
    """Handler for opening MATLAB files (.slx, .m) with system MATLAB."""

    MODEL_TYPES = {"matlab_file"}
    HANDLER_NAME = "MatlabFileHandler"


class DocumentHandler:
    """Handler for opening document files (.md, .pdf, etc.) securely using a proxy runner."""

    MODEL_TYPES = {"document"}
    HANDLER_NAME = "DocumentHandler"

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: Any,
    ) -> bool:
        """Open a document securely using the approved document_proxy."""
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        model_path = getattr(model, "path", None) or ""
        if not model_path:
            logger.error(
                "%s: model '%s' has no path",
                self.HANDLER_NAME,
                getattr(model, "id", "unknown"),
            )
            return False

        file_path = resolve_model_artifact_path(model, repo_path)
        if not file_path.exists():
            logger.warning("%s: file not found: %s", self.HANDLER_NAME, file_path)
            return False

        proxy_script = repo_path / "src" / "launchers" / "document_proxy.py"
        if not proxy_script.exists():
            logger.error(
                "%s: Proxy script not found: %s", self.HANDLER_NAME, proxy_script
            )
            return False

        try:
            from src.shared.python.security.secure_subprocess import secure_popen
            import sys

            process = secure_popen(
                [sys.executable, str(proxy_script), str(file_path)],
                cwd=str(repo_path),
                suite_root=repo_path,
            )
            # Attach the process so it can be tracked/stopped if necessary, though it likely exits immediately.
            process_manager.attach_process(f"Document_{file_path.name}", process)
            logger.info("%s: launched proxy for %s", self.HANDLER_NAME, file_path.name)
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(
                "%s: failed to launch document proxy: %s", self.HANDLER_NAME, e
            )
            return False

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """Document handler does not provide a dockable UI widget."""
        return None


class PhysicsInformedHandler:
    """Handler for the ``physics_informed`` (PINN) tiles.

    The physics-informed models in
    ``src/shared/python/physics_informed/`` are a **library-only** feature
    (epic #5419): ``PhysicsMode`` + ``create_model`` are importable, but no
    interactive front-end has been built yet. Before issue #7984 no handler
    was registered for ``type: physics_informed`` at all, so clicking either
    tile produced a generic "Unknown launch type" toast that read like a
    configuration typo. Reporting the real reason — and the real dependency
    state — is the honest failure this replaces it with.
    """

    MODEL_TYPES = {"physics_informed"}
    PACKAGE = "src/shared/python/physics_informed"

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def status_message(self, model: Any) -> str:
        """Return a user-facing explanation of why this tile cannot launch."""
        mode = getattr(model, "mode", None) or "unknown"
        name = getattr(model, "name", None) or getattr(model, "id", "unknown")
        return (
            f"{name} (mode={mode}) has no interactive UI yet. The physics-informed "
            f"models are available as a library only ({self.PACKAGE}/); see epic "
            "#5419. Tracking issue for a launcher front-end: #7984."
        )

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Report the missing front-end instead of failing silently."""
        if repo_path is None:
            raise ValueError("repo_path must be provided")
        logger.error("PhysicsInformedHandler: %s", self.status_message(model))
        return False

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """No dockable UI exists for physics-informed models yet."""
        return None


class ApiBackedHandler:
    """Handler for API-backed tiles that do not launch local processes directly."""

    MODEL_TYPES = {"api_backed"}

    def can_handle(self, model_type: str) -> bool:
        """Check if this handler supports the model type."""
        return model_type.lower() in self.MODEL_TYPES

    def launch(
        self,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """API-backed tiles do not launch local processes directly."""
        logger.info(
            "ApiBackedHandler: launch requested for api-backed tile %s",
            getattr(model, "id", "unknown"),
        )
        return True

    def get_dockable_ui(self, model: Any, repo_path: Path) -> Any | None:
        """API-backed handler does not provide a dockable UI widget."""
        return None


# ============================================================
# Handler Registry Table (DRY: data-driven registration)
# ============================================================

_MODULE_HANDLERS = [
    ModuleHandler(
        model_types={"humanoid_mujoco", "humanoid", "custom_humanoid"},
        module_name="src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf",
        display_name="MuJoCo Humanoid Golf",
    ),
    ModuleHandler(
        model_types={"comprehensive", "comprehensive_mujoco", "custom_dashboard"},
        module_name="src.engines.physics_engines.mujoco.python.humanoid_launcher",
        display_name="Comprehensive Golf Model",
    ),
    ModuleHandler(
        model_types={"drake", "drake_golf"},
        module_name="src.engines.physics_engines.drake.python.src.drake_gui_app",
        display_name="Drake Golf Model",
    ),
]

# NOTE (#8030): ``script_path`` here is only the fallback used when a model
# declares no ``path``. ``models.yaml`` wins — see ``ScriptHandler.resolve_script``.
# These fallbacks were pointing at files that have never existed
# (``pinocchio_golf/main.py``, ``opensim_golf.py``, and a whole ``myosim/``
# directory), which silently killed the Pinocchio/OpenSim/MyoSuite tiles.
_SCRIPT_HANDLERS = [
    ScriptHandler(
        model_types={"pinocchio", "pinocchio_golf"},
        script_path="src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py",
        display_name="Pinocchio Golf Model",
        cwd_path="src/engines/physics_engines/pinocchio/python",
    ),
    ScriptHandler(
        model_types={"opensim", "opensim_golf"},
        script_path="src/engines/physics_engines/opensim/python/opensim_gui.py",
        display_name="OpenSim Golf Model",
        cwd_path="src/engines/physics_engines/opensim/python",
    ),
    ScriptHandler(
        model_types={"myosim", "myosim_golf", "musculoskeletal"},
        script_path="src/engines/physics_engines/myosuite/python/gui.py",
        display_name="MyoSuite Golf Model",
        cwd_path="src/engines/physics_engines/myosuite/python",
    ),
    ScriptHandler(
        model_types={"openpose", "pose_estimation"},
        script_path="src/shared/python/pose_estimation/openpose_gui.py",
        display_name="OpenPose",
    ),
]


# Backward-compatible aliases for the old per-engine handler classes.
# These were replaced by the data-driven ModuleHandler/ScriptHandler tables
# but some tests import them by name.
HumanoidMuJoCoHandler = type(
    "HumanoidMuJoCoHandler",
    (ModuleHandler,),
    {
        "__init__": lambda self: ModuleHandler.__init__(
            self,
            model_types={"humanoid_mujoco", "humanoid", "custom_humanoid"},
            module_name="src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf",
            display_name="MuJoCo Humanoid Golf",
        ),
    },
)


class ModelHandlerRegistry:
    """Registry for model launch handlers.

    This class implements the Strategy pattern for model launching,
    allowing new handlers to be added without modifying existing code.
    """

    def __init__(self) -> None:
        """Initialize the handler registry with default handlers."""
        self._handlers: list[ModelHandler] = [
            *_MODULE_HANDLERS,
            *_SCRIPT_HANDLERS,
            SpecialAppHandler(),
            PuttingGreenHandler(),
            BiomechExerciseHandler(),
            ProviderExerciseHandler(),
            GolfSimulationSuiteHandler(),
            MatlabFileHandler(),
            SharedRepoHandler(),
            DocumentHandler(),
            PhysicsInformedHandler(),
            ApiBackedHandler(),
        ]

    def register_handler(self, handler: ModelHandler) -> None:
        """Register a new model handler.

        Args:
            handler: Handler instance implementing the ModelHandler protocol.
        """
        self._handlers.append(handler)

    def get_handler(self, model_type: str) -> ModelHandler | None:
        """Get a handler that can launch the given model type.

        Args:
            model_type: The type of model to launch.

        Returns:
            A handler that can launch the model, or None if not found.
        """
        if model_type is None:
            raise ValueError("model_type must be provided")
        for handler in self._handlers:
            if handler.can_handle(model_type):
                return handler
        return None

    def launch_model(
        self,
        model_type: str,
        model: Any,
        repo_path: Path,
        process_manager: ProcessManager,
    ) -> bool:
        """Launch a model using the appropriate handler.

        Args:
            model_type: The type of model to launch.
            model: Model configuration object.
            repo_path: Path to the repository root.
            process_manager: Process manager for subprocess handling.

        Returns:
            True if launch succeeded, False otherwise.
        """
        if model_type is None:
            raise ValueError("model_type must be provided")
        handler = self.get_handler(model_type)
        if handler is None:
            logger.warning(f"No handler found for model type: {model_type}")
            return False

        return handler.launch(model, repo_path, process_manager)
