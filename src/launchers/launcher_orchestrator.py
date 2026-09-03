"""Domain-state orchestration for the UpstreamDrift launcher."""

from __future__ import annotations

from typing import Any

from src.launchers.launcher_constants import (
    REPOS_ROOT,
    _lazy_load_engine_manager,
    _lazy_load_model_registry,
    logger,
)
from src.launchers.ui_components import StartupResults


class LauncherOrchestrator:
    """Coordinate launcher registry, engine, Docker, and model state."""

    def __init__(self) -> None:
        self.registry = None
        self.engine_manager = None
        self.docker_available = False
        self.available_models: dict[str, Any] = {}
        self.special_app_lookup: dict[str, Any] = {}

    def initialize_from_results(self, startup_results: StartupResults | None) -> None:
        """Initialize domain state from async startup results."""
        self.docker_available = (
            startup_results.docker_available if startup_results else False
        )
        self.init_registry(startup_results)
        self.init_engine_manager(startup_results)
        self.build_available_models()

    def init_registry(self, startup_results: StartupResults | None) -> None:
        """Initialize the model registry from startup results or lazy fallback."""
        if startup_results and startup_results.registry is not None:
            self.registry = startup_results.registry
            logger.info("Using pre-loaded model registry from async startup")
            return

        try:
            registry_cls = _lazy_load_model_registry()
            self.registry = registry_cls(REPOS_ROOT / "src/config/models.yaml")
        except ImportError as exc:
            logger.error("Failed to load ModelRegistry: %s", exc)
            self.registry = None

    def init_engine_manager(self, startup_results: StartupResults | None) -> None:
        """Initialize the engine manager from startup results or lazy fallback."""
        if startup_results and startup_results.engine_manager is not None:
            self.engine_manager = startup_results.engine_manager
            logger.info("Using pre-loaded engine manager from async startup")
            return

        try:
            engine_manager_cls, _ = _lazy_load_engine_manager()
            self.engine_manager = engine_manager_cls(REPOS_ROOT)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Failed to initialize EngineManager: %s", exc)
            self.engine_manager = None

    def build_available_models(self) -> None:
        """Collect all known models and auxiliary applications."""
        logger.debug("Building available models from registry...")
        self.available_models.clear()
        self.special_app_lookup.clear()

        if not self.registry:
            logger.warning("No registry available - no models will be loaded")
            return

        all_models = self.registry.get_all_models()
        logger.info("Registry returned %s models", len(all_models))
        all_models = self._restrict_to_desktop_surface(all_models)

        for model in all_models:
            self.available_models[model.id] = model
            logger.debug("  Added model: %s (%s)", model.id, model.name)
            if model.type in ("special_app", "utility", "matlab_app"):
                self.special_app_lookup[model.id] = model

        logger.info(
            "Built available_models with %s entries",
            len(self.available_models),
        )

    @staticmethod
    def _restrict_to_desktop_surface(models: list[Any]) -> list[Any]:
        """Keep only tiles declared for the ``pyqt`` surface (issue #9412).

        ``src/config/models.yaml`` is the single registry for both launchers;
        its ``surfaces`` field decides where a tile appears. The same pass
        registers every desktop tile with the shared ``GUIRegistry`` so the
        Tools-owned registry is derived from the registry file, not
        maintained by hand. Failures degrade to the unfiltered list rather
        than an empty launcher.
        """
        try:
            from src.config.registry_adapters import (
                filter_desktop_models,
                populate_gui_registry,
            )
            from src.config.tile_registry import load_tile_registry

            tile_registry = load_tile_registry()
            kept = filter_desktop_models(models, tile_registry)
            populate_gui_registry(tile_registry)
        except Exception as exc:  # noqa: BLE001 - launcher must still start
            logger.warning("Tile registry surface filter unavailable: %s", exc)
            return models
        if len(kept) != len(models):
            logger.info(
                "Desktop surface shows %s of %s registry models", len(kept), len(models)
            )
        return kept

    def get_model(self, model_id: str) -> Any | None:
        """Retrieve a model or application by ID."""
        if model_id is None:
            raise ValueError("model_id must be provided")
        if model_id in self.available_models:
            return self.available_models[model_id]
        if self.registry:
            return self.registry.get_model(model_id)
        return None
