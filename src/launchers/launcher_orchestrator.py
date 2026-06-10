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

        for model in all_models:
            self.available_models[model.id] = model
            logger.debug("  Added model: %s (%s)", model.id, model.name)
            if model.type in ("special_app", "utility", "matlab_app"):
                self.special_app_lookup[model.id] = model

        logger.info(
            "Built available_models with %s entries",
            len(self.available_models),
        )

    def get_model(self, model_id: str) -> Any | None:
        """Retrieve a model or application by ID."""
        if model_id is None:
            raise ValueError("model_id must be provided")
        if model_id in self.available_models:
            return self.available_models[model_id]
        if self.registry:
            return self.registry.get_model(model_id)
        return None
