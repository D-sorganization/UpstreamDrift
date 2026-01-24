"""Model registry for managing physics models and special applications."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelSpec:
    """Specification for a launchable model or application."""

    id: str
    name: str
    description: str
    type: str  # 'engine_managed' or 'special_app'
    path: str
    engine_type: str | None = None


class ModelRegistry:
    """Registry for managing available models."""

    def __init__(self, config_path: str = "config/models.yaml"):
        self.config_path = Path(config_path)
        self.models: list[ModelSpec] = []
        self._loaded = False

    def load(self, root_path: Path) -> None:
        """Load model registry from configuration file.

        Args:
            root_path: Root directory of the repository to resolve relative paths.
        """
        full_config_path = root_path / self.config_path
        if not full_config_path.exists():
            logger.warning(f"Model config not found at {full_config_path}")
            return

        try:
            with open(full_config_path) as f:
                data = yaml.safe_load(f)

            for item in data.get("models", []):
                self.models.append(ModelSpec(**item))

            self._loaded = True
            logger.info(f"Loaded {len(self.models)} models from registry")

        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")

    def get_all_models(self) -> list[ModelSpec]:
        """Get all registered models."""
        return self.models

    def get_model_by_id(self, model_id: str) -> ModelSpec | None:
        """Get a specific model by ID."""
        for model in self.models:
            if model.id == model_id:
                return model
        return None


# Singleton instance
_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return _registry
