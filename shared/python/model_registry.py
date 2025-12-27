"""Model Registry for managing physics models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Configuration for a physics model."""

    id: str
    name: str
    description: str
    type: str  # 'mjcf', 'urdf', 'matlab'
    path: str


class ModelRegistry:
    """Registry for loading and accessing model configurations."""

    def __init__(self, config_path: str | Path = "config/models.yaml"):
        """Initialize registry.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.models: dict[str, ModelConfig] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load models from YAML file."""
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path) as f:
                data = yaml.safe_load(f)

            if not data or "models" not in data:
                return

            for model_data in data["models"]:
                model = ModelConfig(**model_data)
                self.models[model.id] = model

        except Exception as e:
            print(f"Error loading model registry: {e}")

    def get_model(self, model_id: str) -> ModelConfig | None:
        """Get model by ID."""
        return self.models.get(model_id)

    def get_all_models(self) -> list[ModelConfig]:
        """Get all registered models."""
        return list(self.models.values())

    def get_models_by_type(self, model_type: str) -> list[ModelConfig]:
        """Get models of a specific type (e.g., 'mjcf')."""
        return [m for m in self.models.values() if m.type == model_type]
