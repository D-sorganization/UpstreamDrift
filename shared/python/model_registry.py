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
        """Load models from YAML configuration file.

        Raises:
            ModelRegistryError: If registry file is malformed (NotRaised: gracefully logged)

        This method logs warnings and errors if the registry file is missing,
        malformed, or individual model configurations are invalid, and leaves
        the registry in its current state instead of raising exceptions.
        """
        from .core import setup_logging

        logger = setup_logging(__name__)

        if not self.config_path.exists():
            logger.warning(f"Model registry not found: {self.config_path}")
            return

        try:
            with open(self.config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning(f"Empty model registry: {self.config_path}")
                return

            if "models" not in data:
                logger.error(
                    f"Invalid registry format: missing 'models' key in {self.config_path}"
                )
                return

            for model_data in data["models"]:
                try:
                    model = ModelConfig(**model_data)
                    self.models[model.id] = model
                    logger.debug(f"Loaded model: {model.id}")
                except TypeError as e:
                    logger.error(f"Invalid model configuration: {model_data} - {e}")

            logger.info(f"Loaded {len(self.models)} models from {self.config_path}")

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {self.config_path}: {e}")
        except OSError as e:
            logger.error(f"Failed to read registry file {self.config_path}: {e}")

    def get_model(self, model_id: str) -> ModelConfig | None:
        """
        Get model configuration by its unique ID.

        Args:
            model_id: The unique identifier of the model.

        Returns:
            The model configuration object, or None if not found.
        """
        return self.models.get(model_id)

    def get_all_models(self) -> list[ModelConfig]:
        """
        Retrieve all registered models.

        Returns:
            A list of all ModelConfig objects in the registry.
        """
        return list(self.models.values())

    def get_models_by_type(self, model_type: str) -> list[ModelConfig]:
        """
        Retrieve all models of a specific type (e.g., 'mjcf', 'urdf').

        Args:
            model_type: The type string to filter by.

        Returns:
            A list of ModelConfig objects matching the specified type.
        """
        return [m for m in self.models.values() if m.type == model_type]
