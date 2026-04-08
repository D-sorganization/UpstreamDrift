"""Model Registry for managing physics models."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from src.shared.python.config.model_pack_manifest import (
    ModelPackEntry,
    ModelPackManifest,
)
from src.shared.python.core.contracts import ContractChecker


def _normalize_legacy_model_entry(model_data: dict[str, Any]) -> dict[str, Any]:
    """Coerce legacy registry entries into the stricter manifest contract shape."""
    normalized = dict(model_data)
    description = normalized.get("description")
    if isinstance(description, str) and description.strip() == "":
        fallback_name = normalized.get("name")
        if isinstance(fallback_name, str) and fallback_name.strip():
            normalized["description"] = fallback_name.strip()
    return normalized


@dataclass
class ModelConfig:
    """Configuration for a physics model."""

    id: str
    name: str
    description: str
    type: str  # 'mjcf', 'urdf', 'matlab'
    path: str
    engine_type: str | None = None
    provider: str | None = None
    source_root: str | None = None
    working_dir: str | None = None
    python_paths: tuple[str, ...] = ()


class ModelRegistry(ContractChecker):
    """Registry for loading and accessing model configurations.

    Design by Contract:
        Invariants:
            - models dict is never None
            - config_path is a valid Path object
            - All model IDs in the registry are non-empty strings
    """

    def __init__(self, config_path: str | Path = "config/models.yaml") -> None:
        """Initialize registry.

        Args:
            config_path: Path to the YAML configuration file.
        """
        if not (config_path is not None):
            raise ValueError("config_path must be provided")
        self.config_path = Path(config_path)
        self.models: dict[str, ModelConfig] = {}
        self._load_registry()

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Define class invariants for ModelRegistry."""
        return [
            (
                lambda: self.models is not None and isinstance(self.models, dict),
                "models must be a non-None dict",
            ),
            (
                lambda: (
                    self.config_path is not None and isinstance(self.config_path, Path)
                ),
                "config_path must be a valid Path",
            ),
            (
                lambda: all(isinstance(k, str) and len(k) > 0 for k in self.models),
                "All model IDs must be non-empty strings",
            ),
        ]

    def _load_registry(self) -> None:
        """Load models from YAML configuration file.

        Raises:
            ModelRegistryError: If registry file is malformed (NotRaised: gracefully logged)

        This method logs warnings and errors if the registry file is missing,
        malformed, or individual model configurations are invalid, and leaves
        the registry in its current state instead of raising exceptions.
        """
        from ..core import setup_logging

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
                    if not isinstance(model_data, dict):
                        raise ValueError("legacy model entries must be mappings")
                    legacy_model_data = cast(dict[str, Any], model_data)
                    entry = ModelPackEntry.from_dict(
                        _normalize_legacy_model_entry(legacy_model_data)
                    )
                    model = self._build_model_config(entry)
                    self.models[model.id] = model
                    logger.debug(f"Loaded model: {model.id}")
                except (TypeError, ValueError) as e:
                    logger.error(f"Invalid model configuration: {model_data} - {e}")

            self._load_provider_manifests()
            logger.info(f"Loaded {len(self.models)} models from {self.config_path}")

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {self.config_path}: {e}")
            raise
        except OSError as e:
            logger.error(f"Failed to read registry file {self.config_path}: {e}")
            raise

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

    def _build_model_config(
        self,
        entry: ModelPackEntry,
        *,
        provider: str | None = None,
        source_root: str | None = None,
    ) -> ModelConfig:
        """Convert a strict model-pack entry into the runtime config shape."""
        return ModelConfig(
            id=entry.id,
            name=entry.name,
            description=entry.description,
            type=entry.type,
            path=entry.path,
            engine_type=entry.engine_type,
            provider=entry.provider or provider,
            source_root=entry.source_root or source_root,
            working_dir=entry.working_dir,
            python_paths=entry.python_paths,
        )

    def _iter_provider_manifest_paths(self) -> list[Path]:
        """Discover external provider manifests from configured provider roots."""
        env_value = os.environ.get("UPSTREAM_DRIFT_PROVIDER_ROOTS", "").strip()
        if not env_value:
            return []

        config_root = self.config_path.parent
        manifest_paths: list[Path] = []
        seen: set[Path] = set()

        for raw_root in env_value.split(os.pathsep):
            root_value = raw_root.strip()
            if not root_value:
                continue

            root_path = Path(root_value)
            if not root_path.is_absolute():
                root_path = (config_root / root_path).resolve()

            candidates = [
                root_path / "model_pack.yaml",
                root_path / "model_pack.yml",
                root_path / ".upstreamdrift" / "model_pack.yaml",
                root_path / ".upstreamdrift" / "model_pack.yml",
            ]
            for candidate in candidates:
                if candidate in seen or not candidate.exists():
                    continue
                seen.add(candidate)
                manifest_paths.append(candidate)

        return manifest_paths

    def _load_provider_manifests(self) -> None:
        """Load external provider manifests configured for the launcher migration."""
        from ..core import setup_logging

        logger = setup_logging(__name__)

        for manifest_path in self._iter_provider_manifest_paths():
            try:
                manifest = ModelPackManifest.load(manifest_path)
                for entry in manifest.models:
                    if entry.id in self.models:
                        logger.warning(
                            "Skipping duplicate provider model id '%s' from %s",
                            entry.id,
                            manifest_path,
                        )
                        continue
                    self.models[entry.id] = self._build_model_config(
                        entry,
                        provider=manifest.provider,
                        source_root=str(manifest_path.parent),
                    )
                logger.info(
                    "Loaded %d provider models from %s",
                    len(manifest.models),
                    manifest_path,
                )
            except (OSError, ValueError, yaml.YAMLError) as e:
                logger.warning(
                    "Skipping provider manifest %s due to load failure: %s",
                    manifest_path,
                    e,
                )
