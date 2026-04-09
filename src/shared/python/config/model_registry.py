"""Model Registry for managing physics models."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from src.shared.python.config.model_pack_manifest import (
    CrossEngineIdentity,
    ExchangeArtifact,
    LauncherPresentationMetadata,
    ModelPackEntry,
    ModelPackManifest,
    ProvenanceMetadata,
)
from src.shared.python.config.model_source_providers import (
    ResolvedModelSource,
    collect_engine_provider_paths,
    resolve_model_source,
)
from src.shared.python.config.provider_catalog import iter_provider_manifest_specs
from src.shared.python.core.contracts import ContractChecker, require

_DISCOVERY_MODES = {"local-only", "hybrid", "provider-first"}


def _normalize_legacy_model_entry(model_data: dict[str, Any]) -> dict[str, Any]:
    """Coerce legacy registry entries into the stricter manifest contract shape."""
    normalized = dict(model_data)
    description = normalized.get("description")
    if isinstance(description, str) and description.strip() == "":
        fallback_name = normalized.get("name")
        if isinstance(fallback_name, str) and fallback_name.strip():
            normalized["description"] = fallback_name.strip()
    return normalized


def _normalize_discovery_mode(raw_value: str | None) -> str:
    """Return the configured provider-discovery mode."""
    if raw_value is None:
        return "hybrid"

    mode = raw_value.strip().lower()
    require(mode in _DISCOVERY_MODES, "invalid discovery mode", mode)
    return mode


@dataclass
class ModelConfig:
    """Configuration for a physics model."""

    id: str
    name: str
    description: str
    type: str  # 'mjcf', 'urdf', 'matlab'
    path: str
    engine_type: str | None = None
    capabilities: tuple[str, ...] = ()
    provider: str | None = None
    source_root: str | None = None
    working_dir: str | None = None
    python_paths: tuple[str, ...] = ()
    identity: CrossEngineIdentity | None = None
    exchange_artifacts: tuple[ExchangeArtifact, ...] = ()
    provenance: ProvenanceMetadata | None = None
    launcher: LauncherPresentationMetadata | None = None
    order: int = 99


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
        self.discovery_mode = _normalize_discovery_mode(
            os.environ.get("UPSTREAM_DRIFT_DISCOVERY_MODE")
        )
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
            (
                lambda: self.discovery_mode in _DISCOVERY_MODES,
                "discovery_mode must be one of local-only, hybrid, provider-first",
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

            models_raw = data["models"]
            if self.discovery_mode == "provider-first":
                self._load_provider_manifests()
                self._load_legacy_models(models_raw, overwrite_existing=False)
            else:
                self._load_legacy_models(models_raw, overwrite_existing=True)
                if self.discovery_mode == "hybrid":
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

    def resolve_model_source(
        self,
        model_id: str,
        default_root: Path,
        *,
        approved_roots: tuple[Path, ...] = (),
        fallback_relative: str | Path | None = None,
    ) -> ResolvedModelSource:
        """Resolve canonical source paths for a registered model."""
        require(
            isinstance(model_id, str) and bool(model_id.strip()),
            "invalid model id",
            model_id,
        )
        model = self.get_model(model_id)
        require(model is not None, "model id not found", model_id)
        return resolve_model_source(
            model,
            default_root,
            approved_roots=approved_roots,
            fallback_relative=fallback_relative,
        )

    def get_engine_provider_paths(
        self,
        default_root: Path,
        *,
        approved_roots: tuple[Path, ...] = (),
    ) -> dict[str, tuple[Path, ...]]:
        """Collect resolved provider validation paths grouped by engine type."""
        return collect_engine_provider_paths(
            self.models.values(),
            default_root,
            approved_roots=approved_roots,
        )

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
            capabilities=entry.capabilities,
            provider=entry.provider or provider,
            source_root=entry.source_root or source_root,
            working_dir=entry.working_dir,
            python_paths=entry.python_paths,
            identity=entry.identity,
            exchange_artifacts=entry.exchange_artifacts,
            provenance=entry.provenance,
            launcher=entry.launcher,
            order=entry.order,
        )

    def _load_legacy_models(
        self,
        models_raw: Any,
        *,
        overwrite_existing: bool,
    ) -> None:
        """Load legacy models.yaml entries with explicit overwrite policy."""
        from ..core import setup_logging

        logger = setup_logging(__name__)
        require(
            isinstance(models_raw, list), "legacy models must be a list", models_raw
        )

        for model_data in models_raw:
            try:
                if not isinstance(model_data, dict):
                    raise ValueError("legacy model entries must be mappings")
                legacy_model_data = cast(dict[str, Any], model_data)
                entry = ModelPackEntry.from_dict(
                    _normalize_legacy_model_entry(legacy_model_data)
                )
                if not overwrite_existing and entry.id in self.models:
                    logger.info(
                        "Keeping existing provider-backed model '%s' over legacy duplicate",
                        entry.id,
                    )
                    continue
                model = self._build_model_config(entry)
                self.models[model.id] = model
                logger.debug(f"Loaded model: {model.id}")
            except (TypeError, ValueError) as e:
                logger.error(f"Invalid model configuration: {model_data} - {e}")

    def _iter_provider_manifest_specs(self) -> tuple[tuple[Path, Path], ...]:
        """Discover external provider manifests from env and sibling repos."""
        return iter_provider_manifest_specs(
            self.config_path,
            os.environ.get("UPSTREAM_DRIFT_PROVIDER_ROOTS"),
        )

    def _load_provider_manifests(self) -> None:
        """Load external provider manifests configured for the launcher migration."""
        from ..core import setup_logging

        logger = setup_logging(__name__)

        for provider_root, manifest_path in self._iter_provider_manifest_specs():
            try:
                manifest = ModelPackManifest.load(manifest_path)
                for entry in manifest.models:
                    if (
                        entry.id in self.models
                        and self.discovery_mode != "provider-first"
                    ):
                        logger.warning(
                            "Skipping duplicate provider model id '%s' from %s",
                            entry.id,
                            manifest_path,
                        )
                        continue
                    if (
                        entry.id in self.models
                        and self.discovery_mode == "provider-first"
                    ):
                        logger.info(
                            "Provider-first mode overriding legacy model '%s' from %s",
                            entry.id,
                            manifest_path,
                        )
                    self.models[entry.id] = self._build_model_config(
                        entry,
                        provider=manifest.provider,
                        source_root=str(provider_root),
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
