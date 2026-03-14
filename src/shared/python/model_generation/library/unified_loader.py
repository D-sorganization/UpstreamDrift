"""
Unified model loader supporting URDF and MJCF formats.

Provides a single entry point for loading models from any supported
format, with automatic format detection and standardized output.
Supports user-configurable default model preferences.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from model_generation.converters.mjcf_converter import MJCFConverter
from model_generation.converters.urdf_parser import ParsedModel, URDFParser
from model_generation.core.contracts import postcondition, precondition

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported model file formats."""

    URDF = "urdf"
    MJCF = "mjcf"
    UNKNOWN = "unknown"


# Map of file extensions to formats
_EXTENSION_MAP: dict[str, ModelFormat] = {
    ".urdf": ModelFormat.URDF,
    ".xacro": ModelFormat.URDF,
    ".xml": ModelFormat.MJCF,
    ".mjcf": ModelFormat.MJCF,
}


@dataclass
class LoadResult:
    """Result of loading a model file."""

    model: ParsedModel | None = None
    source_path: Path | None = None
    source_format: ModelFormat = ModelFormat.UNKNOWN
    success: bool = False
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Model name from parsed model or filename."""
        if self.model:
            return str(self.model.name)
        if self.source_path:
            return self.source_path.stem
        return "unknown"


@dataclass
class UserPreferences:
    """User preferences for model explorer, persisted to disk."""

    default_model_id: str = "mujoco_humanoid"
    recent_models: list[str] = field(default_factory=list)
    max_recent: int = 10

    # Display preferences
    show_segments: bool = True
    show_joints: bool = True
    show_collisions: bool = True
    show_inertias: bool = True
    show_frames: bool = False

    def add_recent(self, model_id: str) -> None:
        """Add a model to the recent list."""
        assert model_id is not None, "model_id must be provided"
        assert model_id is not None, "model_id must be provided"
        if model_id in self.recent_models:
            self.recent_models.remove(model_id)
        self.recent_models.insert(0, model_id)
        self.recent_models = self.recent_models[: self.max_recent]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "default_model_id": self.default_model_id,
            "recent_models": self.recent_models,
            "show_segments": self.show_segments,
            "show_joints": self.show_joints,
            "show_collisions": self.show_collisions,
            "show_inertias": self.show_inertias,
            "show_frames": self.show_frames,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserPreferences:
        """Deserialize from dictionary."""
        return cls(
            default_model_id=data.get("default_model_id", "mujoco_humanoid"),
            recent_models=data.get("recent_models", []),
            show_segments=data.get("show_segments", True),
            show_joints=data.get("show_joints", True),
            show_collisions=data.get("show_collisions", True),
            show_inertias=data.get("show_inertias", True),
            show_frames=data.get("show_frames", False),
        )


def detect_format(file_path: Path) -> ModelFormat:
    """
    Detect model format from file extension and content.

    Args:
        file_path: Path to the model file.

    Returns:
        Detected ModelFormat.
    """
    suffix = file_path.suffix.lower()
    if suffix in _EXTENSION_MAP:
        fmt = _EXTENSION_MAP[suffix]
        # For .xml files, peek at content to distinguish MJCF from other XML
        if suffix == ".xml" and file_path.exists():
            try:
                content = file_path.read_text(errors="replace")[:500]
                if "<robot" in content:
                    return ModelFormat.URDF
                if "<mujoco" in content:
                    return ModelFormat.MJCF
            except OSError:
                pass
        return fmt
    return ModelFormat.UNKNOWN


class UnifiedModelLoader:
    """
    Loads models from URDF or MJCF format into a common ParsedModel.

    Provides:
    - Automatic format detection
    - MJCF-to-ParsedModel conversion
    - Bundled library access
    - User preference management for default models

    Invariant: All loaded models are returned as ParsedModel regardless
    of source format, ensuring downstream code works uniformly.
    """

    _PREFS_FILENAME = "model_explorer_prefs.json"

    def __init__(self, prefs_dir: Path | None = None):
        """
        Initialize the unified loader.

        Args:
            prefs_dir: Directory for storing user preferences.
                       Defaults to ~/.model_generation/
        """
        self._urdf_parser = URDFParser()
        self._mjcf_converter = MJCFConverter()
        self._prefs_dir = prefs_dir or Path.home() / ".model_generation"
        self._prefs_dir.mkdir(parents=True, exist_ok=True)
        self._preferences = self._load_preferences()
        self._bundled_manifest: dict[str, Any] | None = None

    # -- Preferences --

    @property
    def preferences(self) -> UserPreferences:
        """Current user preferences."""
        return self._preferences

    def _prefs_path(self) -> Path:
        return self._prefs_dir / self._PREFS_FILENAME

    def _load_preferences(self) -> UserPreferences:
        """Load preferences from disk, returning defaults if absent."""
        path = self._prefs_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return UserPreferences.from_dict(data)
            except (ValueError, KeyError, json.JSONDecodeError, TypeError) as exc:
                logger.warning("Failed to load preferences: %s", exc)
        return UserPreferences()

    def save_preferences(self) -> None:
        """Persist current preferences to disk."""
        path = self._prefs_path()
        try:
            path.write_text(json.dumps(self._preferences.to_dict(), indent=2))
        except OSError as exc:
            logger.error("Failed to save preferences: %s", exc)

    def set_default_model(self, model_id: str) -> None:
        """
        Set the default model and persist the preference.

        Args:
            model_id: ID of the model to set as default.
        """
        assert model_id is not None, "model_id must be provided"
        assert model_id is not None, "model_id must be provided"
        self._preferences.default_model_id = model_id
        self.save_preferences()

    # -- Bundled library --

    def _get_bundled_dir(self) -> Path:
        """Get the path to the bundled model directory."""
        return Path(__file__).parent / "bundled"

    def _get_manifest(self) -> dict[str, Any]:
        """Load and cache the bundled manifest."""
        if self._bundled_manifest is not None:
            return self._bundled_manifest

        manifest_path = self._get_bundled_dir() / "manifest.json"
        if manifest_path.exists():
            try:
                self._bundled_manifest = json.loads(manifest_path.read_text())
            except (ValueError, KeyError, json.JSONDecodeError, TypeError) as exc:
                logger.warning("Failed to load manifest: %s", exc)
                self._bundled_manifest = {"models": []}
        else:
            self._bundled_manifest = {"models": []}
        return self._bundled_manifest

    def list_bundled_models(self) -> list[dict[str, Any]]:
        """Return metadata for all bundled models."""
        return list(self._get_manifest().get("models", []))

    def get_bundled_model_info(self, model_id: str) -> dict[str, Any] | None:
        """Get metadata for a specific bundled model."""
        assert model_id is not None, "model_id must be provided"
        assert model_id is not None, "model_id must be provided"
        for entry in self.list_bundled_models():
            if entry["id"] == model_id:
                return entry
        return None

    # -- Loading --

    @precondition(
        lambda self, file_path: file_path is not None, "file_path must not be None"
    )
    def load_file(self, file_path: str | Path) -> LoadResult:
        """
        Load a model file in any supported format.

        Automatically detects format and converts to ParsedModel.

        Args:
            file_path: Path to a URDF or MJCF file.

        Returns:
            LoadResult with the parsed model or error information.
        """
        assert file_path is not None, "file_path must be provided"
        assert file_path is not None, "file_path must be provided"
        path = Path(file_path)
        if not path.exists():
            return LoadResult(
                source_path=path,
                error=f"File not found: {path}",
            )

        fmt = detect_format(path)

        if fmt == ModelFormat.URDF:
            return self._load_urdf(path)
        if fmt == ModelFormat.MJCF:
            return self._load_mjcf(path)
        # Try URDF first, then MJCF
        result = self._load_urdf(path)
        if result.success:
            return result
        return self._load_mjcf(path)

    @postcondition(
        lambda result: isinstance(result, LoadResult), "Must return LoadResult"
    )
    def load_bundled(self, model_id: str) -> LoadResult:
        """
        Load a model from the bundled library by ID.

        Args:
            model_id: Bundled model identifier (e.g. 'mujoco_humanoid').

        Returns:
            LoadResult with the parsed model.
        """
        assert model_id is not None, "model_id must be provided"
        assert model_id is not None, "model_id must be provided"
        info = self.get_bundled_model_info(model_id)
        if info is None:
            return LoadResult(error=f"Bundled model not found: {model_id}")

        bundled_dir = self._get_bundled_dir()
        model_path = bundled_dir / info["file"]

        if not model_path.exists():
            return LoadResult(
                source_path=model_path,
                error=f"Bundled model file missing: {model_path}",
            )

        result = self.load_file(model_path)

        # Track in recent models
        if result.success:
            self._preferences.add_recent(model_id)
            self.save_preferences()

        return LoadResult(
            model=result.model,
            source_path=result.source_path,
            source_format=result.source_format,
            success=result.success,
            error=result.error,
            warnings=result.warnings,
        )

    def load_default(self) -> LoadResult:
        """
        Load the user's default model.

        Falls back to 'mujoco_humanoid' if the configured default
        cannot be loaded.

        Returns:
            LoadResult with the parsed model.
        """
        default_id = self._preferences.default_model_id
        result = self.load_bundled(default_id)

        if not result.success and default_id != "mujoco_humanoid":
            logger.warning(
                "Default model '%s' failed to load, falling back to mujoco_humanoid",
                default_id,
            )
            result = self.load_bundled("mujoco_humanoid")

        return LoadResult(
            model=result.model,
            source_path=result.source_path,
            source_format=result.source_format,
            success=result.success,
            error=result.error,
            warnings=result.warnings,
        )

    # -- Internal loaders --

    def _load_urdf(self, path: Path) -> LoadResult:
        """Load a URDF file."""
        try:
            model = self._urdf_parser.parse(path)
            return LoadResult(
                model=model,
                source_path=path,
                source_format=ModelFormat.URDF,
                success=True,
                warnings=model.warnings,
            )
        except (OSError, ValueError, KeyError) as exc:
            return LoadResult(
                source_path=path,
                source_format=ModelFormat.URDF,
                error=str(exc),
            )

    def _load_mjcf(self, path: Path) -> LoadResult:
        """Load an MJCF file by converting to ParsedModel."""
        try:
            # Use the MJCF converter's internal parser to get a ParsedModel
            import defusedxml.ElementTree as DefusedET

            xml_string = path.read_text()
            root = DefusedET.fromstring(xml_string)
            model = self._mjcf_converter._parse_mjcf(root)
            model.source_path = path
            model.original_xml = xml_string

            return LoadResult(
                model=model,
                source_path=path,
                source_format=ModelFormat.MJCF,
                success=True,
                warnings=model.warnings,
            )
        except ImportError as exc:
            return LoadResult(
                source_path=path,
                source_format=ModelFormat.MJCF,
                error=str(exc),
            )

    # -- Format conversion utilities --

    def convert_to_urdf(self, source: str | Path) -> str | None:
        """
        Convert an MJCF file to URDF XML string.

        Args:
            source: Path to MJCF file.

        Returns:
            URDF XML string or None on failure.
        """
        try:
            return str(self._mjcf_converter.mjcf_to_urdf(source))
        except (OSError, ValueError, KeyError) as exc:
            logger.error("MJCF to URDF conversion failed: %s", exc)
            return None

    def convert_to_mjcf(self, source: str | Path) -> str | None:
        """
        Convert a URDF file to MJCF XML string.

        Args:
            source: Path to URDF file.

        Returns:
            MJCF XML string or None on failure.
        """
        try:
            return str(self._mjcf_converter.urdf_to_mjcf(source))
        except (OSError, ValueError, KeyError) as exc:
            logger.error("URDF to MJCF conversion failed: %s", exc)
            return None
