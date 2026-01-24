"""Bundled Assets for URDF Generator.

This module provides access to mesh files bundled with the repository.
Meshes are stored locally to ensure:
1. No runtime downloads required
2. Version stability with URDF files
3. Offline functionality

Usage:
    from tools.urdf_generator.bundled_assets import BundledAssets

    assets = BundledAssets()
    model_path = assets.get_model_path("human_subject_with_meshes")
"""

from __future__ import annotations

import json
from src.shared.python.logging_config import get_logger
from pathlib import Path
from typing import Any

logger = get_logger(__name__)

# Directory containing bundled assets
BUNDLED_ASSETS_DIR = Path(__file__).parent


class BundledAssetNotFoundError(Exception):
    """Raised when a bundled asset is not found."""

    pass


class BundledAssets:
    """Manager for bundled mesh and URDF assets.

    This class provides access to models and meshes that are committed
    directly to the repository. NO runtime downloads are performed.
    """

    def __init__(self) -> None:
        """Initialize the bundled assets manager."""
        self.assets_dir = BUNDLED_ASSETS_DIR
        self.human_models_dir = self.assets_dir / "human_models"
        self.golf_equipment_dir = self.assets_dir / "golf_equipment"

        # Ensure directories exist
        self.human_models_dir.mkdir(parents=True, exist_ok=True)
        self.golf_equipment_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Bundled assets directory: {self.assets_dir}")

    def list_available_models(self) -> dict[str, list[str]]:
        """List all bundled models.

        Returns:
            Dictionary with 'human_models' and 'golf_equipment' keys
            containing lists of available model names.
        """
        human_models = []
        golf_equipment = []

        if self.human_models_dir.exists():
            human_models = [
                d.name
                for d in self.human_models_dir.iterdir()
                if d.is_dir() and (d / "model.urdf").exists()
            ]

        if self.golf_equipment_dir.exists():
            golf_equipment = [
                d.name
                for d in self.golf_equipment_dir.iterdir()
                if d.is_dir() and list(d.glob("*.urdf"))
            ]

        return {
            "human_models": human_models,
            "golf_equipment": golf_equipment,
        }

    def get_human_model_path(self, model_name: str) -> Path:
        """Get the path to a bundled human model URDF.

        Args:
            model_name: Name of the model (directory name)

        Returns:
            Path to the model.urdf file

        Raises:
            BundledAssetNotFoundError: If the model is not bundled
        """
        model_dir = self.human_models_dir / model_name
        urdf_path = model_dir / "model.urdf"

        if not urdf_path.exists():
            available = self.list_available_models()["human_models"]
            raise BundledAssetNotFoundError(
                f"Human model '{model_name}' is not bundled.\n"
                f"\n"
                f"Available bundled models: {available or 'None'}\n"
                f"\n"
                f"To add this model:\n"
                f"1. Download the model files to: {model_dir}\n"
                f"2. Ensure model.urdf exists\n"
                f"3. Commit the files to the repository"
            )

        return urdf_path

    def get_golf_equipment_path(self, equipment_name: str) -> Path:
        """Get the path to a bundled golf equipment URDF.

        Args:
            equipment_name: Name of the equipment (e.g., 'driver', 'putter')

        Returns:
            Path to the URDF file

        Raises:
            BundledAssetNotFoundError: If the equipment is not bundled
        """
        equip_dir = self.golf_equipment_dir / equipment_name
        urdf_files = list(equip_dir.glob("*.urdf")) if equip_dir.exists() else []

        if not urdf_files:
            available = self.list_available_models()["golf_equipment"]
            raise BundledAssetNotFoundError(
                f"Golf equipment '{equipment_name}' is not bundled.\n"
                f"\n"
                f"Available bundled equipment: {available or 'None'}\n"
                f"\n"
                f"To add this equipment:\n"
                f"1. Create directory: {equip_dir}\n"
                f"2. Add the URDF file\n"
                f"3. Commit the files to the repository"
            )

        return urdf_files[0]

    def get_meshes_dir(self, model_category: str, model_name: str) -> Path:
        """Get the path to the meshes directory for a model.

        Args:
            model_category: 'human_models' or 'golf_equipment'
            model_name: Name of the model

        Returns:
            Path to the meshes directory

        Raises:
            BundledAssetNotFoundError: If the meshes directory doesn't exist
        """
        if model_category == "human_models":
            meshes_dir = self.human_models_dir / model_name / "meshes"
        elif model_category == "golf_equipment":
            meshes_dir = self.golf_equipment_dir / model_name / "meshes"
        else:
            raise ValueError(
                f"Unknown model category: {model_category}. "
                f"Use 'human_models' or 'golf_equipment'."
            )

        if not meshes_dir.exists():
            raise BundledAssetNotFoundError(
                f"Meshes directory not found: {meshes_dir}\n"
                f"\n"
                f"The model may be missing mesh files.\n"
                f"Download meshes to: {meshes_dir}"
            )

        return meshes_dir

    def get_model_metadata(
        self, model_category: str, model_name: str
    ) -> dict[str, Any]:
        """Get metadata for a bundled model.

        Args:
            model_category: 'human_models' or 'golf_equipment'
            model_name: Name of the model

        Returns:
            Dictionary containing model metadata
        """
        if model_category == "human_models":
            metadata_path = self.human_models_dir / model_name / "metadata.json"
        elif model_category == "golf_equipment":
            metadata_path = self.golf_equipment_dir / model_name / "metadata.json"
        else:
            raise ValueError(f"Unknown model category: {model_category}")

        if not metadata_path.exists():
            return {
                "name": model_name,
                "description": "No metadata available",
                "license": "Unknown",
            }

        return dict(json.loads(metadata_path.read_text(encoding="utf-8")))

    def is_model_bundled(self, model_category: str, model_name: str) -> bool:
        """Check if a model is bundled in the repository.

        Args:
            model_category: 'human_models' or 'golf_equipment'
            model_name: Name of the model

        Returns:
            True if the model is bundled, False otherwise
        """
        try:
            if model_category == "human_models":
                self.get_human_model_path(model_name)
            elif model_category == "golf_equipment":
                self.get_golf_equipment_path(model_name)
            else:
                return False
            return True
        except BundledAssetNotFoundError:
            return False


# Convenience function
def get_bundled_model_path(model_name: str, category: str = "human_models") -> Path:
    """Get the path to a bundled model.

    Convenience function that creates a BundledAssets instance and returns
    the path to the specified model.

    Args:
        model_name: Name of the model
        category: 'human_models' or 'golf_equipment'

    Returns:
        Path to the URDF file
    """
    assets = BundledAssets()
    if category == "human_models":
        return assets.get_human_model_path(model_name)
    elif category == "golf_equipment":
        return assets.get_golf_equipment_path(model_name)
    else:
        raise ValueError(f"Unknown category: {category}")
