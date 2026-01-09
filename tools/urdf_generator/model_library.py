"""Model Library for Human URDFs and Golf Equipment.

This module manages a library of pre-configured URDF models including:
- Human biomechanical models from human-gazebo repository
- Golf club models (drivers, irons, putters, wedges)
- Integration with myoconverter for OpenSim <-> MuJoCo conversion

IMPORTANT: Models should be bundled in the repository when possible.
Downloading is supported but discouraged - bundled assets ensure:
1. No runtime network dependencies
2. Version stability
3. Reproducible builds

Use BundledAssets from bundled_assets/ for local models.
"""

from __future__ import annotations

import json
import logging
import math
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import bundled assets for local model access
try:
    from tools.urdf_generator.bundled_assets import (
        BundledAssetNotFoundError,
        BundledAssets,
    )
except ImportError:
    try:
        from .bundled_assets import BundledAssetNotFoundError, BundledAssets
    except ImportError:
        BundledAssets = None  # type: ignore[misc, assignment]
        BundledAssetNotFoundError = None  # type: ignore[misc, assignment]


class ModelLibrary:
    """Manages URDF model library for golf swing simulation."""

    # Human-gazebo repository pinned version
    # Pin to specific commit instead of 'master' to prevent upstream changes from breaking
    # Last verified: January 7, 2026 (functional with humanSubjectWithMeshes URDF and all meshes)
    # Repository: https://github.com/gbionics/human-gazebo
    # Verified functional on 2026-01-07; includes complete humanSubjectWithMeshes URDF and mesh files
    HUMAN_GAZEBO_COMMIT = "39cfb24fd1e16cdaa24d06b55bd16850f1825fae"

    # Human-gazebo repository base URL (pinned to commit)
    HUMAN_GAZEBO_BASE = (
        f"https://raw.githubusercontent.com/gbionics/human-gazebo/{HUMAN_GAZEBO_COMMIT}"
    )

    # Available human models
    HUMAN_MODELS = {
        "human_with_meshes": {
            "name": "Human Subject with Meshes",
            "description": "Full human body model with detailed STL meshes",
            "urdf_url": f"{HUMAN_GAZEBO_BASE}/humanSubjectWithMeshes/humanSubjectWithMesh.urdf",
            "meshes_base": f"{HUMAN_GAZEBO_BASE}/humanSubjectWithMeshes/meshes",
            "license": "CC-BY-SA 2.0",
            "commit_sha": HUMAN_GAZEBO_COMMIT,  # Track version
            "upstream_repo": "https://github.com/gbionics/human-gazebo",
        },
    }

    # Golf club specifications
    GOLF_CLUBS = {
        "driver": {
            "name": "Driver",
            "loft": 10.5,  # degrees
            "length": 1.143,  # meters (45 inches)
            "mass": 0.310,  # kg
            "head_mass": 0.200,  # kg
            "shaft_mass": 0.065,  # kg
            "grip_mass": 0.045,  # kg
        },
        "iron_5": {
            "name": "5-Iron",
            "loft": 28.0,
            "length": 0.965,  # meters (38 inches)
            "mass": 0.390,
            "head_mass": 0.260,
            "shaft_mass": 0.075,
            "grip_mass": 0.055,
        },
        "iron_7": {
            "name": "7-Iron",
            "loft": 34.0,
            "length": 0.927,  # meters (36.5 inches)
            "mass": 0.410,
            "head_mass": 0.275,
            "shaft_mass": 0.080,
            "grip_mass": 0.055,
        },
        "iron_9": {
            "name": "9-Iron",
            "loft": 42.0,
            "length": 0.889,  # meters (35 inches)
            "mass": 0.430,
            "head_mass": 0.290,
            "shaft_mass": 0.085,
            "grip_mass": 0.055,
        },
        "sand_wedge": {
            "name": "Sand Wedge",
            "loft": 56.0,
            "length": 0.889,  # meters (35 inches)
            "mass": 0.460,
            "head_mass": 0.315,
            "shaft_mass": 0.090,
            "grip_mass": 0.055,
        },
        "putter": {
            "name": "Putter",
            "loft": 3.0,
            "length": 0.864,  # meters (34 inches)
            "mass": 0.370,
            "head_mass": 0.250,
            "shaft_mass": 0.070,
            "grip_mass": 0.050,
        },
    }

    def __init__(self, base_path: Path | None = None) -> None:
        """Initialize model library.

        Args:
            base_path: Base path for storing downloaded models.
                      Defaults to shared/urdf in the project root.
        """
        if base_path is None:
            # Default to project's shared/urdf directory
            base_path = Path(__file__).parent.parent.parent / "shared" / "urdf"

        self.base_path = Path(base_path)
        self.human_models_path = self.base_path / "human_models"
        self.golf_clubs_path = self.base_path / "golf_clubs"
        self.meshes_path = self.base_path.parent / "meshes"

        # Create directories
        self.human_models_path.mkdir(parents=True, exist_ok=True)
        self.golf_clubs_path.mkdir(parents=True, exist_ok=True)
        (self.meshes_path / "human").mkdir(parents=True, exist_ok=True)
        (self.meshes_path / "golf_clubs").mkdir(parents=True, exist_ok=True)

        logger.info(f"Model library initialized at: {self.base_path}")

    def get_human_model(self, model_key: str) -> Path | None:
        """Get a human model, preferring bundled assets over downloads.

        This method first checks for bundled assets in the repository.
        If no bundled asset exists, it falls back to the download path
        (but warns that this is discouraged).

        Args:
            model_key: Key identifying the model

        Returns:
            Path to the URDF file, or None if not available
        """
        # First, try bundled assets (preferred)
        if BundledAssets is not None:
            try:
                bundled = BundledAssets()
                # Map model_key to bundled asset name if different
                bundled_name = model_key.replace("_with_", "_subject_with_")
                if bundled.is_model_bundled("human_models", bundled_name):
                    logger.info(f"Using bundled model: {bundled_name}")
                    return bundled.get_human_model_path(bundled_name)
                if bundled.is_model_bundled("human_models", model_key):
                    logger.info(f"Using bundled model: {model_key}")
                    return bundled.get_human_model_path(model_key)
            except Exception as e:
                logger.debug(f"Bundled asset check failed: {e}")

        # Check if model exists locally (previously downloaded)
        model_dir = self.human_models_path / model_key
        urdf_path = model_dir / "model.urdf"
        if urdf_path.exists():
            logger.info(f"Using cached model: {urdf_path}")
            return urdf_path

        # Model not found locally
        logger.warning(
            f"Model '{model_key}' is not bundled or cached.\n"
            f"Consider bundling this model in the repository for offline use.\n"
            f"Use download_human_model() to download if network access is available."
        )
        return None

    def download_human_model(self, model_key: str, force: bool = False) -> Path | None:
        """Download a human URDF model and its meshes.

        WARNING: Downloading is discouraged. Models should be bundled in the
        repository for offline use and version stability. Use get_human_model()
        to prefer bundled assets.

        Args:
            model_key: Key identifying the model in HUMAN_MODELS dict
            force: Force re-download even if files exist

        Returns:
            Path to downloaded URDF file, or None if download failed
        """
        if model_key not in self.HUMAN_MODELS:
            logger.error(f"Unknown human model: {model_key}")
            return None

        # Warn about downloading
        logger.warning(
            "Downloading model from network. Consider bundling this model locally.\n"
            "See: tools/urdf_generator/bundled_assets/README.md"
        )

        model_info = self.HUMAN_MODELS[model_key]
        model_dir = self.human_models_path / model_key
        urdf_path = model_dir / "model.urdf"

        # Skip if already exists and not forcing
        if urdf_path.exists() and not force:
            logger.info(f"Model {model_key} already exists at: {urdf_path}")
            return urdf_path

        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download URDF file
            logger.info(f"Downloading URDF: {model_info['urdf_url']}")
            with urllib.request.urlopen(model_info["urdf_url"]) as response:
                urdf_content = response.read().decode("utf-8")
                urdf_path.write_text(urdf_content, encoding="utf-8")

            # IMPORTANT: Mesh downloads are NOT implemented.
            # Meshes must be bundled with the repository.
            # The URDF references mesh files that need to exist locally.
            logger.warning(
                f"URDF downloaded but meshes are NOT downloaded.\n"
                f"The model will not render correctly without mesh files.\n"
                f"To fix: Bundle mesh files in the repository at:\n"
                f"  tools/urdf_generator/bundled_assets/human_models/{model_key}/meshes/\n"
                f"See bundled_assets/README.md for instructions."
            )

            # Save metadata
            metadata_path = model_dir / "metadata.json"
            metadata = {
                "model_key": model_key,
                "name": model_info["name"],
                "description": model_info["description"],
                "license": model_info["license"],
                "urdf_file": "model.urdf",
                "warning": "Meshes not bundled - model may not render correctly",
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            return urdf_path

        except Exception as e:
            logger.error(f"Failed to download model {model_key}: {e}")
            return None

    def generate_golf_club_urdf(self, club_key: str) -> Path | None:
        """Generate a URDF file for a golf club.

        Args:
            club_key: Key identifying the club in GOLF_CLUBS dict

        Returns:
            Path to generated URDF file
        """
        if club_key not in self.GOLF_CLUBS:
            logger.error(f"Unknown golf club: {club_key}")
            return None

        club_info = self.GOLF_CLUBS[club_key]
        club_dir = self.golf_clubs_path / club_key
        club_dir.mkdir(parents=True, exist_ok=True)

        urdf_path = club_dir / f"{club_key}.urdf"

        # Generate URDF content
        urdf_content = self._create_golf_club_urdf(club_key, club_info)

        urdf_path.write_text(urdf_content, encoding="utf-8")
        logger.info(f"Generated golf club URDF: {urdf_path}")

        return urdf_path

    def _create_golf_club_urdf(self, club_key: str, club_info: dict) -> str:
        """Create URDF XML content for a golf club.

        Args:
            club_key: Unique identifier for the club
            club_info: Dictionary containing club specifications

        Returns:
            URDF XML content as string
        """
        # Shaft dimensions (tapered cylinder approximated as cylinder)
        shaft_length = club_info["length"] - 0.254 - 0.076  # minus head and grip
        shaft_radius = 0.006  # Average radius 6mm

        # Head dimensions (simplified box for now)
        head_length = 0.100
        head_width = 0.050
        head_height = 0.040

        # Grip dimensions
        grip_length = 0.254  # Standard 10 inches
        grip_radius = 0.013  # ~26mm diameter

        urdf = f"""<?xml version="1.0"?>
<robot name="{club_key}">
    <!-- Base link (reference frame) -->
    <link name="base_link">
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="0.01 0.01 0.01"/>
            </geometry>
            <material name="transparent">
                <color rgba="0 0 0 0"/>
            </material>
        </visual>
    </link>

    <!-- Grip -->
    <link name="grip">
        <inertial>
            <mass value="{club_info['grip_mass']}"/>
            <origin xyz="0 0 {grip_length/2}" rpy="0 0 0"/>
            <inertia ixx="{club_info['grip_mass'] * (3*grip_radius**2 + grip_length**2) / 12}"
                     iyy="{club_info['grip_mass'] * (3*grip_radius**2 + grip_length**2) / 12}"
                     izz="{club_info['grip_mass'] * grip_radius**2 / 2}"
                     ixy="0" ixz="0" iyz="0"/>
        </inertial>
        <visual>
            <origin xyz="0 0 {grip_length/2}" rpy="0 0 0"/>
            <geometry>
                <cylinder length="{grip_length}" radius="{grip_radius}"/>
            </geometry>
            <material name="black">
                <color rgba="0.1 0.1 0.1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 {grip_length/2}" rpy="0 0 0"/>
            <geometry>
                <cylinder length="{grip_length}" radius="{grip_radius}"/>
            </geometry>
        </collision>
    </link>

    <joint name="base_to_grip" type="fixed">
        <parent link="base_link"/>
        <child link="grip"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
    </joint>

    <!-- Shaft -->
    <link name="shaft">
        <inertial>
            <mass value="{club_info['shaft_mass']}"/>
            <origin xyz="0 0 {shaft_length/2}" rpy="0 0 0"/>
            <inertia ixx="{club_info['shaft_mass'] * (3*shaft_radius**2 + shaft_length**2) / 12}"
                     iyy="{club_info['shaft_mass'] * (3*shaft_radius**2 + shaft_length**2) / 12}"
                     izz="{club_info['shaft_mass'] * shaft_radius**2 / 2}"
                     ixy="0" ixz="0" iyz="0"/>
        </inertial>
        <visual>
            <origin xyz="0 0 {shaft_length/2}" rpy="0 0 0"/>
            <geometry>
                <cylinder length="{shaft_length}" radius="{shaft_radius}"/>
            </geometry>
            <material name="graphite">
                <color rgba="0.2 0.2 0.2 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 {shaft_length/2}" rpy="0 0 0"/>
            <geometry>
                <cylinder length="{shaft_length}" radius="{shaft_radius}"/>
            </geometry>
        </collision>
    </link>

    <joint name="grip_to_shaft" type="fixed">
        <parent link="grip"/>
        <child link="shaft"/>
        <origin xyz="0 0 {grip_length}" rpy="0 0 0"/>
    </joint>

    <!-- Club Head -->
    <link name="club_head">
        <inertial>
            <mass value="{club_info['head_mass']}"/>
            <origin xyz="{head_length/2} 0 0" rpy="0 {club_info['loft']*math.pi/180} 0"/>
            <inertia ixx="{club_info['head_mass'] * (head_width**2 + head_height**2) / 12}"
                     iyy="{club_info['head_mass'] * (head_length**2 + head_height**2) / 12}"
                     izz="{club_info['head_mass'] * (head_length**2 + head_width**2) / 12}"
                     ixy="0" ixz="0" iyz="0"/>
        </inertial>
        <visual>
            <origin xyz="{head_length/2} 0 0" rpy="0 {club_info['loft']*math.pi/180} 0"/>
            <geometry>
                <box size="{head_length} {head_width} {head_height}"/>
            </geometry>
            <material name="steel">
                <color rgba="0.7 0.7 0.7 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="{head_length/2} 0 0" rpy="0 {club_info['loft']*math.pi/180} 0"/>
            <geometry>
                <box size="{head_length} {head_width} {head_height}"/>
            </geometry>
        </collision>
    </link>

    <joint name="shaft_to_head" type="fixed">
        <parent link="shaft"/>
        <child link="club_head"/>
        <origin xyz="0 0 {shaft_length}" rpy="0 0 0"/>
    </joint>
</robot>
"""
        return urdf

    def list_available_models(self) -> dict[str, list[str]]:
        """List all available models in the library.

        Returns:
            Dictionary with 'human' and 'golf_clubs' keys containing lists of model names
        """
        return {
            "human": list(self.HUMAN_MODELS.keys()),
            "golf_clubs": list(self.GOLF_CLUBS.keys()),
        }

    def get_model_info(self, category: str, model_key: str) -> dict[str, Any] | None:
        """Get information about a specific model.

        Args:
            category: 'human' or 'golf_clubs'
            model_key: Key identifying the model

        Returns:
            Dictionary with model information, or None if not found
        """
        if category == "human":
            return self.HUMAN_MODELS.get(model_key)
        elif category == "golf_clubs":
            return self.GOLF_CLUBS.get(model_key)
        else:
            logger.error(f"Unknown category: {category}")
            return None
