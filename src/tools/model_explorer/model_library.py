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

import importlib
import json
import logging
import math
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

# Add project root to path for src imports when run as standalone script
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from src.shared.python.logging_config import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


try:
    import robot_descriptions

    # Try to get list of descriptions (simple heuristic: attributes ending in _description)
    ROBOT_DESCRIPTIONS_AVAILABLE = True
except ImportError:
    robot_descriptions = None
    ROBOT_DESCRIPTIONS_AVAILABLE = False

logger = get_logger(__name__)

# Try to import bundled assets for local model access
try:
    from src.tools.model_explorer.bundled_assets import (
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
    # Available human models
    HUMAN_MODELS = {
        "mujoco_humanoid": {
            "name": "MuJoCo Humanoid (Default)",
            "description": "Full Body Golf Swing Model (Embedded)",
            "type": "embedded",
            "embedded_key": "full_body_golf_swing",
            "license": "Internal",
            "urdf_url": "",  # Not downloadable
        },
        "human_with_meshes_local": {
            "name": "Human Subject with Meshes (Local)",
            "description": "Full body model with detailed STL meshes (from local human-gazebo)",
            "type": "local_submodule",
            "urdf_subpath": "src/shared/tools/human-gazebo/humanSubjectWithMeshes/humanSubjectWithMesh.urdf",
        },
        "human_with_spinal_cord_local": {
            "name": "Human Subject with Spinal Cord (Local)",
            "description": "Full body model with spinal cord meshes (from local human-gazebo)",
            "type": "local_submodule",
            "urdf_subpath": "src/shared/tools/human-gazebo/humanSubjectWithSpinalCordMeshes/humanSubjectWithSpinalCordMeshes.urdf",
        },
        "human_with_meshes": {
            "name": "Human Subject with Meshes (Remote)",
            "description": "Full human body model with detailed STL meshes (Downloadable)",
            "urdf_url": f"{HUMAN_GAZEBO_BASE}/humanSubjectWithMeshes/humanSubjectWithMesh.urdf",
            "meshes_base": f"{HUMAN_GAZEBO_BASE}/humanSubjectWithMeshes/meshes",
            "license": "CC-BY-SA 2.0",
            "commit_sha": HUMAN_GAZEBO_COMMIT,
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
        # ... (Clubs truncated for brevity if not modifying, but I must replace the block)
        "iron_5": {
            "name": "5-Iron",
            "loft": 28.0,
            "length": 0.965,
            "mass": 0.390,
            "head_mass": 0.260,
            "shaft_mass": 0.075,
            "grip_mass": 0.055,
        },
        "iron_7": {
            "name": "7-Iron",
            "loft": 34.0,
            "length": 0.927,
            "mass": 0.410,
            "head_mass": 0.275,
            "shaft_mass": 0.080,
            "grip_mass": 0.055,
        },
        "putter": {
            "name": "Putter",
            "loft": 3.0,
            "length": 0.864,
            "mass": 0.370,
            "head_mass": 0.250,
            "shaft_mass": 0.070,
            "grip_mass": 0.050,
        },
    }

    # Pendulum Models
    PENDULUM_MODELS = {
        "double_pendulum_2d": {
            "name": "Double Pendulum (2D)",
            "description": "Simple planar double pendulum representing the arms and club.",
            "path": "src/engines/pendulum_models/double_pendulum.xml",
            "type": "mjcf",
        },
        "triple_pendulum_3d": {
            "name": "Triple Pendulum (3D)",
            "description": "3D triple pendulum with shoulder, wrist, and club rotation.",
            "path": "src/engines/pendulum_models/triple_pendulum.xml",
            "type": "mjcf",
        },
    }

    # Robotic Manipulators
    ROBOTIC_MODELS = {
        "kuka_iiwa_golf": {
            "name": "KUKA LBR iiwa 14 (Golf Attachment)",
            "description": "7-DOF robotic manipulator with golf club end-effector.",
            "path": "src/engines/physics_engines/drake/models/iiwa14_golf.urdf",
            "type": "urdf",
        },
        "ur5_golf": {
            "name": "Universal Robots UR5",
            "description": "6-DOF collaborative robot arm.",
            "path": "src/engines/physics_engines/mujoco/models/ur5_golf.xml",
            "type": "mjcf",
        },
    }

    # Components
    COMPONENT_MODELS = {
        "flexible_shaft": {
            "name": "Flexible Shaft Element",
            "description": "Beam element modeling shaft flexibility.",
            "type": "component",
        },
        "golf_ball": {
            "name": "Golf Ball (Standard)",
            "description": "Standard golf ball with contact geometry.",
            "type": "component",
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
        # Handle embedded MuJoCo Humanoid special case
        if model_key == "mujoco_humanoid":
            return self._get_cached_embedded_model("full_body_golf_swing")

        # Handle local submodule (repo-bundled via git submodule)
        model_info = self.HUMAN_MODELS.get(model_key)
        if model_info and model_info.get("type") == "local_submodule":
            path = _project_root / model_info["urdf_subpath"]
            if path.exists():
                logger.info(f"Using local submodule model: {path}")
                return path
            else:
                logger.warning(
                    f"Local submodule model not found at {path}.\n"
                    f"Ensure submodules are initialized: git submodule update --init --recursive"
                )
                return None

        # First, try bundled assets (preferred)
        if BundledAssets is not None:
            try:
                bundled = BundledAssets()
                # Map model_key to bundled asset name if different
                bundled_name = model_key.replace("_with_", "_subject_with_")
                if bundled.is_model_bundled("human_models", bundled_name):
                    logger.info(f"Using bundled model: {bundled_name}")
                    path = bundled.get_human_model_path(bundled_name)
                    return path  # type: ignore[no-any-return]
                if bundled.is_model_bundled("human_models", model_key):
                    logger.info(f"Using bundled model: {model_key}")
                    path = bundled.get_human_model_path(model_key)
                    return path  # type: ignore[no-any-return]
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

    def _get_cached_embedded_model(self, embedded_key: str) -> Path | None:
        """Extract an embedded model and cache it to a file.

        Args:
            embedded_key: Key in get_embedded_mujoco_models()

        Returns:
            Path to the cached XML file
        """
        embedded = self.get_embedded_mujoco_models()
        if embedded_key not in embedded:
            logger.error(f"Embedded model key not found: {embedded_key}")
            return None

        content = embedded[embedded_key]["content"]

        # Determine cache path
        # Use a stable directory for this model
        cache_dir = self.human_models_path / "mujoco_humanoid"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save as XML
        file_path = cache_dir / "model.xml"

        # Write content if it changed or doesn't exist
        # For simplicity, always write (it's fast)
        try:
            file_path.write_text(content, encoding="utf-8")
            logger.info(f"Cached embedded model to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to cache embedded model: {e}")
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

        # Skip download for embedded models (they are local)
        if self.HUMAN_MODELS[model_key].get("type") == "embedded":
            logger.info(f"Model {model_key} is embedded, skipping download.")
            return self.get_human_model(model_key)

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
            <mass value="{club_info["grip_mass"]}"/>
            <origin xyz="0 0 {grip_length / 2}" rpy="0 0 0"/>
            <inertia ixx="{club_info["grip_mass"] * (3 * grip_radius**2 + grip_length**2) / 12}"
                     iyy="{club_info["grip_mass"] * (3 * grip_radius**2 + grip_length**2) / 12}"
                     izz="{club_info["grip_mass"] * grip_radius**2 / 2}"
                     ixy="0" ixz="0" iyz="0"/>
        </inertial>
        <visual>
            <origin xyz="0 0 {grip_length / 2}" rpy="0 0 0"/>
            <geometry>
                <cylinder length="{grip_length}" radius="{grip_radius}"/>
            </geometry>
            <material name="black">
                <color rgba="0.1 0.1 0.1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 {grip_length / 2}" rpy="0 0 0"/>
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
            <mass value="{club_info["shaft_mass"]}"/>
            <origin xyz="0 0 {shaft_length / 2}" rpy="0 0 0"/>
            <inertia ixx="{club_info["shaft_mass"] * (3 * shaft_radius**2 + shaft_length**2) / 12}"
                     iyy="{club_info["shaft_mass"] * (3 * shaft_radius**2 + shaft_length**2) / 12}"
                     izz="{club_info["shaft_mass"] * shaft_radius**2 / 2}"
                     ixy="0" ixz="0" iyz="0"/>
        </inertial>
        <visual>
            <origin xyz="0 0 {shaft_length / 2}" rpy="0 0 0"/>
            <geometry>
                <cylinder length="{shaft_length}" radius="{shaft_radius}"/>
            </geometry>
            <material name="graphite">
                <color rgba="0.2 0.2 0.2 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 {shaft_length / 2}" rpy="0 0 0"/>
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
            <mass value="{club_info["head_mass"]}"/>
            <origin xyz="{head_length / 2} 0 0" rpy="0 {club_info["loft"] * math.pi / 180} 0"/>
            <inertia ixx="{club_info["head_mass"] * (head_width**2 + head_height**2) / 12}"
                     iyy="{club_info["head_mass"] * (head_length**2 + head_height**2) / 12}"
                     izz="{club_info["head_mass"] * (head_length**2 + head_width**2) / 12}"
                     ixy="0" ixz="0" iyz="0"/>
        </inertial>
        <visual>
            <origin xyz="{head_length / 2} 0 0" rpy="0 {club_info["loft"] * math.pi / 180} 0"/>
            <geometry>
                <box size="{head_length} {head_width} {head_height}"/>
            </geometry>
            <material name="steel">
                <color rgba="0.7 0.7 0.7 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="{head_length / 2} 0 0" rpy="0 {club_info["loft"] * math.pi / 180} 0"/>
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

    def list_available_models(self) -> dict[str, Any]:
        """List all available models in the library.

        Returns:
            Dictionary with keys:
            - 'human': Pre-defined human models
            - 'golf_clubs': Pre-defined golf clubs
            - 'pendulum': Pendulum models
            - 'robotic': Robotic manipulator models
            - 'component': Component models
            - 'discovered': URDF/MJCF files found in repository
            - 'embedded': MuJoCo XML strings embedded in python code
        """
        discovered = self.discover_repo_models()
        embedded = self.get_embedded_mujoco_models()
        robot_descs = self.discover_robot_descriptions()

        return {
            "human": list(self.HUMAN_MODELS.keys()),
            "golf_clubs": list(self.GOLF_CLUBS.keys()),
            "pendulum": list(self.PENDULUM_MODELS.keys()),
            "robotic": list(self.ROBOTIC_MODELS.keys()),
            "component": list(self.COMPONENT_MODELS.keys()),
            "discovered": discovered,
            "embedded": embedded,
            "robot_descriptions": robot_descs,
        }

    def get_model_info(self, category: str, model_key: str) -> dict[str, Any] | None:
        """Get information about a specific model.

        Args:
            category: 'human', 'golf_clubs', 'pendulum', 'robotic', 'component',
                     'discovered', or 'embedded'
            model_key: Key identifying the model

        Returns:
            Dictionary with model information, or None if not found
        """
        if category == "human":
            return self.HUMAN_MODELS.get(model_key)
        elif category == "golf_clubs":
            return self.GOLF_CLUBS.get(model_key)
        elif category == "pendulum":
            return self.PENDULUM_MODELS.get(model_key)
        elif category == "robotic":
            return self.ROBOTIC_MODELS.get(model_key)
        elif category == "component":
            return self.COMPONENT_MODELS.get(model_key)
        elif category == "discovered":
            discovered = self.discover_repo_models()
            for model in discovered:
                if model["config_key"] == model_key:
                    return model
            return None
        elif category == "embedded":
            embedded = self.get_embedded_mujoco_models()
            return embedded.get(model_key)
        elif category == "robot_descriptions":
            models = self.discover_robot_descriptions()
            for model in models:
                if model["config_key"] == model_key:
                    return model
            return None
        else:
            logger.error(f"Unknown category: {category}")
            return None

    def discover_repo_models(self) -> list[dict[str, Any]]:
        """Scan the repository for URDF and MJCF models.

        Returns:
            List of dictionaries containing model info:
            {
                'name': filename,
                'path': absolute path,
                'type': 'urdf' | 'mjcf',
                'config_key': unique key
            }
        """
        models = []
        # Use project root defined at module level
        src_root = _project_root / "src"

        if not src_root.exists():
            logger.warning(f"Source root not found at {src_root}")
            return []

        # Walk through directory
        for root, _, files in os.walk(src_root):
            # Skip bundled assets and cache dirs to avoid dupes/junk
            if "bundled_assets" in root or "__pycache__" in root:
                continue

            for file in files:
                file_path = Path(root) / file

                # Check for URDF
                if file.lower().endswith(".urdf"):
                    models.append(
                        {
                            "name": file,
                            "description": f"URDF file at {file_path.relative_to(_project_root)}",
                            "path": str(file_path),
                            "type": "urdf",
                            "config_key": f"urdf_{file}_{hash(str(file_path))}",
                        }
                    )

                # Check for MJCF (xml with <mujoco tag)
                elif file.lower().endswith(".xml") or file.lower().endswith(".mjcf"):
                    try:
                        # Quick check for mujoco tag
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            start = f.read(500)
                            if "<mujoco" in start or "<robot" in start:
                                model_type = "mjcf" if "<mujoco" in start else "urdf"
                                models.append(
                                    {
                                        "name": file,
                                        "description": f"{model_type.upper()} file at {file_path.relative_to(_project_root)}",
                                        "path": str(file_path),
                                        "type": model_type,
                                        "config_key": f"repo_{file}_{hash(str(file_path))}",
                                    }
                                )
                    except Exception:
                        pass  # reading error, skip

        return sorted(models, key=lambda x: x["name"])

    def discover_robot_descriptions(self) -> list[dict[str, Any]]:
        """Discover models available in the robot_descriptions package.

        Returns:
            List of model info dictionaries.
        """
        if not ROBOT_DESCRIPTIONS_AVAILABLE or not robot_descriptions:
            return []

        models = []
        # Inspect module attributes
        for attr in dir(robot_descriptions):
            if attr.endswith("_description"):
                try:
                    module = getattr(robot_descriptions, attr)
                    # Check for URDF_PATH or MJCF_PATH
                    urdf_path = getattr(module, "URDF_PATH", None)
                    mjcf_path = getattr(module, "MJCF_PATH", None)

                    if urdf_path or mjcf_path:
                        path = urdf_path if urdf_path else mjcf_path
                        m_type = "urdf" if urdf_path else "mjcf"
                        name = (
                            attr.replace("_description", "").replace("_", " ").title()
                        )

                        models.append(
                            {
                                "name": name,
                                "description": f"Community model from robot_descriptions ({m_type.upper()})",
                                "path": str(path),
                                "type": m_type,
                                "config_key": attr,
                                "package": "robot_descriptions",
                            }
                        )
                except Exception:
                    continue

        return sorted(models, key=lambda x: x["name"])

    def get_embedded_mujoco_models(self) -> dict[str, dict[str, Any]]:
        """Retrieve MuJoCo models embedded in python code.

        Target: src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.models
        """
        models = {}
        try:
            # Dynamic import to avoid hard dependency on non-tool code
            module_name = (
                "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.models"
            )
            if module_name not in sys.modules:
                importlib.import_module(module_name)

            module = sys.modules[module_name]

            # Scan module attributes
            for attr_name in dir(module):
                if attr_name.endswith("_XML") and attr_name.isupper():
                    content = getattr(module, attr_name)
                    if isinstance(content, str) and "<mujoco" in content:
                        key = attr_name.lower().replace("_xml", "")
                        models[key] = {
                            "name": attr_name.replace("_XML", "")
                            .replace("_", " ")
                            .title(),
                            "description": "Embedded MuJoCo model",
                            "content": content,
                            "type": "mjcf_string",
                            "config_key": key,
                        }
        except ImportError:
            logger.warning(
                "Could not import mujoco_humanoid_golf.models - embedded models unavailable"
            )
        except Exception as e:
            logger.error(f"Error loading embedded models: {e}")

        return models
