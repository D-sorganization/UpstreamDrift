"""Standard model management for Golf Modeling Suite.

This module provides standardized model loading across all physics engines,
ensuring consistent biomechanical models for cross-engine validation.
"""

import urllib.request
from pathlib import Path
from typing import Any

import yaml

from src.shared.python.common_utils import GolfModelingError
from src.shared.python.constants import DEG_TO_RAD
from src.shared.python.io_utils import ensure_directory
from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


class StandardModelManager:
    """Manages standardized models for consistent cross-engine simulation."""

    def __init__(self, suite_root: Path | None = None):
        """Initialize standard model manager.

        Args:
            suite_root: Root directory of Golf Modeling Suite
        """
        if suite_root is None:
            suite_root = Path(__file__).parent.parent.parent

        self.suite_root = Path(suite_root)
        self.models_dir = self.suite_root / "shared" / "urdf"
        self.meshes_dir = self.suite_root / "shared" / "meshes"
        self.config_file = self.models_dir / "standard_models.yaml"

        # Ensure directories exist
        ensure_directory(self.models_dir)
        ensure_directory(self.meshes_dir)

        # Load configuration
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load standard models configuration."""
        if not self.config_file.exists():
            # Create default configuration
            default_config = {
                "standard_humanoid": {
                    "name": "Standard Humanoid with Meshes",
                    "description": "High-fidelity human body model from human-gazebo repository",
                    "source": "https://github.com/robotology/human-gazebo",
                    "urdf_path": "shared/urdf/human_models/humanSubject06_66dof.urdf",
                    "mesh_dir": "shared/meshes/human",
                    "license": "CC-BY-SA 2.0",
                    "dof": 66,
                    "mass_kg": 75.0,
                    "height_m": 1.75,
                },
                "simple_humanoid": {
                    "name": "Simple Humanoid",
                    "description": "Simplified human model for basic testing",
                    "urdf_path": "shared/urdf/simple_humanoid.urdf",
                    "dof": 12,
                    "mass_kg": 70.0,
                    "height_m": 1.70,
                },
                "golf_clubs": {
                    "driver": {
                        "name": "Driver",
                        "urdf_path": "shared/urdf/golf_clubs/driver.urdf",
                        "loft_deg": 10.5,
                        "length_m": 1.143,
                        "mass_kg": 0.310,
                    },
                    "iron_7": {
                        "name": "7-Iron",
                        "urdf_path": "shared/urdf/golf_clubs/7_iron.urdf",
                        "loft_deg": 34.0,
                        "length_m": 0.927,
                        "mass_kg": 0.410,
                    },
                },
            }

            with open(self.config_file, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)

            return default_config

        with open(self.config_file) as f:
            config = yaml.safe_load(f)
            return config or {}

    def get_standard_humanoid_path(self) -> Path:
        """Get path to standard humanoid URDF.

        Returns:
            Path to the standard humanoid URDF file

        Raises:
            GolfModelingError: If standard humanoid is not available
        """
        urdf_path = self.suite_root / str(self.config["standard_humanoid"]["urdf_path"])

        if not urdf_path.exists():
            logger.info("Standard humanoid not found, attempting to download...")
            self.download_standard_humanoid()

        if not urdf_path.exists():
            raise GolfModelingError(
                f"Standard humanoid URDF not found at {urdf_path}. "
                "Run 'golf-suite --setup-models' to download required models."
            )

        return urdf_path

    def download_standard_humanoid(self) -> bool:
        """Download standard humanoid model from human-gazebo repository.

        Returns:
            True if download successful, False otherwise
        """
        try:
            logger.info("Downloading standard humanoid model from human-gazebo...")

            # Create target directories
            human_models_dir = self.models_dir / "human_models"
            human_meshes_dir = self.meshes_dir / "human"
            human_models_dir.mkdir(parents=True, exist_ok=True)
            human_meshes_dir.mkdir(parents=True, exist_ok=True)

            # Download specific files from human-gazebo repository
            base_url = (
                "https://raw.githubusercontent.com/robotology/human-gazebo/master"
            )

            files_to_download = [
                ("models/humanSubject06_66dof/model.urdf", "humanSubject06_66dof.urdf"),
                (
                    "models/humanSubject06_66dof/conf/human_model.yaml",
                    "human_model.yaml",
                ),
            ]

            for remote_path, local_filename in files_to_download:
                url = f"{base_url}/{remote_path}"
                local_path = human_models_dir / local_filename

                logger.info(f"Downloading {url} -> {local_path}")
                urllib.request.urlretrieve(url, local_path)

            # Download mesh files (this is a simplified approach - in practice you'd want
            # to download the actual mesh files from the repository)
            self._create_temporary_meshes(human_meshes_dir)

            logger.info("Standard humanoid model downloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to download standard humanoid model: {e}")
            return False

    def _create_temporary_meshes(self, mesh_dir: Path) -> None:
        """Create temporary mesh files for development.

        In production, this would download actual STL files from human-gazebo.
        """
        # Create basic temporary STL files
        temporary_meshes = [
            "head.stl",
            "torso.stl",
            "pelvis.stl",
            "left_upper_arm.stl",
            "left_forearm.stl",
            "left_hand.stl",
            "right_upper_arm.stl",
            "right_forearm.stl",
            "right_hand.stl",
            "left_thigh.stl",
            "left_shin.stl",
            "left_foot.stl",
            "right_thigh.stl",
            "right_shin.stl",
            "right_foot.stl",
        ]

        for mesh_name in temporary_meshes:
            mesh_path = mesh_dir / mesh_name
            if not mesh_path.exists():
                # Create minimal STL file (just header for now)
                with open(mesh_path, "w") as f:
                    f.write("solid temporary\n")
                    f.write("endsolid temporary\n")

    def get_golf_club_path(self, club_type: str = "driver") -> Path:
        """Get path to golf club URDF.

        Args:
            club_type: Type of golf club (driver, iron_7, etc.)

        Returns:
            Path to golf club URDF file
        """
        if club_type not in self.config["golf_clubs"]:
            raise GolfModelingError(f"Unknown golf club type: {club_type}")

        club_config: dict[str, Any] = self.config["golf_clubs"][club_type]
        urdf_path = self.suite_root / str(club_config["urdf_path"])

        if not urdf_path.exists():
            self._generate_golf_club_urdf(club_type, urdf_path)

        return urdf_path

    def _generate_golf_club_urdf(self, club_type: str, output_path: Path) -> None:
        """Generate golf club URDF file."""
        club_config: dict[str, Any] = self.config["golf_clubs"][club_type]

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate URDF content
        urdf_content = f"""<?xml version="1.0"?>
<robot name="{club_config["name"].lower().replace(" ", "_")}">

  <!-- Base link (grip) -->
  <link name="grip">
    <visual>
      <geometry>
        <cylinder radius="0.015" length="0.25"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.015" length="0.25"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.050"/>
      <inertia ixx="0.0001" iyy="0.0001" izz="0.00001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Shaft -->
  <link name="shaft">
    <visual>
      <geometry>
        <cylinder radius="0.006" length="{club_config["length_m"] - 0.35:.3f}"/>
      </geometry>
      <material name="steel">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.006" length="{club_config["length_m"] - 0.35:.3f}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.060"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.00001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Club head -->
  <link name="head">
    <visual>
      <geometry>
        <box size="0.10 0.06 0.04"/>
      </geometry>
      <material name="titanium">
        <color rgba="0.9 0.9 0.9 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.10 0.06 0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{club_config["mass_kg"] - 0.110:.3f}"/>
      <inertia ixx="0.0005" iyy="0.0008" izz="0.0005" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="grip_to_shaft" type="fixed">
    <parent link="grip"/>
    <child link="shaft"/>
    <origin xyz="0 0 0.125" rpy="0 0 0"/>
  </joint>

  <joint name="shaft_to_head" type="fixed">
    <parent link="shaft"/>
    <child link="head"/>
    <origin xyz="0 0 {(club_config["length_m"] - 0.35) / 2:.3f}"
            rpy="0 {club_config["loft_deg"] * DEG_TO_RAD:.4f} 0"/>
  </joint>

</robot>"""

        with open(output_path, "w") as f:
            f.write(urdf_content)

        logger.info(f"Generated {club_type} URDF at {output_path}")

    def validate_model_compatibility(self, urdf_path: Path) -> dict[str, bool]:
        """Validate that a URDF model works with all physics engines.

        Args:
            urdf_path: Path to URDF file to validate

        Returns:
            Dictionary mapping engine names to compatibility status
        """
        results = {}

        # Test MuJoCo compatibility
        try:
            import mujoco

            # Convert URDF to MJCF and test loading
            mujoco.MjModel.from_xml_path(str(urdf_path))
            results["mujoco"] = True
        except Exception as e:
            logger.warning(f"MuJoCo compatibility issue: {e}")
            results["mujoco"] = False

        # Test Drake compatibility
        try:
            from pydrake.multibody.parsing import Parser
            from pydrake.multibody.plant import MultibodyPlant

            plant = MultibodyPlant(time_step=0.001)
            parser = Parser(plant)
            parser.AddModelFromFile(str(urdf_path))
            plant.Finalize()
            results["drake"] = True
        except Exception as e:
            logger.warning(f"Drake compatibility issue: {e}")
            results["drake"] = False

        # Test Pinocchio compatibility
        try:
            import pinocchio as pin

            pin.buildModelFromUrdf(str(urdf_path))
            results["pinocchio"] = True
        except Exception as e:
            logger.warning(f"Pinocchio compatibility issue: {e}")
            results["pinocchio"] = False

        return results

    def list_available_models(self) -> dict[str, Any]:
        """List all available standard models.

        Returns:
            Dictionary of available models with metadata
        """
        return {
            "humanoid": {
                "standard": self.config["standard_humanoid"],
                "simple": self.config["simple_humanoid"],
            },
            "golf_clubs": self.config["golf_clubs"],
        }

    def setup_all_models(self) -> bool:
        """Download and setup all standard models.

        Returns:
            True if all models setup successfully
        """
        success = True

        # Download standard humanoid
        if not self.download_standard_humanoid():
            success = False

        # Generate golf club URDFs
        for club_type in self.config["golf_clubs"]:
            try:
                self.get_golf_club_path(club_type)
            except Exception as e:
                logger.error(f"Failed to setup {club_type}: {e}")
                success = False

        return success
