"""MyoConverter integration for OpenSim <-> MuJoCo conversion.

This module provides a Python interface to MyoConverter for converting
OpenSim musculoskeletal models to MuJoCo format and vice versa.

MyoConverter Repository: https://github.com/MyoHub/myoconverter

Features:
- Convert OpenSim (.osim) models to MuJoCo (.xml) format
- Optimize muscle kinematics and kinetics
- Generate validation PDF reports
- Handle geometry and mesh files

Note: MyoConverter requires specific dependencies. Install via:
    conda install -c conda-forge myoconverter
    or follow Docker setup for Windows/MacOS
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MyoConverter:
    """Interface for OpenSim to MuJoCo model conversion."""

    def __init__(self) -> None:
        """Initialize MyoConverter interface."""
        self.myoconverter_available = self._check_availability()

        if not self.myoconverter_available:
            logger.warning(
                "MyoConverter not available. "
                "Install with: conda install -c conda-forge myoconverter"
            )

    def _check_availability(self) -> bool:
        """Check if myoconverter is installed.

        Returns:
            True if myoconverter is available, False otherwise
        """
        try:
            import myoconverter  # noqa: F401

            return True
        except ImportError:
            return False

    def convert_osim_to_mujoco(
        self,
        osim_file: Path,
        geometry_folder: Path,
        output_folder: Path,
        **kwargs: Any,
    ) -> Path | None:
        """Convert OpenSim model to MuJoCo format.

        Args:
            osim_file: Path to OpenSim .osim model file
            geometry_folder: Path to folder containing geometry/mesh files
            output_folder: Path to output folder for converted model
            **kwargs: Additional configuration options:
                - convert_steps: List of conversion steps [1, 2, 3]
                - muscle_list: List of specific muscles to optimize (None for all)
                - osim_data_overwrite: Overwrite OpenSim model state files
                - conversion: Perform 'Cvt#' conversion process
                - validation: Perform 'Vlt#' validation process
                - speedy: Reduce checking notes to increase speed
                - generate_pdf: Generate validation PDF report
                - add_ground_geom: Add ground geometry to model
                - treat_as_normal_path_point: Use constraints for moving path points

        Returns:
            Path to converted MuJoCo XML file, or None if conversion failed
        """
        if not self.myoconverter_available:
            logger.error("MyoConverter not installed. Cannot perform conversion.")
            return None

        try:
            from myoconverter.O2MPipeline import O2MPipeline

            # Default configuration
            default_config = {
                "convert_steps": [1, 2, 3],  # All three steps
                "muscle_list": None,  # Optimize all muscles
                "osim_data_overwrite": True,
                "conversion": True,
                "validation": True,
                "speedy": False,
                "generate_pdf": True,
                "add_ground_geom": True,
                "treat_as_normal_path_point": False,
            }

            # Merge with user-provided kwargs
            config = {**default_config, **kwargs}

            # Ensure paths are strings
            osim_file_str = str(osim_file)
            geometry_folder_str = str(geometry_folder)
            output_folder_str = str(output_folder)

            logger.info(f"Converting OpenSim model: {osim_file}")
            logger.info(f"Output folder: {output_folder}")

            # Run conversion pipeline
            O2MPipeline(osim_file_str, geometry_folder_str, output_folder_str, **config)

            # Find converted XML file
            output_path = Path(output_folder)
            xml_files = list(output_path.glob("*_cvt*.xml"))

            if xml_files:
                converted_file = xml_files[0]
                logger.info(f"Conversion successful: {converted_file}")
                return converted_file
            else:
                logger.error("Conversion completed but no output XML file found")
                return None

        except ImportError as e:
            logger.error(f"Failed to import myoconverter: {e}")
            return None
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return None

    def load_converted_model_keyframe(self, model_path: Path) -> str:
        """Generate Python code to load converted model with keyframe.

        MyoConverter models require loading a keyframe to initialize joint values
        that satisfy all joint/muscle path constraints.

        Args:
            model_path: Path to converted MuJoCo XML file

        Returns:
            Python code snippet as string
        """
        code = f"""import mujoco

# Load converted model
model = mujoco.MjModel.from_xml_path("{model_path}")
data = mujoco.MjData(model)

# IMPORTANT: Load the keyframe to initialize joint values
# This ensures all joint/muscle path constraints are met
mujoco.mj_resetDataKeyframe(model, data, 0)

# Now the model is ready for simulation
"""
        return code

    def get_example_models(self) -> dict[str, str]:
        """Get dictionary of example models converted with MyoConverter.

        Returns:
            Dictionary mapping model names to their GitHub URLs
        """
        base_url = "https://github.com/MyoHub/myoconverter/blob/main/models/mjc"

        return {
            "tug_of_war": f"{base_url}/TugOfWar/tugofwar_cvt3.xml",
            "simple_arm": "https://github.com/MyoHub/myo_sim/blob/main/elbow/myoelbow_2dof6muscles.xml",
            "single_leg": f"{base_url}/Leg6Dof9Musc/leg6dof9musc_cvt3.xml",
            "gait_2d": f"{base_url}/Gait10dof18musc/gait10dof18musc_cvt3.xml",
            "gait_3d": f"{base_url}/Gait2354Simbody/gait2354_cvt3.xml",
            "neck_model": f"{base_url}/Neck6D/neck6d_cvt3.xml",
        }

    def validate_conversion(self, mujoco_xml: Path, osim_file: Path) -> bool:
        """Validate a converted MuJoCo model against original OpenSim model.

        Args:
            mujoco_xml: Path to converted MuJoCo XML file
            osim_file: Path to original OpenSim .osim file

        Returns:
            True if validation passes, False otherwise
        """
        # This would require implementing validation logic
        # For now, just check if files exist
        if not mujoco_xml.exists():
            logger.error(f"MuJoCo file not found: {mujoco_xml}")
            return False

        if not osim_file.exists():
            logger.error(f"OpenSim file not found: {osim_file}")
            return False

        logger.info("Basic file validation passed")
        return True


def install_myoconverter_instructions() -> str:
    """Provide installation instructions for MyoConverter.

    Returns:
        Multi-line string with installation instructions
    """
    instructions = """
    MyoConverter Installation Instructions
    =====================================

    Linux (Recommended):
    -------------------
    conda create -n myoconv python=3.9
    conda activate myoconv
    conda install -c conda-forge myoconverter

    Windows/MacOS (Docker):
    -----------------------
    1. Install Docker Desktop
    2. Follow instructions at:
       https://github.com/MyoHub/myoconverter/blob/main/docker/README.md

    Verification:
    -------------
    python -c "import myoconverter; print('MyoConverter installed successfully!')"

    Documentation:
    --------------
    https://myoconverter.readthedocs.io/en/latest/

    Citation:
    ---------
    If you use MyoConverter, please cite:
    - Wang et al. (2022), "MyoSim: Fast and physiologically realistic MuJoCo models"
    - Ikkala & Hämäläinen (2022), "Converting biomechanical models from OpenSim to Mujoco"
    """
    return instructions
