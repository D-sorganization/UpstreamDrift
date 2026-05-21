# ruff: noqa: T201
"""
Lower Body Model - MuJoCo Simulation Viewer
"""

import argparse
import logging

from .builder import build_lower_body_xml
from .simulator import LowerBodySimulator

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lower Body Model Simulation")
    parser.add_argument(
        "--gui", action="store_true", help="Launch MuJoCo interactive viewer"
    )
    parser.add_argument(
        "--demo-iaa",
        action="store_true",
        help="Run a demo Induced Acceleration Analysis",
    )
    args = parser.parse_args()

    xml = build_lower_body_xml()
    sim = LowerBodySimulator(xml)

    # Initialize posture and set stability targets
    sim.setup_initial_pose(hip_anterior_tilt=20.0, knee_flexion=30.0, foot_angle=20.0)

    if args.demo_iaa:
        logger.info(
            "Running Induced Acceleration Analysis on right hip (X axis) with 10.0 Nm torque..."
        )
        iaa = sim.analyze_induced_acceleration("act_r_hip_x", 10.0)
        logger.info("Pelvis Induced Accelerations:")
        for k, v in iaa.items():
            logger.info(f"  {k}: {v:.4f}")

    if args.gui:
        try:
            import mujoco.viewer

            logger.info("Launching MuJoCo Viewer. Press ESC to exit.")
            mujoco.viewer.launch(sim.model, sim.data)
        except ImportError:
            logger.warning(
                "mujoco.viewer not available. Make sure you install correctly."
            )

    if not args.gui and not args.demo_iaa:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
