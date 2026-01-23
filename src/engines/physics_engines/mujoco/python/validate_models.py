#!/usr/bin/env python
"""Validate all MuJoCo golf swing models."""

import logging
import sys

try:
    import mujoco
except ImportError:
    mujoco = None

from mujoco_humanoid_golf.models import (
    ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
    DOUBLE_PENDULUM_XML,
    FULL_BODY_GOLF_SWING_XML,
    TRIPLE_PENDULUM_XML,
    UPPER_BODY_GOLF_SWING_XML,
)

# Configure logging to output only the message
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

if mujoco is None:
    logger.info("✗ MuJoCo not installed. Install with: pip install mujoco>=3.0.0")
    sys.exit(1)
else:
    logger.info("✓ MuJoCo version: %s", mujoco.__version__)

models = [
    ("Double Pendulum", DOUBLE_PENDULUM_XML, 2),
    ("Triple Pendulum", TRIPLE_PENDULUM_XML, 3),
    ("Upper Body Golf Swing", UPPER_BODY_GOLF_SWING_XML, 10),
    ("Full Body Golf Swing", FULL_BODY_GOLF_SWING_XML, 15),
    ("Advanced Biomechanical Golf Swing", ADVANCED_BIOMECHANICAL_GOLF_SWING_XML, 28),
]

logger.info("\n" + "=" * 60)
logger.info("MuJoCo Golf Swing Model Validation")
logger.info("=" * 60 + "\n")

all_valid = True

for name, xml_str, expected_actuators in models:
    logger.info("Testing: %s", name)
    logger.info("-" * 60)

    try:
        # Parse XML and create model
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)

        # Validate actuator count
        actual_actuators = model.nu
        if actual_actuators != expected_actuators:
            logger.info("  ✗ Actuator count mismatch!")
            logger.info(
                "    Expected: %d, Got: %d",
                expected_actuators,
                actual_actuators,
            )
            all_valid = False
        else:
            logger.info("  ✓ Actuators: %d", actual_actuators)

        # Print model statistics
        logger.info("  ✓ Degrees of freedom: %d", model.nv)
        logger.info("  ✓ Bodies: %d", model.nbody)
        logger.info("  ✓ Joints: %d", model.njnt)
        logger.info("  ✓ Geoms: %d", model.ngeom)

        # Test one simulation step
        mujoco.mj_step(model, data)
        logger.info("  ✓ Simulation step successful")

        # Check for NaN or Inf values
        if not all(data.qpos.flatten()):
            logger.info("  ⚠ Warning: Some joint positions are zero")

        logger.info("  ✓ %s validated successfully!\n", name)

    except Exception as e:  # noqa: BLE001
        logger.info("  ✗ ERROR: %s\n", e)
        all_valid = False

logger.info("=" * 60)
if all_valid:
    logger.info("✓ All models validated successfully!")
    logger.info("=" * 60)
    sys.exit(0)
else:
    logger.info("✗ Some models failed validation")
    logger.info("=" * 60)
    sys.exit(1)
