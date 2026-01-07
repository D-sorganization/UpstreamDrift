"""Quick test script to verify new models load correctly."""

import logging
import sys

sys.path.insert(0, "python")

import mujoco as mj
from mujoco_golf_pendulum.models import (
    ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
    CLUB_CONFIGS,
    GIMBAL_JOINT_DEMO_XML,
    TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML,
    generate_flexible_club_xml,
    generate_rigid_club_xml,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def test_model(name: str, xml_string: str) -> bool:
    """Test that a model loads without errors."""
    try:
        model = mj.MjModel.from_xml_string(xml_string)
        data = mj.MjData(model)
        mj.mj_step(model, data)
    except (ValueError, RuntimeError) as e:
        logger.exception(f"✗ {name}: FAILED - {e}")
        return False
    else:
        logger.info(f"✓ {name}: OK ({model.nq} DOF, {model.njnt} joints)")
        return True


def main() -> int:
    """Test all new models."""
    logger.info("=" * 70)
    logger.info("Testing New Joint Models")
    logger.info("=" * 70)

    results = []

    # Test basic models
    logger.info("\n1. Basic Models:")
    results.append(
        test_model("Two-Link Universal Joint", TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML),
    )
    results.append(test_model("Gimbal Joint Demo", GIMBAL_JOINT_DEMO_XML))
    results.append(
        test_model("Advanced Biomechanical", ADVANCED_BIOMECHANICAL_GOLF_SWING_XML),
    )

    # Test club configurations
    logger.info("\n2. Club Configurations:")
    for club_type in CLUB_CONFIGS:
        try:
            generate_rigid_club_xml(club_type)
            generate_flexible_club_xml(club_type, num_segments=3)
            logger.info(f"✓ {club_type}: Generated rigid and flexible variants")
            results.append(True)
        except (ValueError, RuntimeError, KeyError) as e:
            logger.exception(f"✗ {club_type}: FAILED - {e}")
            results.append(False)

    # Summary
    logger.info("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✓ All tests passed!")
        return 0
    logger.error("✗ Some tests failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
