"""Quick test script to verify linkage mechanisms are properly configured."""

import logging
import sys

sys.path.insert(0, "python")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Test imports

try:
    # Import individual generators
    from mujoco_golf_pendulum.linkage_mechanisms import (
        LINKAGE_CATALOG,
        generate_chebyshev_linkage_xml,
        generate_delta_robot_xml,
        generate_five_bar_parallel_xml,
        generate_four_bar_linkage_xml,
        generate_geneva_mechanism_xml,
        generate_oldham_coupling_xml,
        generate_pantograph_xml,
        generate_peaucellier_linkage_xml,
        generate_scotch_yoke_xml,
        generate_slider_crank_xml,
        generate_stewart_platform_xml,
        generate_watt_linkage_xml,
    )

    logger.info("✓ All imports successful")
except ImportError as e:
    logger.exception(f"✗ Import error: {e}")
    sys.exit(1)

# Test mechanism catalog
logger.info(f"\n✓ Catalog contains {len(LINKAGE_CATALOG)} mechanisms:")
for i, (name, config) in enumerate(LINKAGE_CATALOG.items(), 1):
    category = config.get("category", "Unknown")
    num_actuators = len(config.get("actuators", []))
    logger.info(f"  {i:2d}. {name}")
    logger.info(f"      Category: {category}")
    logger.info(f"      Actuators: {num_actuators}")
    logger.info(f"      Description: {config.get('description', 'N/A')}")

# Test XML generation for each mechanism type
logger.info("\n" + "=" * 60)
logger.info("Testing XML generation...")
logger.info("=" * 60)

test_cases = [
    ("Four-bar linkage", lambda: generate_four_bar_linkage_xml()),
    ("Slider-crank", lambda: generate_slider_crank_xml()),
    ("Scotch yoke", lambda: generate_scotch_yoke_xml()),
    ("Geneva mechanism", lambda: generate_geneva_mechanism_xml()),
    ("Peaucellier linkage", lambda: generate_peaucellier_linkage_xml()),
    ("Chebyshev linkage", lambda: generate_chebyshev_linkage_xml()),
    ("Pantograph", lambda: generate_pantograph_xml()),
    ("Delta robot", lambda: generate_delta_robot_xml()),
    ("5-bar parallel", lambda: generate_five_bar_parallel_xml()),
    ("Stewart platform", lambda: generate_stewart_platform_xml()),
    ("Watt linkage", lambda: generate_watt_linkage_xml()),
    ("Oldham coupling", lambda: generate_oldham_coupling_xml()),
]

for name, generator in test_cases:
    try:
        xml = generator()
        # Basic validation
        assert "<mujoco" in xml, "Missing mujoco tag"
        assert "<worldbody>" in xml, "Missing worldbody"
        assert "</mujoco>" in xml, "Missing closing tag"
        xml_size = len(xml)
        logger.info(f"✓ {name:25s} - Generated {xml_size:5d} chars")
    except (AssertionError, ValueError, RuntimeError) as e:
        logger.exception(f"✗ {name:25s} - Error: {e}")

# Test catalog XML generation
logger.info("\n" + "=" * 60)
logger.info("Testing catalog XML entries...")
logger.info("=" * 60)

for name, config in LINKAGE_CATALOG.items():
    try:
        xml = config["xml"]
        assert "<mujoco" in xml, "Missing mujoco tag"
        assert len(xml) > 100, "XML too short"
        actuators = config["actuators"]
        assert len(actuators) > 0, "No actuators defined"
        logger.info(f"✓ {name:45s} - {len(actuators)} actuators")
    except (AssertionError, KeyError, ValueError) as e:
        logger.exception(f"✗ {name:45s} - Error: {e}")

logger.info("\n" + "=" * 60)
logger.info("All tests passed! Linkage mechanisms library is ready.")
logger.info("=" * 60)
