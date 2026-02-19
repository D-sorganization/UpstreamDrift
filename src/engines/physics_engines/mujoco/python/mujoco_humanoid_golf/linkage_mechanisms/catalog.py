"""Catalog of all available linkage mechanisms for GUI integration.

Provides the LINKAGE_CATALOG dictionary that maps mechanism names to their
XML generators, actuators, categories, and descriptions.

Extracted from __init__.py for SRP (#1485).
"""

from .four_bar import generate_four_bar_linkage_xml
from .parallel_mechanisms import (
    generate_delta_robot_xml,
    generate_five_bar_parallel_xml,
    generate_pantograph_xml,
    generate_stewart_platform_xml,
)
from .slider_mechanisms import generate_scotch_yoke_xml, generate_slider_crank_xml
from .special_mechanisms import generate_geneva_mechanism_xml, generate_oldham_coupling_xml
from .straight_line import (
    generate_chebyshev_linkage_xml,
    generate_peaucellier_linkage_xml,
    generate_watt_linkage_xml,
)

LINKAGE_CATALOG = {
    "Four-Bar: Grashof Crank-Rocker": {
        "xml": generate_four_bar_linkage_xml(link_type="grashof_crank_rocker"),
        "actuators": ["crank_motor"],
        "category": "Four-Bar Linkages",
        "description": "Classic crank-rocker mechanism (Grashof condition satisfied)",
    },
    "Four-Bar: Double Crank": {
        "xml": generate_four_bar_linkage_xml(link_type="grashof_double_crank"),
        "actuators": ["crank_motor"],
        "category": "Four-Bar Linkages",
        "description": "Both output links can rotate fully (Grashof)",
    },
    "Four-Bar: Double Rocker": {
        "xml": generate_four_bar_linkage_xml(link_type="grashof_double_rocker"),
        "actuators": ["crank_motor"],
        "category": "Four-Bar Linkages",
        "description": "Both output links oscillate",
    },
    "Four-Bar: Parallelogram": {
        "xml": generate_four_bar_linkage_xml(link_type="parallel"),
        "actuators": ["crank_motor"],
        "category": "Four-Bar Linkages",
        "description": "Special case with parallel opposite links",
    },
    "Slider-Crank (Horizontal)": {
        "xml": generate_slider_crank_xml(orientation="horizontal"),
        "actuators": ["crank_motor"],
        "category": "Slider Mechanisms",
        "description": "Piston engine mechanism, converts rotation to linear motion",
    },
    "Slider-Crank (Vertical)": {
        "xml": generate_slider_crank_xml(orientation="vertical"),
        "actuators": ["crank_motor"],
        "category": "Slider Mechanisms",
        "description": "Vertical orientation slider-crank",
    },
    "Scotch Yoke": {
        "xml": generate_scotch_yoke_xml(),
        "actuators": ["crank_motor"],
        "category": "Slider Mechanisms",
        "description": "Perfect simple harmonic motion generator",
    },
    "Geneva Mechanism": {
        "xml": generate_geneva_mechanism_xml(),
        "actuators": ["drive_motor"],
        "category": "Special Mechanisms",
        "description": "Intermittent motion converter (used in film projectors)",
    },
    "Oldham Coupling": {
        "xml": generate_oldham_coupling_xml(),
        "actuators": ["input_motor"],
        "category": "Special Mechanisms",
        "description": "Couples parallel shafts with offset",
    },
    "Peaucellier-Lipkin Linkage": {
        "xml": generate_peaucellier_linkage_xml(),
        "actuators": ["drive_motor"],
        "category": "Straight-Line Mechanisms",
        "description": "Exact straight-line mechanism (mathematical perfection)",
    },
    "Chebyshev Linkage": {
        "xml": generate_chebyshev_linkage_xml(),
        "actuators": ["left_crank_motor"],
        "category": "Straight-Line Mechanisms",
        "description": "Approximate straight-line (walking mechanism)",
    },
    "Watt's Linkage": {
        "xml": generate_watt_linkage_xml(),
        "actuators": ["left_motor"],
        "category": "Straight-Line Mechanisms",
        "description": "Steam engine straight-line guidance (historical)",
    },
    "Pantograph": {
        "xml": generate_pantograph_xml(),
        "actuators": ["arm1_motor"],
        "category": "Scaling Mechanisms",
        "description": "Geometric scaling and copying mechanism",
    },
    "Delta Robot (3-DOF Parallel)": {
        "xml": generate_delta_robot_xml(),
        "actuators": ["motor1", "motor2", "motor3"],
        "category": "Parallel Mechanisms",
        "description": "High-speed pick-and-place robot",
    },
    "5-Bar Parallel Manipulator": {
        "xml": generate_five_bar_parallel_xml(),
        "actuators": ["left_motor", "right_motor"],
        "category": "Parallel Mechanisms",
        "description": "2-DOF planar parallel robot",
    },
    "Stewart Platform (6-DOF)": {
        "xml": generate_stewart_platform_xml(),
        "actuators": [
            "leg1_motor",
            "leg2_motor",
            "leg3_motor",
            "leg4_motor",
            "leg5_motor",
            "leg6_motor",
        ],
        "category": "Parallel Mechanisms",
        "description": "Flight simulator platform, full 6-DOF control",
    },
}
