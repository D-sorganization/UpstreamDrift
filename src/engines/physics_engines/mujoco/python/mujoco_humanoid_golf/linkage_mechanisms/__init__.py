"""Linkage Mechanisms Library for MuJoCo Physics Exploration.

This package provides a comprehensive collection of classic linkage mechanisms
and parallel manipulators for educational and research purposes.

Architecture (#1485):
    Logic has been decomposed from this __init__.py into sub-modules:
    - four_bar.py: Four-bar linkage variants
    - slider_mechanisms.py: Slider-crank, Scotch yoke
    - special_mechanisms.py: Geneva drive, Oldham coupling
    - straight_line.py: Peaucellier, Chebyshev, Watt linkages
    - parallel_mechanisms.py: Delta robot, Stewart platform, 5-bar, pantograph
    - catalog.py: LINKAGE_CATALOG for GUI integration

    This __init__.py contains only re-exports, no logic.
"""

from .catalog import LINKAGE_CATALOG
from .four_bar import generate_four_bar_linkage_xml
from .parallel_mechanisms import (
    generate_delta_robot_xml,
    generate_five_bar_parallel_xml,
    generate_pantograph_xml,
    generate_stewart_platform_xml,
)
from .slider_mechanisms import generate_scotch_yoke_xml, generate_slider_crank_xml
from .special_mechanisms import (
    generate_geneva_mechanism_xml,
    generate_oldham_coupling_xml,
)
from .straight_line import (
    generate_chebyshev_linkage_xml,
    generate_peaucellier_linkage_xml,
    generate_watt_linkage_xml,
)

__all__ = [
    "LINKAGE_CATALOG",
    "generate_chebyshev_linkage_xml",
    "generate_delta_robot_xml",
    "generate_five_bar_parallel_xml",
    "generate_four_bar_linkage_xml",
    "generate_geneva_mechanism_xml",
    "generate_oldham_coupling_xml",
    "generate_pantograph_xml",
    "generate_peaucellier_linkage_xml",
    "generate_scotch_yoke_xml",
    "generate_slider_crank_xml",
    "generate_stewart_platform_xml",
    "generate_watt_linkage_xml",
]
