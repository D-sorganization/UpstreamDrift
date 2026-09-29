"""Course geometry, conditions, and rasterization for the strategy MDP."""

from __future__ import annotations

from src.shared.python.sg_optimizer.course.conditions import (
    CourseConditions,
    GreenModel,
    RoughModel,
    TreeModel,
)
from src.shared.python.sg_optimizer.course.course_io import (
    HoleGeometry,
    load_hole_geojson,
    save_hole_geojson,
)
from src.shared.python.sg_optimizer.course.features import StateFeatures
from src.shared.python.sg_optimizer.course.geometry import (
    LatLonPoint,
    UTMPoint,
    haversine_m,
    project_to_utm,
    utm_to_latlon,
)
from src.shared.python.sg_optimizer.course.library import list_classics, load_classic
from src.shared.python.sg_optimizer.course.rasterize import (
    LIE_CODES,
    LIE_NAMES,
    LIE_PRIORITY,
    LieRaster,
    SyntheticHole,
    rasterize_synthetic,
)

__all__ = [
    # Phase 1
    "CourseConditions",
    "GreenModel",
    "LIE_CODES",
    "LIE_NAMES",
    "LIE_PRIORITY",
    "LieRaster",
    "RoughModel",
    "SyntheticHole",
    "TreeModel",
    "rasterize_synthetic",
    # Phase 2
    "HoleGeometry",
    "LatLonPoint",
    "StateFeatures",
    "UTMPoint",
    "haversine_m",
    "list_classics",
    "load_classic",
    "load_hole_geojson",
    "project_to_utm",
    "save_hole_geojson",
    "utm_to_latlon",
]
