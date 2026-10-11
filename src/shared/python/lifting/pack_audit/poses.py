"""Same-input pose sets used to compare the four engines.

Every pose is a mapping of canonical coordinate to radians, applied relative
to the all-zero pose of the biomech parity standard.  Lift poses are generic
shapes (hip hinge, rack, overhead), not any pack's phase data, so no pack is
privileged.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

from .names import CANONICAL_COORDINATES, SIDES
from .packs import PackLocation


def _both(pattern: str, deg: float) -> dict[str, float]:
    return {pattern.format(s=s): math.radians(deg) for s in SIDES}


def lift_shape_poses() -> dict[str, dict[str, float]]:
    """Generic lifting shapes, each using canonical sign conventions."""
    hinge = {
        **_both("hip_{s}_flex", 80),
        **_both("knee_{s}_flex", -60),
        **_both("ankle_{s}_flex", 15),
        "lumbar_flex": math.radians(15),
    }
    rack = {
        **_both("shoulder_{s}_flex", 70),
        **_both("elbow_{s}_flex", 140),
        **_both("wrist_{s}_flex", 30),
    }
    overhead = {
        **_both("shoulder_{s}_flex", 170),
        **_both("elbow_{s}_flex", 0),
    }
    squat_bottom = {
        **_both("hip_{s}_flex", 110),
        **_both("knee_{s}_flex", -120),
        **_both("ankle_{s}_flex", 25),
        "lumbar_flex": math.radians(10),
    }
    for pose in (hinge, rack, overhead, squat_bottom):
        unknown = set(pose) - set(CANONICAL_COORDINATES)
        if unknown:
            raise ValueError(f"pose uses non-canonical coordinates {sorted(unknown)}")
    return {
        "hip_hinge": hinge,
        "front_rack": rack,
        "overhead": overhead,
        "squat_bottom": squat_bottom,
    }


def load_standard(pack: PackLocation) -> dict[str, Any]:
    """The pack's vendored parity standard (byte-identical across the packs)."""
    pack.activate()
    conformance = importlib.import_module(
        f"{pack.package}.shared.parity._canonical.conformance"
    )
    return dict(conformance.load_standard())


def standard_poses(
    pack: PackLocation, std: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """The standard's flexion/frontal/axial test poses, in radians."""
    topology = importlib.import_module(
        f"{pack.package}.shared.parity._canonical.topology"
    )
    return dict(topology.standard_poses(std))


def reference_origins(
    pack: PackLocation, std: dict[str, Any], q: dict[str, float], height_m: float
) -> dict[str, tuple[float, float, float]]:
    """The standard's reference FK (pelvis frame) scaled to *height_m*."""
    if height_m <= 0:
        raise ValueError("height_m must be positive")
    topology = importlib.import_module(
        f"{pack.package}.shared.parity._canonical.topology"
    )
    scaled = {
        **std,
        "anthropometrics": {**std["anthropometrics"], "height_m": height_m},
    }
    return dict(topology.reference_origins(scaled, q))
