"""Optional lumbar:thoracic coupling of the full-body-v2 trunk (SHOW-A3, #14552).

Per axis, ``thoracic = (thoracic share / lumbar share) * lumbar`` with shares
from ``src/config/spine_coupling_laws.json`` (Bruno 2015) via the migration's
:class:`RegionalPrior`. The coupling is declarative spec data
(``trunk_coupling``, OFF by default); an exporter realises it as one joint
equality per axis through :func:`trunk_coupling_equalities`. Both coordinates
of an axis share an ISB sign, so the model ratio equals the ISB one.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.full_body_migration import RegionalPrior
from src.shared.python.motion_matching.full_body_schema import ISB_SPINE_AXES
from src.shared.python.motion_matching.full_body_v2 import (
    FULL_BODY_V2_SCHEMA_VERSION,
    V2_TRUNK,
)

__all__ = ["trunk_coupling_equalities", "trunk_coupling_ratios", "with_trunk_coupling"]

#: Coordinate-name stem of each ISB axis (``lumbar_<stem>``, ``thoracic_<stem>``).
AXIS_STEM = {
    "flexion_extension": "flexion",
    "lateral_bending": "lateral_bending",
    "axial_rotation": "axial_rotation",
}


def trunk_coupling_ratios(prior: RegionalPrior) -> dict[str, float]:
    """Thoracic-per-lumbar angle ratio by axis stem, from the regional split.

    Postcondition: one finite positive ratio per axis, ``(1 - s) / s`` with
    ``s`` the lumbar share (the prior guarantees ``0 < s < 1``).
    """
    require(isinstance(prior, RegionalPrior), "prior must be a RegionalPrior")
    out = {
        AXIS_STEM[axis]: (1.0 - prior.lumbar_share[axis]) / prior.lumbar_share[axis]
        for axis in ISB_SPINE_AXES
    }
    ensure(all(r > 0.0 for r in out.values()), "coupling ratios must be positive")
    return out


def with_trunk_coupling(
    spec: Mapping[str, Any],
    *,
    enabled: bool = False,
    law_id: str = "bruno2015_regional",
) -> dict[str, Any]:
    """Copy of a ``full-body-v2`` spec carrying a ``trunk_coupling`` record.

    Preconditions: ``spec`` is ``full-body-v2`` (``ValueError``), ``enabled``
    is a ``bool`` (``TypeError``), ``law_id`` names an available regional
    split (``ValueError``). The input is never modified. With
    ``enabled=False`` (the default) the record is informative only.
    """
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    if spec.get("schema_version") != FULL_BODY_V2_SCHEMA_VERSION:
        raise ValueError("trunk coupling needs a full-body-v2 spec")
    prior = RegionalPrior.from_coupling_law(law_id)
    ratios = trunk_coupling_ratios(prior)
    names = set(V2_TRUNK.coordinates)
    constraints = []
    for stem, ratio in ratios.items():
        thoracic, lumbar = f"thoracic_{stem}", f"lumbar_{stem}"
        ensure({thoracic, lumbar} <= names, "coupled coordinates are v2 trunk")
        constraints.append(
            {
                "joint1": thoracic,
                "joint2": lumbar,
                "polycoef": [0.0, ratio, 0.0, 0.0, 0.0],
            }
        )
    out = copy.deepcopy(dict(spec))
    out["trunk_coupling"] = {
        "enabled": enabled,
        "law_id": prior.law_id,
        "law_version": prior.version,
        "ratios": ratios,
        "constraints": constraints,
    }
    return out


def trunk_coupling_equalities(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Joint equalities to emit for ``spec``: empty unless coupling is enabled.

    Each item is ``{"joint1", "joint2", "polycoef"}`` meaning
    ``q[joint1] = polycoef . [1, q[joint2], q[joint2]^2, ...]``.
    """
    record = spec.get("trunk_coupling")
    if not isinstance(record, Mapping) or record.get("enabled") is not True:
        return []
    return [dict(c) for c in record["constraints"]]
