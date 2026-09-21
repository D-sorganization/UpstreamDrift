"""Length scaling of individual segments in a full-body specification.

Generic lower limbs never match a specific golfer. Scaling a segment by ``s``
about its own frame origin multiplies by ``s`` the translation of every joint
it carries (its distal joints), the centres of mass and placements of its
solids and any contact sphere or marker attachment fixed to it, and the
inertia tensors by ``s**2`` (length scaling at constant mass, an accepted
approximation that the provenance records). The result is a new document
with its own hash; the input is not modified.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _scaled_transform(value: Any, scale: float) -> list[list[float]]:
    m = np.asarray(value, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError("Expected a finite 4x4 transform")
    out = m.copy()
    out[:3, 3] *= scale
    return out.tolist()


def scale_segments(
    document: Mapping[str, Any], scales: Mapping[str, float]
) -> dict[str, Any]:
    """Return a copy of ``document`` with the named bodies scaled in length.

    Preconditions: every key of ``scales`` names a body of the document; every
    scale is finite and positive. Postcondition: joints whose parent is a
    scaled body have their ``parent_to_base`` translation multiplied by the
    scale; the body's solids, contact spheres and marker offsets are scaled
    the same way; nothing else changes.
    """
    bodies = {b["name"]: b for b in document["bodies"]}
    for name, scale in scales.items():
        if name not in bodies:
            raise ValueError(f"Unknown body to scale: {name}")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Scale for {name} must be finite and positive")
    if not scales:
        return dict(document)

    def factor(name: str) -> float:
        return float(scales.get(name, 1.0))

    new_bodies = []
    for body in document["bodies"]:
        s = factor(body["name"])
        if s == 1.0:
            new_bodies.append(body)
            continue
        solids = []
        for solid in body["solids"]:
            entry = dict(solid)
            entry["com_m"] = (s * np.asarray(solid["com_m"], dtype=float)).tolist()
            entry["inertia_com_kg_m2"] = (
                s * s * np.asarray(solid["inertia_com_kg_m2"], dtype=float)
            ).tolist()
            entry["placement"] = _scaled_transform(solid["placement"], s)
            solids.append(entry)
        new_bodies.append({**body, "solids": solids})

    new_joints = []
    for joint in document["joints"]:
        s = factor(joint["parent"])
        if s == 1.0:
            new_joints.append(joint)
        else:
            new_joints.append(
                {
                    **joint,
                    "parent_to_base": _scaled_transform(joint["parent_to_base"], s),
                }
            )

    contact = dict(document["contact"])
    contact["spheres"] = [
        {
            **sphere,
            "position_m": (
                factor(sphere["body"]) * np.asarray(sphere["position_m"], dtype=float)
            ).tolist(),
        }
        for sphere in document["contact"]["spheres"]
    ]
    attachments = {}
    for label, attachment in document["marker_attachments"].items():
        s = factor(attachment["body"])
        offset = attachment.get("offset_m")
        if s != 1.0 and offset is not None:
            offset = (s * np.asarray(offset, dtype=float)).tolist()
        attachments[label] = {**attachment, "offset_m": offset}

    out = dict(document)
    out["bodies"] = new_bodies
    out["joints"] = new_joints
    out["contact"] = contact
    out["marker_attachments"] = attachments
    out["provenance"] = (
        str(document.get("provenance", ""))
        + " | segments scaled "
        + (
            ", ".join(f"{name} x{scale:.4f}" for name, scale in sorted(scales.items()))
            + " (lengths, COMs, sphere and marker offsets by s; inertias by s^2; masses kept)"
        )
    )
    return out
