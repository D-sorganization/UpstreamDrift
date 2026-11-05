"""Club-head centre-of-gravity offset behind the face (SHOW-C4 follow-up, #14600).

``club_models`` places the head mass on the shaft axis (``com_m = [0, 0, 0]``),
so the chain shaft (``club_shaft_chain``) only bends from the head's inertia
load and never from the load acting through the head's real centre of gravity,
which sits behind the face (a driver's CG depth is of the order of 3-4 cm).
Under the centripetal load of the downswing that offset is a moment arm on the
shaft tip, the mechanism behind the toe-down deflection of drivers at impact
(Butler and Winfield 1994, as described in the secondary literature).

:func:`apply_head_cg_offset` moves the head solid's ``com_m`` to
``-behind_face_m * n`` where ``n`` is the rendered unit face normal in the
club-body frame (``clubface_vector``). It is opt-in so the default document,
and the hashes computed from it, are unchanged. Because the offset is on the
head solid, which the chain moves to the tip link intact, the rigid club and
the chain club carry the same head CG and the total mass and centre of mass
of the zero-deflection chain still equal the rigid club's.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.shared.python.contracts import ensure, require

HEAD_SOLID_SUFFIX = "/Clubhead"
MAX_BEHIND_FACE_M = 0.08  # beyond the head's own depth (driver half depth is 0.05 m)


def face_normal_in_club_frame(document: Mapping[str, Any]) -> np.ndarray:
    """Rendered unit face normal in the club-body frame of ``document``."""
    from src.shared.python.model_appearance import club_assembly as ca
    from src.shared.python.motion_matching.club_face_target import (
        require_club_assembly,
    )

    assembly, _ = require_club_assembly(document)
    normal = np.asarray(ca.clubface_vector(assembly), dtype=float)
    return normal / np.linalg.norm(normal)


def face_plane_axis_in_club_frame(document: Mapping[str, Any]) -> np.ndarray:
    """Unit axis ``face_normal x y`` of the club-body frame (y is the shaft axis).

    It lies in the face plane and is perpendicular to the shaft. Its sign is
    fixed by the frame algebra only; it is not tied to the physical toe or
    heel of the head, so callers name it by this axis.
    """
    axis = np.cross(face_normal_in_club_frame(document), (0.0, 1.0, 0.0))
    return np.asarray(axis / np.linalg.norm(axis), dtype=float)


def apply_head_cg_offset(
    document: Mapping[str, Any], *, behind_face_m: float, along_face_axis_m: float = 0.0
) -> dict[str, Any]:
    """Copy of ``document`` whose head solid has its CG ``behind_face_m`` behind the face.

    Preconditions: ``0 <= behind_face_m <= MAX_BEHIND_FACE_M`` (finite) and the
    document has exactly one ``*/Clubhead`` solid and a recognised club
    assembly (``ValueError`` otherwise).
    Postconditions: only that solid's ``com_m`` changes, to
    ``-behind_face_m * n + along_face_axis_m * (n x y)``; its mass and its inertia about its own CG are kept.
    """
    require(
        math.isfinite(behind_face_m) and 0.0 <= behind_face_m <= MAX_BEHIND_FACE_M,
        "behind_face_m must be in [0, MAX_BEHIND_FACE_M] metres",
        behind_face_m,
    )
    doc = json.loads(json.dumps(document))
    heads = [
        solid
        for body in doc["bodies"]
        for solid in body["solids"]
        if str(solid["name"]).endswith(HEAD_SOLID_SUFFIX)
    ]
    require(
        len(heads) == 1, "document must have exactly one Clubhead solid", len(heads)
    )
    offset = -behind_face_m * face_normal_in_club_frame(document)
    offset = offset + along_face_axis_m * face_plane_axis_in_club_frame(document)
    heads[0]["com_m"] = [float(v) for v in offset]
    ensure(
        math.isclose(
            float(np.linalg.norm(heads[0]["com_m"])),
            math.hypot(behind_face_m, along_face_axis_m),
            abs_tol=1e-12,
        ),
        "head CG offset must have the requested magnitude",
    )
    return doc
