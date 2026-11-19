"""Eyes, brows, lids and mouth placed from the CC0 skin mesh landmarks (#15276).

The MakeHuman head carries sculpted lids, nose, lips and ears but no eyeballs
or colour breaks, so it reads as a blank mannequin. These parts are built in
head-length units from :class:`FaceLandmarks` and conformed to the local skin
surface with a nearest-vertex lookup, so each sits on the face without
floating or sinking. Visual only, like the rest of the head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from src.shared.python.model_appearance.geometry import Array, Mesh, ellipsoid_mesh

_X, _Y = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class FaceLandmarks:
    """Feature placement in head lengths (x forward, y left, z up).

    The sculpted lids of the CC0 head are nearly closed, so the eye is a thin
    white almond conformed to the face with the iris, pupil and an upper lid
    line standing proud of it, each by its own lift off the skin.
    """

    eye_y: float = 0.118  # half the pupillary distance, from the sculpted slit
    eye_z: float = 0.532
    eye_half: tuple[float, float, float] = (0.006, 0.060, 0.030)  # x, y, z
    eye_lift: float = 0.004
    eye_smooth: float = 3.0  # wide stencil; eye parts are rigid, not conformed
    iris_half: tuple[float, float, float] = (0.006, 0.020, 0.020)
    iris_drop: float = 0.003  # iris centre below the eye centre
    iris_lift: float = 0.008
    pupil_half: tuple[float, float, float] = (0.005, 0.0085, 0.0085)
    pupil_lift: float = 0.0115
    lid_rise: float = 0.022  # upper lid line above the eye centre
    lid_half: tuple[float, float, float] = (0.005, 0.050, 0.005)
    lid_lift: float = 0.0085
    brow_rise: float = 0.070
    brow_half: tuple[float, float, float] = (0.010, 0.072, 0.012)
    brow_tilt: float = 0.16  # outer end rises by this fraction of its length
    brow_lift: float = 0.004
    mouth_z: float = 0.276
    mouth_half: tuple[float, float, float] = (0.008, 0.075, 0.006)
    mouth_lift: float = 0.003


class FaceSurface:
    """Outer skin depth x(y, z) of the face front, linearly interpolated.

    The head mesh also holds eye-socket and lip interiors behind the visible
    skin, so only the outermost vertex per ``envelope_cell`` is kept; a small
    stencil average then removes the jitter of the 4 mm triangles.
    """

    def __init__(
        self,
        skin: Mesh,
        envelope_cell: float = 0.012,
        blur: float = 0.006,
        min_x: float = 0.2,
    ) -> None:
        front = skin.vertices[skin.vertices[:, 0] > min_x]
        outer: dict[tuple[int, int], Array] = {}
        for p in front:
            cell = (round(p[1] / envelope_cell), round(p[2] / envelope_cell))
            if cell not in outer or p[0] > outer[cell][0]:
                outer[cell] = p
        pts = np.array(list(outer.values()))
        self._linear = LinearNDInterpolator(pts[:, 1:3], pts[:, 0])
        self._nearest = NearestNDInterpolator(pts[:, 1:3], pts[:, 0])
        self._stencil = blur * np.array(
            [[0.0, 0.0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float
        )

    def depth(self, y: float, z: float, scale: float = 1.0) -> float:
        """Skin depth x at (y, z), blurred over a five-point stencil.

        ``scale`` widens the stencil for features (the eyes) that must ignore
        the sculpted detail under them.
        """
        q = scale * self._stencil + np.array([y, z])
        vals = self._linear(q)
        bad = np.isnan(vals)
        vals[bad] = self._nearest(q[bad])
        return float(vals.mean())

    def conform(
        self, mesh: Mesh, centre: Array, lift: float, scale: float = 1.0
    ) -> Mesh:
        """Shift each vertex in x by how far the skin there is from the centre."""
        base = self.depth(centre[1], centre[2], scale)
        out = mesh.vertices.copy()
        out[:, 0] += [self.depth(p[1], p[2], scale) - base for p in mesh.vertices]
        out[:, 0] += lift
        return Mesh(out, mesh.faces)


def _blob(centre: Array, half_xyz: tuple[float, float, float], axis: Array) -> Mesh:
    """Ellipsoid with ``(x, y, z)`` half sizes whose long axis is x or y-ish."""
    hx, hy, hz = half_xyz
    if abs(axis[0]) > 0.9:  # along x: sizes (x, y, z), width runs along y
        return ellipsoid_mesh(
            centre, np.array([hx, hy, hz]), axis, rings=6, sides=12, width_hint=_Y
        )
    return ellipsoid_mesh(  # along y: sizes (y, x, z), width runs along x
        centre, np.array([hy, hx, hz]), axis, rings=6, sides=12, width_hint=_X
    )


def face_parts(
    skin: Mesh, marks: FaceLandmarks | None = None
) -> list[tuple[str, Mesh, str]]:
    """``(name, mesh, material role)`` of eyes, lids, brows and mouth.

    ``skin`` is the unit head-length skin mesh. Postcondition: every mesh is a
    closed ellipsoid of positive volume, left (y > 0) and right mirrored.
    """
    m = marks or FaceLandmarks()
    surf = FaceSurface(skin)

    def decal(centre_yz, half, axis, lift, scale=1.0):
        c = np.array([surf.depth(centre_yz[0], centre_yz[1], scale), *centre_yz])
        if scale != 1.0:  # rigid: the sculpted lids are too coarse to follow
            return _blob(c + _X * lift, half, axis)
        return surf.conform(_blob(c, half, axis), c, lift)

    out: list[tuple[str, Mesh, str]] = []
    for side, sign in (("l", 1.0), ("r", -1.0)):
        y, z = sign * m.eye_y, m.eye_z
        out.append(
            (
                f"eye_{side}",
                decal((y, z), m.eye_half, _X, m.eye_lift, m.eye_smooth),
                "eye_white",
            )
        )
        iris = decal((y, z - m.iris_drop), m.iris_half, _X, m.iris_lift, m.eye_smooth)
        out.append((f"iris_{side}", iris, "iris"))
        pupil = decal(
            (y, z - m.iris_drop), m.pupil_half, _X, m.pupil_lift, m.eye_smooth
        )
        out.append((f"pupil_{side}", pupil, "pupil"))
        lid = decal((y, z + m.lid_rise), m.lid_half, _Y, m.lid_lift, m.eye_smooth)
        out.append((f"lid_{side}", lid, "lid"))
        tilt = np.array([0.0, 1.0, sign * m.brow_tilt])  # outer end rises
        brow = decal((y, z + m.brow_rise), m.brow_half, tilt, m.brow_lift)
        out.append((f"brow_{side}", brow, "brow"))
    mouth = decal((0.0, m.mouth_z), m.mouth_half, _Y, m.mouth_lift)
    out.append(("mouth", mouth, "lip_seam"))
    return out
