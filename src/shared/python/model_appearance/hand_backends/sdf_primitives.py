"""Geometric helpers and SDF primitive builders for :mod:`sdf_pose`."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from src.shared.python.model_appearance.hand_backends.sdf_field import Primitive
from src.shared.python.model_appearance.hand_backends.sdf_settings import (
    SdfHandSettings,
)


def vec3(values: np.ndarray) -> tuple[float, float, float]:
    """Convert a 3-element array into a plain float tuple."""
    return (float(values[0]), float(values[1]), float(values[2]))


def unit(vector: np.ndarray) -> np.ndarray:
    """Normalise a non-zero vector."""
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("cannot normalise a near-zero vector")
    return arr / norm


def orthogonal_dir(direction: np.ndarray, axis_u: np.ndarray) -> np.ndarray:
    """Unit ``direction`` with its component along ``axis_u`` removed."""
    dir_a = np.asarray(direction, dtype=np.float64)
    dropped = dir_a - float(np.dot(dir_a, axis_u)) * axis_u
    norm = float(np.linalg.norm(dropped))
    if norm >= 1e-8:
        return dropped / norm
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, axis_u))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    crossed = np.asarray(np.cross(axis_u, helper), dtype=np.float64)
    return crossed / float(np.linalg.norm(crossed))


def radial_dir(theta: float) -> np.ndarray:
    """Unit radial vector in the XY plane perpendicular to the grip axis."""
    return np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)


def tangential_dir(theta: float) -> np.ndarray:
    """Unit tangential vector around the grip axis in the XY plane."""
    return np.array([-math.sin(theta), math.cos(theta), 0.0], dtype=np.float64)


def push_outside(point: np.ndarray, min_rho: float) -> np.ndarray:
    """Move ``point`` radially out to ``min_rho`` without changing z."""
    pushed = np.array(point, dtype=np.float64, copy=True)
    rho = float(np.hypot(pushed[0], pushed[1]))
    if rho >= min_rho:
        return pushed
    if rho < 1e-8:
        pushed[0] = min_rho
        return pushed
    pushed[0] *= min_rho / rho
    pushed[1] *= min_rho / rho
    return pushed


def clear_of_grip_ellipsoid(
    centre: np.ndarray,
    along: np.ndarray,
    palmar: np.ndarray,
    radii: tuple[float, float, float],
    grip_radius_m: float,
    clearance_m: float,
    *,
    margin_m: float = 0.0,
) -> np.ndarray:
    """Keep a rotated ellipsoid clear of the grip cylinder using directional support."""
    centre_a = np.asarray(centre, dtype=np.float64)
    along_a = np.asarray(along, dtype=np.float64)
    along_norm = float(np.linalg.norm(along_a))
    if along_norm < 1e-8:
        return push_outside(
            centre_a, grip_radius_m + clearance_m + max(radii) + margin_m
        )
    axis_u = along_a / along_norm
    pal = orthogonal_dir(palmar, axis_u)
    side = np.asarray(np.cross(axis_u, pal), dtype=np.float64)
    rho = float(np.hypot(centre_a[0], centre_a[1]))
    if rho < 1e-8:
        u_rad = -pal
        u_rad_xy = np.array([u_rad[0], u_rad[1], 0.0], dtype=np.float64)
        u_norm = float(np.hypot(u_rad_xy[0], u_rad_xy[1]))
        u_rad = (
            u_rad_xy / u_norm
            if u_norm >= 1e-8
            else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        )
    else:
        u_rad = np.array([centre_a[0] / rho, centre_a[1] / rho, 0.0], dtype=np.float64)
    r_support = math.sqrt(
        (float(radii[0]) * float(np.dot(u_rad, side))) ** 2
        + (float(radii[1]) * float(np.dot(u_rad, pal))) ** 2
        + (float(radii[2]) * float(np.dot(u_rad, axis_u))) ** 2
    )
    return push_outside(centre_a, grip_radius_m + clearance_m + r_support + margin_m)


def scaled_radii(
    radii: tuple[float, float, float], scale: float
) -> tuple[float, float, float]:
    """Scale an ellipsoid radii tuple by ``scale``."""
    return (radii[0] * scale, radii[1] * scale, radii[2] * scale)


def cone_primitive(
    name: str,
    group: str,
    kind: str,
    start: np.ndarray,
    end: np.ndarray,
    radii: tuple[float, float],
    palmar: np.ndarray,
    *,
    blend: str = "joint",
) -> Primitive:
    """Create a tapered round-cone union primitive; ``radii`` is (start, end)."""
    return Primitive(
        name=name,
        group=group,
        kind=kind,
        operation="union",
        shape="cone",
        start=vec3(start),
        end=vec3(end),
        radius0=float(radii[0]),
        radius1=float(radii[1]),
        palmar=vec3(palmar),
        aspect=1.0,
        blend=blend,
        radii=(0.0, 0.0, 0.0),
    )


def sphere_primitive(
    name: str,
    group: str,
    kind: str,
    centre: np.ndarray,
    radius: float,
    palmar: np.ndarray,
    *,
    blend: str,
) -> Primitive:
    """Create a sphere SDF primitive."""
    return Primitive(
        name=name,
        group=group,
        kind=kind,
        operation="union",
        shape="sphere",
        start=vec3(centre),
        end=vec3(centre),
        radius0=float(radius),
        radius1=float(radius),
        palmar=vec3(palmar),
        aspect=1.0,
        blend=blend,
        radii=(0.0, 0.0, 0.0),
    )


def ellipsoid_primitive(
    name: str,
    group: str,
    kind: str,
    centre: np.ndarray,
    along: np.ndarray,
    palmar: np.ndarray,
    radii: tuple[float, float, float],
    *,
    blend: str,
) -> Primitive:
    """Create an oriented ellipsoid SDF primitive."""
    return Primitive(
        name=name,
        group=group,
        kind=kind,
        operation="union",
        shape="ellipsoid",
        start=vec3(centre),
        end=vec3(along),
        radius0=0.0,
        radius1=0.0,
        palmar=vec3(palmar),
        aspect=1.0,
        blend=blend,
        radii=radii,
    )


def tangent_chain(
    lengths: tuple[float, ...],
    tangent_radius: float,
    mcp_azimuth: float,
    z: float,
    curl: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build bone axes tangent to ``tangent_radius`` around the grip axis."""
    before = lengths[0] / 2.0
    theta = mcp_azimuth + curl * math.atan2(before, tangent_radius)
    lift = np.array([0.0, 0.0, z], dtype=np.float64)
    bones: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for length in lengths:
        after = length - before
        if after <= 0.0:
            raise ValueError(f"phalanx length {length} cannot continue the wrap")
        outward = radial_dir(theta)
        tangent = curl * tangential_dir(theta)
        start = tangent_radius * outward - before * tangent + lift
        end = tangent_radius * outward + after * tangent + lift
        bones.append((start, end, outward))
        theta += curl * 2.0 * math.atan2(after, tangent_radius)
        before = after
    return bones


def closest_axis_point(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float]:
    """Point on segment closest to the grip axis and its normalized fraction."""
    delta = end - start
    denom = float(delta[0] ** 2 + delta[1] ** 2)
    if denom < 1e-12:
        return 0.5 * (start + end), 0.5
    fraction = -(start[0] * delta[0] + start[1] * delta[1]) / denom
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return start + fraction * delta, fraction


def add_pad(
    primitives: list[Primitive],
    name: str,
    group: str,
    start: np.ndarray,
    end: np.ndarray,
    radius0: float,
    radius1: float,
    contact_radius: float,
) -> None:
    """Add palmar sphere bringing tapered phalanx contact out to ``contact_radius``."""
    point, fraction = closest_axis_point(start, end)
    radius_here = (1.0 - fraction) * radius0 + fraction * radius1
    if contact_radius - radius_here <= 0.0004:
        return
    radial = np.array([point[0], point[1], 0.0], dtype=np.float64)
    rho = float(np.linalg.norm(radial))
    if rho < 1e-8:
        return
    palmar = -radial / rho
    pad_radius = max(radius_here * 0.85, 0.004)
    shift = contact_radius - pad_radius
    if shift <= 0.0:
        return
    primitives.append(
        sphere_primitive(
            f"{name}_pad",
            group,
            "pad",
            point + palmar * shift,
            pad_radius,
            palmar,
            blend="joint",
        )
    )


def add_knuckle(
    primitives: list[Primitive],
    name: str,
    group: str,
    centre: np.ndarray,
    palmar: np.ndarray,
    bone_radius: float,
    settings: SdfHandSettings,
    scale: float,
) -> None:
    """Add a dorsal knuckle bump sphere at a joint centre."""
    dorsal = -palmar
    bump = bone_radius * settings.knuckle_radius_fraction
    offset = bone_radius * settings.knuckle_dorsal_fraction
    primitives.append(
        sphere_primitive(
            name,
            group,
            "knuckle",
            centre + dorsal * offset,
            max(bump, 0.002 * scale),
            palmar,
            blend="joint",
        )
    )


def add_crease(
    primitives: list[Primitive],
    name: str,
    centre: np.ndarray,
    axis: np.ndarray,
    palmar: np.ndarray,
    bone_radius: float,
    settings: SdfHandSettings,
) -> None:
    """Add a subtractive palmar flexion crease cutter."""
    side = unit(np.asarray(np.cross(axis, palmar), dtype=np.float64))
    half = side * bone_radius * settings.crease_span_fraction
    depth = palmar * bone_radius * settings.crease_depth_fraction
    radius = max(bone_radius * settings.crease_radius_fraction, 0.001)
    cutter = cone_primitive(
        name,
        "crease",
        "crease",
        centre + depth - half,
        centre + depth + half,
        (radius, radius),
        palmar,
        blend="crease",
    )
    primitives.append(replace(cutter, operation="subtract"))


@dataclass(frozen=True)
class PoseFrame:
    """Scaled per-hand quantities shared by the finger, palm, and thumb posers."""

    settings: SdfHandSettings
    scale: float
    grip_radius_m: float
    clearance: float
    lift: float
    curl: int
    margin_m: float

    def clear(
        self,
        centre: np.ndarray,
        along: np.ndarray,
        palmar: np.ndarray,
        radii: tuple[float, float, float],
    ) -> np.ndarray:
        """Push an ellipsoid centre clear of the grip with this hand's margins."""
        return clear_of_grip_ellipsoid(
            centre,
            along,
            palmar,
            radii,
            self.grip_radius_m,
            self.clearance,
            margin_m=self.margin_m,
        )


Bone = tuple[np.ndarray, np.ndarray, np.ndarray]


def add_finger_joints(
    primitives: list[Primitive],
    frame: PoseFrame,
    finger: str,
    bones: list[Bone],
    radii: tuple[tuple[float, float], ...],
) -> None:
    """Add knuckle bumps at MCP/PIP/DIP and flexion creases at PIP/DIP."""
    joints = (
        ("mcp", bones[0][0], radii[0][0], bones[0][1] - bones[0][0], bones[0][2]),
        ("pip", bones[0][1], radii[0][1], bones[1][1] - bones[1][0], bones[1][2]),
        ("dip", bones[1][1], radii[1][1], bones[2][1] - bones[2][0], bones[2][2]),
    )
    for label, centre, radius, axis, outward in joints:
        palmar = -outward
        add_knuckle(
            primitives,
            f"{finger}_{label}",
            finger,
            centre,
            palmar,
            radius,
            frame.settings,
            frame.scale,
        )
        if label != "mcp":
            add_crease(
                primitives,
                f"{finger}_{label}_crease",
                centre,
                axis,
                palmar,
                radius,
                frame.settings,
            )


def add_nail(
    primitives: list[Primitive],
    frame: PoseFrame,
    finger: str,
    distal: Bone,
    distal_length: float,
    distal_tip_radius: float,
) -> None:
    """Add the fingernail ellipsoid on the dorsal side of the distal phalanx."""
    settings = frame.settings
    distal_start, distal_end, distal_out = distal
    distal_axis = unit(distal_end - distal_start)
    nail_centre = (
        distal_end
        - distal_axis * (distal_length * settings.nail_along_fraction)
        + distal_out * (distal_tip_radius * 0.45)
    )
    primitives.append(
        ellipsoid_primitive(
            f"{finger}_nail",
            finger,
            "nail",
            nail_centre,
            distal_axis,
            -distal_out,
            scaled_radii(settings.nail_radii_m, frame.scale),
            blend="nail",
        )
    )


def blend_thumb_bones(
    frame: PoseFrame,
    *,
    thumb_azimuth: float,
    outward: np.ndarray,
    tangent_radius: float,
    straightness: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Thumb phalanx axes blended from the straight to the flexed pose."""
    thumb = frame.settings.thumb
    scale = frame.scale
    centre_xy = tangent_radius * outward
    stations = tuple(station * scale for station in thumb.stations_m)
    tan_dir = float(frame.curl) * np.cross(
        np.array([0.0, 0.0, 1.0], dtype=np.float64), outward
    )
    sweep = frame.settings.thumb_opposition_sweep_m * scale
    tan_offsets = (-0.35 * sweep, 0.15 * sweep, 0.65 * sweep)
    straight = [
        (
            centre_xy
            + np.array([0.0, 0.0, stations[0]], dtype=np.float64)
            + tan_dir * tan_offsets[0],
            centre_xy
            + np.array([0.0, 0.0, stations[1]], dtype=np.float64)
            + tan_dir * tan_offsets[1],
        ),
        (
            centre_xy
            + np.array([0.0, 0.0, stations[1]], dtype=np.float64)
            + tan_dir * tan_offsets[1],
            centre_xy
            + np.array([0.0, 0.0, stations[2]], dtype=np.float64)
            + tan_dir * tan_offsets[2],
        ),
    ]
    flexed = tangent_chain(
        tuple(length * scale for length in thumb.phalanges_m),
        tangent_radius,
        thumb_azimuth,
        0.0,
        frame.curl,
    )
    thumb_bones: list[tuple[np.ndarray, np.ndarray]] = []
    for index, (start_s, end_s) in enumerate(straight):
        start = straightness * start_s + (1.0 - straightness) * flexed[index][0]
        end = straightness * end_s + (1.0 - straightness) * flexed[index][1]
        thumb_bones.append((start, end))
    return thumb_bones


def measure_skeleton(primitives: tuple[Primitive, ...]) -> tuple[float, float]:
    """Return measured middle-finger chord length and palm width in metres."""
    phalanx = {
        prim.name: prim
        for prim in primitives
        if prim.group == "middle" and prim.kind == "phalanx"
    }
    mc_base = next(prim for prim in primitives if prim.name == "middle_metacarpal_base")
    mc_head = next(prim for prim in primitives if prim.name == "middle_metacarpal_head")
    chord = float(np.linalg.norm(np.asarray(mc_head.end) - np.asarray(mc_base.start)))
    length = chord
    for name in ("middle_proximal", "middle_middle", "middle_distal"):
        prim = phalanx[name]
        length += float(np.linalg.norm(np.asarray(prim.end) - np.asarray(prim.start)))
    index = next(prim for prim in primitives if prim.name == "index_proximal")
    little = next(prim for prim in primitives if prim.name == "little_proximal")
    width = abs(index.start[2] - little.start[2]) + index.radius0 + little.radius0
    return length, width


def mirror_y(
    primitives: tuple[Primitive, ...], dorsal: np.ndarray
) -> tuple[tuple[Primitive, ...], np.ndarray]:
    """Mirror all SDF primitives and dorsal hint across the y = 0 plane."""

    def flip(point: tuple[float, float, float]) -> tuple[float, float, float]:
        return (point[0], -point[1], point[2])

    mirrored = tuple(
        replace(
            prim,
            start=flip(prim.start),
            end=flip(prim.end),
            palmar=flip(prim.palmar),
        )
        for prim in primitives
    )
    dorsal_m = np.array([dorsal[0], -dorsal[1], dorsal[2]], dtype=np.float64)
    return mirrored, dorsal_m


__all__ = [
    "Bone",
    "PoseFrame",
    "add_crease",
    "add_finger_joints",
    "add_knuckle",
    "add_nail",
    "add_pad",
    "blend_thumb_bones",
    "clear_of_grip_ellipsoid",
    "closest_axis_point",
    "cone_primitive",
    "ellipsoid_primitive",
    "measure_skeleton",
    "mirror_y",
    "orthogonal_dir",
    "push_outside",
    "radial_dir",
    "scaled_radii",
    "sphere_primitive",
    "tangent_chain",
    "tangential_dir",
    "unit",
    "vec3",
]
