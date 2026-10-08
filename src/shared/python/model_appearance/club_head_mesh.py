"""Realistic clubhead meshes shared by every engine's visual layer (GCV-11).

One adapter, many consumers (MuJoCo, OpenSim, MeshCat, MyoSuite, native
export). Geometry order of preference:

1. the Tools parametric builder (``rate_of_closure.club``) through an
   optional-import gateway, so a Tools bump improves every engine at once;
2. committed generated STLs under ``assets/club_heads/`` (with a provenance
   manifest) when Tools is not importable. The two sources are held equal by
   a unit test.

Frames. The *head frame* is the Tools frame: x toward the target (face normal
at zero loft), y up, z toward the toe (right-handed golfer, toe away). The
*club frame* is the spec's club-body frame (``motion_matching.club_models``):
origin at the sole point on the shaft axis, shaft toward the grip along -y,
the shaft axis offset ``axis_offset_m`` along +z. The face normal is club
``+x`` at zero loft: in the matched address poses the head-up axis is vertical
and ``+x`` points down the target line (checked against the driver and 7-iron
reference bundles, where the residual roll about the shaft is under 6 degrees).

Tools heads carry no hosel, so a short tapered hosel tube on the lie-angle
shaft axis is added here. The mesh is visual only: head mass, inertia and the
rigid-body centre of mass stay as specified (physics identity).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.club_shaft_mesh import tapered_tube
from src.shared.python.model_appearance.geometry import Mesh

Array = NDArray[np.float64]

REPO_ROOT = Path(__file__).resolve().parents[4]
ASSET_DIR = REPO_ROOT / "assets" / "club_heads"
MANIFEST_PATH = ASSET_DIR / "provenance.json"
TOOLS_LIBRARY_MODULE = "rate_of_closure.club.library"
TOOLS_HEAD_MODULE = "rate_of_closure.club.parametric_head"
DEFAULT_AXIS_OFFSET_M = 0.064  # club-body shaft axis offset (club_models)
HOSEL_RADIUS_M = 0.0072
HOSEL_ABOVE_CROWN_M = {"Driver": 0.012, "Wood": 0.014, "Hybrid": 0.022}
HOSEL_ABOVE_CROWN_DEFAULT_M = 0.030  # irons and wedges
_HEEL_INSET_FRACTION = 0.12

_IRONS = {3: "3-Iron", 5: "5-Iron", 7: "7-Iron", 9: "9-Iron"}
_WEDGES = {
    46: "Pitching Wedge",
    52: "Gap Wedge",
    56: "Sand Wedge",
    60: "Lob Wedge",
}
_WOODS = {3: "3-Wood", 5: "5-Wood"}
_NAME_RE = re.compile(
    r"^(?P<kind>driver|fairway|wood|hybrid|iron|wedge)"
    r"[\s_\-(]*(?P<num>\d{1,2})?[)]?$"
)
_IRON_RE = re.compile(r"^(?P<num>\d{1,2})[\s_\-]*iron$")


class ClubHeadUnavailableError(FileNotFoundError):
    """Raised when neither the Tools builder nor a committed STL has the head."""


@dataclass(frozen=True)
class ClubHeadMesh:
    """A head in the head frame (metres) plus the spec data that placed it."""

    library_name: str
    source: str  # "tools_parametric" | "committed_stl"
    loft_deg: float
    lie_deg: float
    head_mesh: Mesh  # body only, closed
    mesh: Mesh  # body + hosel, two closed shells
    sole_point: Array  # head frame, where the shaft axis meets the sole plane
    shaft_direction: Array  # head frame, unit vector sole -> grip
    hosel_top_m: float  # distance from the sole point to the hosel top on the axis


def library_name_for(name: str) -> str:
    """Tools library name for a club alias such as ``iron7`` or ``wedge56``.

    Accepts ``driver``, ``fairway3``/``wood5``, ``hybrid3``, ``iron7``,
    ``7-iron``, ``iron(9)`` and ``wedge56``. Raises ``TypeError`` for a
    non-string and ``ValueError`` for an unsupported club or number.
    """
    if not isinstance(name, str):
        raise TypeError("club name must be a string")
    key = name.strip().lower()
    if not key:
        raise ValueError("club name must be non-empty")
    match = _IRON_RE.match(key)
    if match is not None:
        kind, num = "iron", int(match.group("num"))
    else:
        match = _NAME_RE.match(key)
        if match is None:
            raise ValueError(f"unsupported club name {name!r}")
        kind = match.group("kind")
        num = None if match.group("num") is None else int(match.group("num"))
    if kind == "driver":
        return "Driver 10.5°"
    if kind == "hybrid":
        return "3-Hybrid"
    if kind in ("fairway", "wood"):
        return _lookup(_WOODS, 3 if num is None else num, "fairway wood", name)
    if kind == "iron":
        return _lookup(_IRONS, 7 if num is None else num, "iron", name)
    return _lookup(_WEDGES, 56 if num is None else num, "wedge loft", name)


def _lookup(table: dict[int, str], value: int, label: str, raw: str) -> str:
    if value in table:
        return table[value]
    if label == "iron" and 3 <= value <= 9:  # even irons use the nearest library head
        return table[min(table, key=lambda n: (abs(n - value), n))]
    if label == "wedge loft":
        near = min(table, key=lambda n: abs(n - value))
        if abs(near - value) <= 2:
            return table[near]
    raise ValueError(f"unsupported {label} {value} in club name {raw!r}")


# --------------------------------------------------------------------- sources


def _import_tools() -> tuple[Any, Any] | None:
    try:
        return import_module(TOOLS_LIBRARY_MODULE), import_module(TOOLS_HEAD_MODULE)
    except ImportError:
        return None


def tools_builder_available() -> bool:
    """True when the Tools parametric head builder can be imported."""
    return _import_tools() is not None


def triangles_from_tools(library_name: str) -> Array | None:
    """``(n, 3, 3)`` head triangles in metres from Tools, or ``None``."""
    modules = _import_tools()
    if modules is None:
        return None
    library, parametric = modules
    triangles = np.asarray(
        parametric.build_parametric_head(library.get_club(library_name)), float
    )
    return triangles


def _manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise ClubHeadUnavailableError(f"missing provenance manifest {MANIFEST_PATH}")
    data: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return data


def _read_binary_stl_mm(path: Path) -> Array:
    raw = path.read_bytes()
    count = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    record = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    if len(raw) != 84 + count * record.itemsize:
        raise ValueError(f"{path} is not a binary STL of {count} triangles")
    return np.asarray(
        np.frombuffer(raw, dtype=record, count=count, offset=84)["v"], float
    )


def triangles_from_committed_stl(library_name: str) -> Array:
    """``(n, 3, 3)`` head triangles in metres from the committed STL."""
    entry = _manifest()["heads"].get(library_name)
    if entry is None:
        raise ClubHeadUnavailableError(f"no committed head STL for {library_name!r}")
    path = REPO_ROOT / entry["path"]
    if not path.is_file():
        raise ClubHeadUnavailableError(f"committed head STL missing: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"{path} does not match its provenance sha256")
    return _read_binary_stl_mm(path) * 1e-3


def _spec_angles(library_name: str) -> tuple[float, float]:
    entry = _manifest()["heads"].get(library_name)
    if entry is not None:
        return float(entry["loft_deg"]), float(entry["lie_deg"])
    modules = _import_tools()
    if modules is None:
        raise ClubHeadUnavailableError(f"no spec for {library_name!r}")
    spec = modules[0].get_club(library_name)
    return float(spec.loft_deg), float(spec.lie_deg)


# -------------------------------------------------------------------- geometry


def _weld(triangles: Array) -> Mesh:
    flat = triangles.reshape(-1, 3)
    keys = np.round(flat * 1e8).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    faces = np.asarray(inverse, dtype=np.int64).reshape(-1, 3)
    mesh = Mesh(flat[first], faces)
    if mesh.volume() < 0.0:
        mesh = Mesh(mesh.vertices, faces[:, ::-1].copy())
    return mesh


def _concat(*meshes: Mesh) -> Mesh:
    verts, faces, base = [], [], 0
    for mesh in meshes:
        verts.append(mesh.vertices)
        faces.append(mesh.faces + base)
        base += len(mesh.vertices)
    return Mesh(np.vstack(verts), np.vstack(faces))


def shaft_direction_head(lie_deg: float) -> Array:
    """Unit sole-to-grip direction in the head frame (leans toward the heel)."""
    tau = math.radians(90.0 - lie_deg)
    return np.array([0.0, math.cos(tau), -math.sin(tau)])


def _sole_point(vertices: Array) -> Array:
    centre_band = np.abs(vertices[:, 2]) < 0.003
    mid = vertices[:, 1] - vertices[:, 1].min()
    face_x = float(
        vertices[centre_band & (np.abs(mid - 0.5 * mid.max()) < 0.003), 0].max()
    )
    zmin, zmax = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    z0 = zmin + _HEEL_INSET_FRACTION * (zmax - zmin)
    return np.array([face_x, float(vertices[:, 1].min()), z0])


def _hosel_top(sole: Array, direction: Array, vertices: Array, club_type: str) -> float:
    crown = float(vertices[:, 1].max()) - sole[1]
    above = HOSEL_ABOVE_CROWN_M.get(club_type, HOSEL_ABOVE_CROWN_DEFAULT_M)
    return (crown + above) / float(direction[1])


def _hosel(sole: Array, direction: Array, t_top: float) -> Mesh:
    return tapered_tube(
        sole + 0.35 * t_top * direction,
        sole + t_top * direction,
        HOSEL_RADIUS_M,
        0.9 * HOSEL_RADIUS_M,
    )


def _club_type(library_name: str) -> str:
    if library_name.startswith("Driver"):
        return "Driver"
    if "Wood" in library_name:
        return "Wood"
    return "Hybrid" if "Hybrid" in library_name else "Iron"


@lru_cache(maxsize=32)
def load_club_head(name: str) -> ClubHeadMesh:
    """The head for a club alias or library name, Tools first, STL fallback.

    Postconditions: ``head_mesh`` and ``mesh`` are closed with outward
    winding; units are metres in the head frame.
    """
    library_name = _resolve(name)
    triangles = triangles_from_tools(library_name)
    source = "tools_parametric"
    if triangles is None:
        triangles = triangles_from_committed_stl(library_name)
        source = "committed_stl"
    head = _weld(triangles)
    loft, lie = _spec_angles(library_name)
    sole = _sole_point(head.vertices)
    direction = shaft_direction_head(lie)
    t_top = _hosel_top(sole, direction, head.vertices, _club_type(library_name))
    hosel = _hosel(sole, direction, t_top)
    return ClubHeadMesh(
        library_name,
        source,
        loft,
        lie,
        head,
        _concat(head, hosel),
        sole,
        direction,
        t_top,
    )


def _resolve(name: str) -> str:
    try:
        return library_name_for(name)
    except ValueError:
        if isinstance(name, str) and name in _manifest()["heads"]:
            return name
        raise


def measured_face_normal(mesh: Mesh) -> Array:
    """Area-weighted mean normal of the face patch around its centre.

    Works on a head-frame mesh; the patch is the triangles within 15 mm of the
    face centre whose normals point toward the target.
    """
    tri = mesh.vertices[mesh.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(2.0 * area[:, None], 1e-18)
    centroid = tri.mean(axis=1)
    x_mid = 0.5 * (mesh.vertices[:, 0].min() + mesh.vertices[:, 0].max())
    forward = (normal[:, 0] > 0.3) & (centroid[:, 0] > x_mid)
    mid_y = 0.5 * (mesh.vertices[:, 1].min() + mesh.vertices[:, 1].max())
    near = np.hypot(centroid[:, 1] - mid_y, centroid[:, 2]) < 0.015
    pick = forward & near
    if not pick.any():
        raise ValueError("mesh has no face patch toward +x")
    mean = (normal[pick] * area[pick, None]).sum(axis=0)
    return np.asarray(mean / np.linalg.norm(mean), float)


def head_to_club_rotation(head: ClubHeadMesh) -> Array:
    """Rotation taking head-frame vectors to the club-body frame.

    Maps (x_h, shaft direction, x_h x shaft) to (+x, -y, -z): the face looks
    along club ``+x`` and the shaft runs toward the grip along ``-y``.
    """
    x_h = np.array([1.0, 0.0, 0.0])
    s_h = head.shaft_direction
    basis_h = np.column_stack([x_h, s_h, np.cross(x_h, s_h)])
    basis_c = np.diag([1.0, -1.0, -1.0])
    return np.asarray(basis_c @ basis_h.T, float)


def club_frame_mesh(
    head: ClubHeadMesh, *, axis_offset_m: float = DEFAULT_AXIS_OFFSET_M
) -> Mesh:
    """Head plus hosel in the club-body frame (metres, outward winding)."""
    if not math.isfinite(axis_offset_m):
        raise ValueError("axis_offset_m must be finite")
    rot = head_to_club_rotation(head)
    shift = np.array([0.0, 0.0, axis_offset_m]) - rot @ head.sole_point
    return Mesh(head.mesh.vertices @ rot.T + shift, head.mesh.faces)
