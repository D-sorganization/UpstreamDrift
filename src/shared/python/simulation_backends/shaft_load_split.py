"""Internal loads along the flexible-shaft chain, by part (#15150 Track Z, Z9).

The flexible shaft is a chain of rigid links (``club_shaft_chain``): link
``i`` runs from shaft joint ``i`` toward the head, and the last link carries
the head. At each joint the proximal side passes one wrench to everything
distal of it; for a serial chain that is the sum of the distal links' demands
(momentum rate less weight). :func:`section_loads` sums them and resolves the
wrench at the joint into axial force, shear, torsion and bending moment.

:func:`shaft_load_split` evaluates the per-link demand at the ZTCF and ZVCF
states of the existing split solve (:func:`hand_force_split.part_states`), so
the shaft loads use the same accelerations as the hand split and the parts
add to the total (ZTCF: drift with gravity, zero applied torque; ZVCF: the
applied torques alone; owner decision #14497).

Frame convention (world frame throughout):

- ``e_i`` is the unit shaft axis at joint ``i``, pointing tip-ward (grip
  toward head), taken on the distal link.
- ``F_i`` and ``M_i`` are the force and the moment about joint ``i``'s point
  that the proximal side applies to the distal chain.
- ``axial_n = -F_i . e_i``: tension positive (the proximal side pulls the
  distal chain back toward the grip).
- ``shear_n = F_i - (F_i . e_i) e_i``, ``torsion_nm = M_i . e_i`` (right hand
  about the tip-ward axis) and ``bending_nm = M_i - (M_i . e_i) e_i``.

Scope: pointwise at each state, as the hand split. The station at the grip
(the hands' load on the club) is the Z2 hand split, not repeated here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from src.shared.python.simulation_backends.hand_force_split import (
    PARTS,
    checked_rows,
    part_states,
)
from src.shared.python.simulation_backends.protocol import (
    ConstrainedEquationsOfMotionProvider,
)

Array = NDArray[np.float64]
#: ``stations(q) -> (points (n, 3), tip-ward axes (n, 3))`` of the shaft
#: joints, world frame, grip to head.
StationGeometry = Callable[[Array], tuple[Array, Array]]
#: ``link_load(q, v, a, gravity) -> (forces (n, 3), moments about the world
#: origin (n, 3))``: each link's momentum rate less its weight when
#: ``gravity``, world frame, in station order.
LinkLoad = Callable[[Array, Array, Array, bool], tuple[Array, Array]]

SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config/shaft_load_split.json"

__all__ = [
    "SETTINGS_PATH",
    "ShaftLoadSettings",
    "ShaftLoadSplit",
    "ShaftSectionLoads",
    "load_settings",
    "section_loads",
    "shaft_load_split",
]


@dataclass(frozen=True)
class ShaftLoadSettings:
    """Tunables of the shaft-load split (``src/config/shaft_load_split.json``)."""

    axis_min_norm: float
    static_check_rtol: float


def load_settings(path: Path = SETTINGS_PATH) -> ShaftLoadSettings:
    """Read :class:`ShaftLoadSettings`; ``ValueError`` on a non-positive value."""
    record = json.loads(path.read_text(encoding="utf-8"))
    settings = ShaftLoadSettings(
        axis_min_norm=float(record["axis_min_norm"]),
        static_check_rtol=float(record["static_check_rtol"]),
    )
    if settings.axis_min_norm <= 0.0 or settings.static_check_rtol <= 0.0:
        raise ValueError("shaft-load settings must be positive")
    return settings


@dataclass(frozen=True)
class ShaftSectionLoads:
    """Internal loads at each shaft joint over the rows (frame: module doc).

    ``axial_n`` and ``torsion_nm`` are ``(rows, n)``; ``shear_n`` and
    ``bending_nm`` are ``(rows, n, 3)`` world vectors normal to the axis.
    """

    axial_n: Array
    shear_n: Array
    torsion_nm: Array
    bending_nm: Array

    def __add__(self, other: ShaftSectionLoads) -> ShaftSectionLoads:
        """Element-wise sum (parts add); ``ValueError`` on a shape mismatch."""
        pairs = tuple(zip(self._fields(), other._fields(), strict=True))
        if any(a.shape != b.shape for a, b in pairs):
            raise ValueError("shaft loads of different shape cannot be added")
        return ShaftSectionLoads(*(a + b for a, b in pairs))

    def _fields(self) -> tuple[Array, Array, Array, Array]:
        return self.axial_n, self.shear_n, self.torsion_nm, self.bending_nm

    @property
    def shear_magnitude_n(self) -> Array:
        """``|shear|`` at each joint, ``(rows, n)``."""
        return np.linalg.norm(self.shear_n, axis=-1)

    @property
    def bending_magnitude_nm(self) -> Array:
        """``|bending moment|`` at each joint, ``(rows, n)``."""
        return np.linalg.norm(self.bending_nm, axis=-1)


def _station_arrays(*arrays: Array) -> tuple[Array, ...]:
    out = tuple(np.asarray(x, dtype=float) for x in arrays)
    shape = out[0].shape
    if len(shape) != 2 or shape[0] < 1 or shape[1] != 3:
        raise ValueError(f"station arrays must have shape (n, 3), got {shape}")
    if any(x.shape != shape for x in out):
        raise ValueError("points, axes, forces and moments must share one shape")
    return out


def section_loads(
    points: Array,
    axes: Array,
    link_forces: Array,
    link_moments: Array,
    settings: ShaftLoadSettings | None = None,
) -> ShaftSectionLoads:
    """Axial, shear, torsion and bending at each shaft joint, one state.

    ``link_forces``/``link_moments`` are each link's demand (world, moment
    about the origin), link ``i`` distal of joint ``i``. Preconditions: all
    four are ``(n, 3)`` and every axis is longer than ``axis_min_norm``
    (``ValueError``). Postcondition: rows of length 1, frame as the module doc.
    """
    points, axes, forces, moments = _station_arrays(
        points, axes, link_forces, link_moments
    )
    limit = (settings or load_settings()).axis_min_norm
    lengths = np.linalg.norm(axes, axis=1)
    if not np.all(lengths > limit):
        raise ValueError("every shaft axis must have a non-zero length")
    unit = axes / lengths[:, None]
    force = np.cumsum(forces[::-1], axis=0)[::-1]
    moment = np.cumsum(moments[::-1], axis=0)[::-1] - np.cross(points, force)
    along_f = np.einsum("ij,ij->i", force, unit)
    along_m = np.einsum("ij,ij->i", moment, unit)
    return ShaftSectionLoads(
        axial_n=-along_f[None],
        shear_n=(force - along_f[:, None] * unit)[None],
        torsion_nm=along_m[None],
        bending_nm=(moment - along_m[:, None] * unit)[None],
    )


@dataclass(frozen=True)
class ShaftLoadSplit:
    """Internal shaft loads by part (:data:`PARTS`)."""

    parts: Mapping[str, ShaftSectionLoads]

    def part(self, part: str) -> ShaftSectionLoads:
        """``part`` in :data:`PARTS`; else ``ValueError``."""
        if part not in PARTS:
            raise ValueError(f"part must be one of {PARTS}, got {part!r}")
        return self.parts[part]

    def total(self) -> ShaftSectionLoads:
        """ZTCF plus ZVCF: the internal loads under the applied torques."""
        drift, control = (self.part(p) for p in PARTS)
        return drift + control


def shaft_load_split(
    provider: ConstrainedEquationsOfMotionProvider,
    stations: StationGeometry,
    link_load: LinkLoad,
    q: Array,
    v: Array,
    tau: Array,
) -> ShaftLoadSplit:
    """Internal shaft loads, ZTCF and ZVCF, at every row of ``q``, ``v``, ``tau``.

    Preconditions: ``q``, ``v`` and ``tau`` share a row count; ``stations``
    and ``link_load`` return ``(n, 3)`` arrays of one ``n`` (``ValueError``).
    Postcondition: ``part("ztcf") + part("zvcf") == total()``, and the total
    equals :func:`section_loads` of ``link_load`` at the solve under ``tau``.
    """
    q, v, tau = checked_rows(q, v, tau)
    settings = load_settings()
    rows: dict[str, list[ShaftSectionLoads]] = {p: [] for p in PARTS}
    for k in range(q.shape[0]):
        points, axes = stations(q[k])
        for part, rates, a, _, gravity in part_states(provider, q[k], v[k], tau[k]):
            forces, moments = link_load(q[k], rates, a, gravity)
            rows[part].append(section_loads(points, axes, forces, moments, settings))
    return ShaftLoadSplit(MappingProxyType({p: _stacked(rows[p]) for p in PARTS}))


def _stacked(rows: list[ShaftSectionLoads]) -> ShaftSectionLoads:
    """One :class:`ShaftSectionLoads` from single-row ones, in row order."""
    fields = (r._fields() for r in rows)
    return ShaftSectionLoads(*(np.concatenate(f) for f in zip(*fields, strict=True)))
