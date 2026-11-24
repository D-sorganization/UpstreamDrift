"""Tension and compression by part: ZTCF and ZVCF section loads (#15150 Z4b).

Segment shading shows the axial load in each body segment (tension positive).
This module splits it the way the hand arrows are split:

* ZTCF (drift) is the inverse dynamics of the drift solve: ``tau = 0`` with
  ``q`` and ``v`` kept, so gravity, momentum and passive springs, with the
  hand weld and the planted feet re-solved;
* ZVCF (control) is the inverse dynamics of the control solve: the applied
  torques alone at ``(q, 0)``. Section loads are linear in the acceleration,
  the external wrenches and gravity, so the gravity term is removed by
  subtracting the still, unloaded pose; gravity is counted once, in the ZTCF.

The two parts add to ``total``, the inverse dynamics of the summed solve.
``recorded`` keeps the export's own shading (recorded motion, contact-law
ground), which this split does not replace unless asked.

The ground reaction splits the same way (Z4c): each part's planted-foot
reactions replace the contact arrows, drawn at each foot's centre of
pressure, and each case keeps only the force kinds that belong to it (no
actuator torques in the ZTCF, no weight in the ZVCF).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.mujoco_axial_loads import BodyWrench
from src.shared.python.force_overlay.contracts import OverlayWrench, WrenchKind

__all__ = [
    "FORCE_NOTES",
    "PART_KINDS",
    "SHADING_NOTES",
    "SHADING_PARTS",
    "PartLoad",
    "PartSplit",
    "centre_of_pressure",
    "part_ground",
    "swing_part_splits",
    "with_part_wrenches",
    "part_axial_frames",
    "part_axial_source",
    "swing_part_frames",
    "with_part_shading",
]

#: ``recorded`` is the export's own shading; the others come from the split.
SHADING_PARTS = ("recorded", "total", "ztcf", "zvcf")
SHADING_NOTES = {
    "recorded": "",
    "total": "Shading: ZTCF + ZVCF section loads (feet planted, hands re-solved)",
    "ztcf": "Shading: ZTCF drift only (gravity, momentum, passive; tau = 0)",
    "zvcf": "Shading: ZVCF control only (applied torques alone, no gravity)",
}
_SPLIT = ("total", "ztcf", "zvcf")
#: force kinds each case keeps besides its own ground reaction: the ZTCF has
#: ``tau = 0`` (no actuator torques), the ZVCF has no gravity (no weight)
PART_KINDS = {
    "ztcf": frozenset({WrenchKind.GRAVITY}),
    "zvcf": frozenset({WrenchKind.JOINT_ACTUATOR, WrenchKind.MUSCLE}),
    "total": frozenset(
        {WrenchKind.GRAVITY, WrenchKind.JOINT_ACTUATOR, WrenchKind.MUSCLE}
    ),
}
FORCE_NOTES = {
    "recorded": "",
    "total": "Ground and torques: ZTCF + ZVCF (feet planted, centre of pressure)",
    "ztcf": "Ground: ZTCF drift part (feet planted); no applied torques (tau = 0)",
    "zvcf": "Ground: ZVCF control part (feet planted); no weight (no gravity)",
}
#: below this vertical load a foot's centre of pressure is undefined
COP_MIN_VERTICAL_N = 1.0

Sampler = Callable[[Any, Any, Sequence[Any], Any], AxialLoadFrame | None]


@dataclass(frozen=True)
class PartLoad:
    """One part's acceleration and reactions: the ground wrenches on the
    planted bodies and the hand weld closure, in the sampler's types."""

    accel: np.ndarray
    ground: tuple[Any, ...]
    closure: Any


def _combine(
    a: AxialLoadFrame, b: AxialLoadFrame, sign: float, source: str
) -> AxialLoadFrame:
    values = {
        body: None
        if x is None or b.values_n.get(body) is None
        else float(x) + sign * float(b.values_n[body])  # type: ignore[arg-type]
        for body, x in a.values_n.items()
    }
    return AxialLoadFrame(a.time_s, values, source)


def part_axial_frames(
    sample: Sampler,
    rates: np.ndarray,
    drift: PartLoad,
    control: PartLoad,
    unloaded: Any,
) -> dict[str, AxialLoadFrame | None]:
    """``{"total", "ztcf", "zvcf"}`` section loads at one state.

    ``sample(rates, accel, ground, closure)`` is the inverse dynamics of one
    load case; ``unloaded`` is a zero closure. Postcondition: where every
    sample has a value, ``total = ztcf + zvcf``; a body without one is
    ``None`` in every part. An unsupported model gives ``None`` parts.
    """
    zero = np.zeros_like(np.asarray(rates, dtype=float))
    ztcf = sample(rates, drift.accel, drift.ground, drift.closure)
    raw = sample(zero, control.accel, control.ground, control.closure)
    still = sample(zero, zero, (), unloaded)
    if ztcf is None or raw is None or still is None:
        return dict.fromkeys(_SPLIT)
    ztcf = AxialLoadFrame(ztcf.time_s, dict(ztcf.values_n), "ZTCF drift section load")
    zvcf = _combine(raw, still, -1.0, "ZVCF control section load (no gravity)")
    total = _combine(ztcf, zvcf, 1.0, "ZTCF + ZVCF section load")
    return {"total": total, "ztcf": ztcf, "zvcf": zvcf}


def centre_of_pressure(
    point: Sequence[float],
    force: Sequence[float],
    moment: Sequence[float],
    min_vertical_n: float = COP_MIN_VERTICAL_N,
) -> tuple[float, float, float]:
    """Point in the horizontal plane of ``point`` where ``force`` acts with no
    tilting moment (``moment`` is about ``point``; world z up).

    With ``|F_z| < min_vertical_n`` there is no centre of pressure and
    ``point`` is returned.
    """
    p, f, m = (np.asarray(x, dtype=float) for x in (point, force, moment))
    if abs(f[2]) < min_vertical_n:
        return (float(p[0]), float(p[1]), float(p[2]))
    return (float(p[0] - m[1] / f[2]), float(p[1] + m[0] / f[2]), float(p[2]))


def _sum_wrench(a: BodyWrench, b: BodyWrench) -> BodyWrench:
    if a.body != b.body or not np.allclose(a.point_m, b.point_m, atol=1e-12):
        raise ValueError(f"ground parts of {a.body!r} are not at the same point")
    force = tuple(np.add(a.force_n or (0, 0, 0), b.force_n or (0, 0, 0)).tolist())
    torque = tuple(np.add(a.torque_nm or (0, 0, 0), b.torque_nm or (0, 0, 0)).tolist())
    return BodyWrench(a.body, a.point_m, force, torque)  # type: ignore[arg-type]


def part_ground(
    drift: Sequence[BodyWrench], control: Sequence[BodyWrench]
) -> dict[str, tuple[BodyWrench, ...]]:
    """``{"ztcf", "zvcf", "total"}`` ground wrenches; ``total`` is the sum.

    Raises:
        ValueError: the two parts are not on the same bodies and points.
    """
    if len(drift) != len(control):
        raise ValueError("ground parts must name the same planted bodies")
    total = tuple(_sum_wrench(d, c) for d, c in zip(drift, control, strict=True))
    return {"ztcf": tuple(drift), "zvcf": tuple(control), "total": total}


def _ground_arrow(wrench: BodyWrench, part: str) -> OverlayWrench:
    force = wrench.force_n or (0.0, 0.0, 0.0)
    point = centre_of_pressure(wrench.point_m, force, wrench.torque_nm or (0, 0, 0))
    return OverlayWrench(
        WrenchKind.CONTACT,
        "contact:" + re.sub(r"[^A-Za-z0-9_.:-]", "_", wrench.body),
        wrench.body,
        point,
        force_n=tuple(map(float, force)),  # type: ignore[arg-type]
        source=f"{part.upper()} ground reaction, feet planted (centre of pressure)",
    )


def with_part_wrenches(
    frame_at: Callable[[int], Any],
    ground_at: Callable[[int], Mapping[str, Sequence[BodyWrench]]],
    part: str,
) -> Callable[[int], Any]:
    """``frame_at`` showing ``part``'s ground reaction and force kinds only.

    The contact arrows are replaced by the part's planted-foot reactions;
    other kinds are kept when :data:`PART_KINDS` lists them for ``part``.
    ``recorded`` returns ``frame_at`` itself.

    Raises:
        ValueError: ``part`` is not in :data:`SHADING_PARTS`.
    """
    if part not in SHADING_PARTS:
        raise ValueError(f"force part must be one of {SHADING_PARTS}, got {part!r}")
    if part == "recorded":
        return frame_at
    kinds = PART_KINDS[part]

    def at(index: int) -> Any:
        frame = frame_at(index)
        kept = tuple(w for w in frame.wrenches if w.kind in kinds)
        ground = tuple(_ground_arrow(w, part) for w in ground_at(index)[part])
        return replace(frame, wrenches=(*kept, *ground))

    return at


def with_part_shading(
    frame_at: Callable[[int], Any],
    parts_at: Callable[[int], Mapping[str, AxialLoadFrame | None]],
    part: str,
) -> Callable[[int], Any]:
    """``frame_at`` with its axial loads replaced by ``part`` of the split.

    ``recorded`` returns ``frame_at`` itself. A frame without axial loads,
    or blanked around ball contact (every value ``None``), is kept.

    Raises:
        ValueError: ``part`` is not in :data:`SHADING_PARTS`.
    """
    if part not in SHADING_PARTS:
        raise ValueError(f"shading part must be one of {SHADING_PARTS}, got {part!r}")
    if part == "recorded":
        return frame_at

    def at(index: int) -> Any:
        frame = frame_at(index)
        axial = frame.axial_loads
        if axial is None or all(x is None for x in axial.values_n.values()):
            return frame
        return replace(frame, axial_loads=parts_at(index)[part])

    return at


@dataclass(frozen=True)
class PartSplit:
    """One state's split: section loads and ground wrenches by part."""

    axial: dict[str, AxialLoadFrame | None]
    ground: dict[str, tuple[BodyWrench, ...]]


class _PartAxialSource:
    """Split section loads of a specification, bundle coordinate order."""

    def __init__(self, spec_bytes: bytes, order: tuple[str, ...]) -> None:
        from src.engines.physics_engines.mujoco.python.overlay_source import (
            MujocoOverlaySource,
        )
        from src.tools.native_viewer_export.ztcf_feed import split_provider

        self._order = order
        self._source = MujocoOverlaySource(spec_bytes)
        self._split = split_provider(spec_bytes, order)

    def _named(self, x: Any) -> dict[str, float]:
        return dict(zip(self._order, map(float, x), strict=True))

    def _load(self, q: np.ndarray, accel: Any, lam: np.ndarray) -> PartLoad:
        from src.shared.python.simulation_backends.hand_force_split import (
            CLOSURE_ROWS,
        )

        rot = np.asarray(self._split.frame_at(q)[0], dtype=float)
        weld = np.concatenate([rot @ lam[:3], rot @ lam[3:CLOSURE_ROWS]])
        feet = np.asarray(lam[CLOSURE_ROWS:], dtype=float).reshape(-1, 6)
        ground = tuple(
            BodyWrench(body, tuple(point), tuple(w[:3]), tuple(w[3:]))
            for (body, point), w in zip(self._split.ground_points(q), feet, strict=True)
        )
        closure = self._source.closure_from_multiplier(self._named(q), weld)
        return PartLoad(np.asarray(accel, dtype=float), ground, closure)

    def _sampler(self, q: np.ndarray, time_s: float) -> Sampler:
        coords = self._named(q)

        def sample(rates: Any, accel: Any, ground: Any, closure: Any) -> Any:
            return self._source.axial_load_frame(
                coords,
                self._named(rates),
                self._named(accel),
                time_s,
                external_wrenches=ground,
                closure=closure,
            )

        return sample

    def _parts(self, q: Any, v: Any, tau: Any) -> tuple[PartLoad, PartLoad]:
        from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
            drift_and_control_states,
        )

        q = np.asarray(q, dtype=float)
        (a_d, lam_d), (a_c, lam_c) = drift_and_control_states(
            self._split.provider, q, v, tau
        )
        return self._load(q, a_d, lam_d), self._load(q, a_c, lam_c)

    def split(self, q: Any, v: Any, tau: Any, time_s: float) -> PartSplit:
        """Section loads and ground wrenches by part at one state."""
        drift, control = self._parts(q, v, tau)
        unloaded = self._source.closure_from_multiplier(self._named(q), np.zeros(6))
        axial = part_axial_frames(
            self._sampler(np.asarray(q, float), time_s),
            np.asarray(v, dtype=float),
            drift,
            control,
            unloaded,
        )
        return PartSplit(axial, part_ground(drift.ground, control.ground))

    def __call__(
        self, q: Any, v: Any, tau: Any, time_s: float
    ) -> dict[str, AxialLoadFrame | None]:
        """The split section loads at one state."""
        return self.split(q, v, tau, time_s).axial

    def summed(self, q: Any, v: Any, tau: Any, time_s: float) -> Any:
        """Inverse dynamics of the summed solve (the check on ``total``)."""
        from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
            constrained_acceleration_at,
        )

        q = np.asarray(q, dtype=float)
        a, lam = constrained_acceleration_at(self._split.provider, q, v, tau)
        load = self._load(q, a, lam)
        return self._sampler(q, time_s)(v, a, load.ground, load.closure)


def swing_part_frames(
    swing: Any,
) -> Callable[[int], dict[str, AxialLoadFrame | None]]:
    """Split section loads by source state of ``swing``, lazily, cached.

    Raises ``ValueError`` for a kinematic replay (no efforts to split).
    """
    splits = swing_part_splits(swing)
    return lambda index: splits(index).axial


def swing_part_splits(swing: Any) -> Callable[[int], PartSplit]:
    """:class:`PartSplit` by source state of ``swing``, lazily, cached.

    Raises ``ValueError`` for a kinematic replay (no efforts to split).
    """
    from src.tools.native_viewer_export.ztcf_feed import _states

    bundle = swing.bundle
    if not np.any(np.asarray(bundle.efforts)):
        raise ValueError(
            "ZTCF/ZVCF shading needs a dynamics bundle with efforts; this is a "
            "kinematic replay (run the kinetic match first)"
        )
    source = part_axial_source(bundle.spec_bytes, tuple(bundle.coordinate_order))
    q, v, tau = _states(swing)
    dt = float(bundle.dt_s)
    cache: dict[int, PartSplit] = {}

    def at(index: int) -> PartSplit:
        k = min(max(int(index), 0), q.shape[0] - 1)
        if k not in cache:
            cache[k] = source.split(q[k], v[k], tau[k], k * dt)
        return cache[k]

    return at


def part_axial_source(spec_bytes: bytes, order: tuple[str, ...]) -> _PartAxialSource:
    """Split section loads ``(q, v, tau, time_s) -> {part: frame}`` of a spec.

    Raises ``ImportError`` without MuJoCo or Pinocchio.
    """
    return _PartAxialSource(spec_bytes, tuple(order))
