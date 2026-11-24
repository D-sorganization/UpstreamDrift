"""Tip-to-tail ZTCF/ZVCF hand-force arrows for the native export (#15685, K5).

Per frame, the force the closing (trail) hand applies to the club splits into
its drift part (ZTCF, ``tau = 0``) and its control part (ZVCF, the applied
torques alone); the two are drawn tip to tail with the total. The split is
evaluated pointwise on the closed-loop Pinocchio provider of the bundle's own
specification (``hand_force_split``), so it needs a dynamics bundle whose
efforts are the torques that produced the motion; a kinematic replay (zero
efforts) is refused.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np

from src.shared.python.force_overlay.glyphs import ArrowGlyph, ForceGlyphStyle
from src.shared.python.force_overlay.tip_to_tail import (
    TipToTailStyle,
    tip_to_tail_arrows,
)

__all__ = ["LEGEND", "ztcf_zvcf_arrows"]

LEGEND = (
    "Trail hand on club: ZTCF drift (blue) + ZVCF control (orange) = total "
    "(white); one null-space solution"
)


def _closure_provider(spec_bytes: bytes, order: tuple[str, ...]) -> tuple[Any, Any]:
    """Closed-loop provider in ``order`` and its grip-frame function."""
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )
    from src.shared.python.loop_nullspace.provider import (
        AdapterConstrainedProvider,
        ReorderedProvider,
    )

    adapter = PinocchioForceAdapter(spec_bytes)
    native = tuple(adapter.coordinate_order)
    provider: Any = AdapterConstrainedProvider(adapter)
    if native == order:
        return provider, adapter.closure_wrench_frame
    take = np.array([order.index(n) for n in native])
    reordered = ReorderedProvider(provider, native, order)
    return reordered, lambda q: adapter.closure_wrench_frame(np.asarray(q)[take])


def _states(swing: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(q, v, tau)`` per source state of ``swing`` (efforts held per step)."""
    bundle = swing.bundle
    q = np.asarray(swing.q, dtype=float)
    times = np.arange(q.shape[0], dtype=float) * float(bundle.dt_s)
    if swing.v is not None:
        v = np.asarray(swing.v, dtype=float)
    elif q.shape == bundle.reference_v.shape:
        v = np.asarray(bundle.reference_v, dtype=float)
    else:
        v = np.gradient(q, times, axis=0)
    efforts = np.asarray(bundle.efforts, dtype=float)
    step = np.clip((times / float(bundle.dt_s)).astype(int), 0, efforts.shape[0] - 1)
    return q, v, efforts[step]


def ztcf_zvcf_arrows(
    swing: Any,
    style: ForceGlyphStyle,
    colours: TipToTailStyle | None = None,
) -> Callable[[int], tuple[ArrowGlyph, ...]]:
    """Arrows by source state index, evaluated lazily and cached.

    Raises:
        ValueError: the bundle has no efforts (a kinematic replay), or its
            specification declares no closure.
    """
    from src.shared.python.simulation_backends.hand_force_split import (
        hand_force_split,
    )

    bundle = swing.bundle
    if not np.any(np.asarray(bundle.efforts)):
        raise ValueError(
            "the ZTCF/ZVCF hand force needs a dynamics bundle with efforts; this "
            "is a kinematic replay (run the kinetic match first)"
        )
    if "closure" not in json.loads(bundle.spec_bytes):
        raise ValueError("the specification declares no hand-club closure")
    provider, frame_at = _closure_provider(
        bundle.spec_bytes, tuple(bundle.coordinate_order)
    )
    q, v, tau = _states(swing)
    cache: dict[int, tuple[ArrowGlyph, ...]] = {}

    def arrows(index: int) -> tuple[ArrowGlyph, ...]:
        k = min(max(int(index), 0), q.shape[0] - 1)
        if k not in cache:
            split = hand_force_split(provider, frame_at, q[k], v[k], tau[k])
            cache[k] = tip_to_tail_arrows(
                split.point_m[0], split.drift_n[0], split.control_n[0], style, colours
            )
        return cache[k]

    return arrows
