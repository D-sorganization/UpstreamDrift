"""Dual twist and wrench transforms between engine closure conventions (NSX-50).

The closure rows of each engine differ in row order, frame, reference point and
sign.  NSX-02 (:mod:`loop_nullspace.closure_convention`) owns that algebra as the
6 x 6 matrix ``C`` with ``twist_c = C twist_n``.  This module only exposes the
dual pair on top of it; it re-implements none of the row or sign logic::

    twist_c  = C   twist_n          wrench_n = C^T wrench_c
    twist_n  = C^-1 twist_c         wrench_c = C^-T wrench_n

so that the power ``twist . wrench`` is identical in both conventions.  The
sign in ``C`` is the relative-twist sign of the engine's closure convention, so
these are closure (relative grip) twists and wrenches, not single-body spatial
vectors.  Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require
from src.shared.python.loop_nullspace.closure_convention import (
    NativeClosureConvention,
    closure_conversion_matrix,
)

Array = npt.NDArray[np.float64]

_CONVENTION_SOURCES = {
    "pinocchio": "src.engines.physics_engines.pinocchio.python.native_model",
    "mujoco": "src.engines.physics_engines.mujoco.python.native_model",
    "drake": "src.engines.physics_engines.drake.python.full_body_model",
    "opensim": "src.engines.physics_engines.opensim.python.full_body_parity",
}
_POWER_RTOL = 1e-9


def engine_closure_convention(engine: str) -> NativeClosureConvention:
    """Return the verified native closure convention declared by ``engine``."""
    if engine not in _CONVENTION_SOURCES:
        raise ValueError(
            f"unknown engine {engine!r}; expected one of {sorted(_CONVENTION_SOURCES)}"
        )
    module = import_module(_CONVENTION_SOURCES[engine])
    convention = module.CLOSURE_CONVENTION
    if not isinstance(convention, NativeClosureConvention):
        raise TypeError(f"{engine} CLOSURE_CONVENTION is not a NativeClosureConvention")
    return convention


@dataclass(frozen=True)
class ClosureFrames:
    """Frames and reference points that fix the conversion matrix ``C``."""

    native: NativeClosureConvention
    native_frame_rotation: npt.ArrayLike
    grip_frame_rotation: npt.ArrayLike
    native_reference_point: npt.ArrayLike
    grip_reference_point: npt.ArrayLike

    def matrix(self) -> Array:
        """``C`` with ``twist_canonical = C @ twist_native`` (NSX-02)."""
        return closure_conversion_matrix(
            native=self.native,
            native_frame_rotation=self.native_frame_rotation,
            grip_frame_rotation=self.grip_frame_rotation,
            native_reference_point=self.native_reference_point,
            grip_reference_point=self.grip_reference_point,
        )


def _vec6(value: npt.ArrayLike, name: str) -> Array:
    vec = np.asarray(value, dtype=float)
    require(vec.shape == (6,), f"{name} must have length 6")
    require(bool(np.isfinite(vec).all()), f"{name} must be finite")
    return vec


def twist_to_canonical(twist: npt.ArrayLike, frames: ClosureFrames) -> Array:
    """``C @ twist`` (native closure twist to the canonical convention)."""
    return frames.matrix() @ _vec6(twist, "twist")


def twist_from_canonical(twist: npt.ArrayLike, frames: ClosureFrames) -> Array:
    """``C^-1 @ twist`` (inverse of :func:`twist_to_canonical`)."""
    return np.linalg.solve(frames.matrix(), _vec6(twist, "twist"))


def wrench_from_canonical(wrench: npt.ArrayLike, frames: ClosureFrames) -> Array:
    """``C^T @ wrench``: the native wrench dual to :func:`twist_to_canonical`."""
    return frames.matrix().T @ _vec6(wrench, "wrench")


def wrench_to_canonical(wrench: npt.ArrayLike, frames: ClosureFrames) -> Array:
    """``C^-T @ wrench`` (inverse of :func:`wrench_from_canonical`)."""
    return np.linalg.solve(frames.matrix().T, _vec6(wrench, "wrench"))


def check_pairing_preserved(
    twist_a: npt.ArrayLike,
    wrench_a: npt.ArrayLike,
    twist_b: npt.ArrayLike,
    wrench_b: npt.ArrayLike,
    *,
    rtol: float = _POWER_RTOL,
) -> float:
    """Raise ``ValueError`` unless ``twist_a . wrench_a == twist_b . wrench_b``.

    Returns the pairing ``twist_a . wrench_a``; tolerance is relative to the
    product of the vector norms (floor ``1e-12``).
    """
    ta, wa = _vec6(twist_a, "twist_a"), _vec6(wrench_a, "wrench_a")
    tb, wb = _vec6(twist_b, "twist_b"), _vec6(wrench_b, "wrench_b")
    power_a, power_b = float(ta @ wa), float(tb @ wb)
    scale = max(float(np.linalg.norm(ta) * np.linalg.norm(wa)), 1e-12)
    if abs(power_a - power_b) > rtol * scale:
        raise ValueError(f"power not preserved: {power_a!r} vs {power_b!r}")
    return power_a


__all__ = [
    "ClosureFrames",
    "check_pairing_preserved",
    "engine_closure_convention",
    "twist_from_canonical",
    "twist_to_canonical",
    "wrench_from_canonical",
    "wrench_to_canonical",
]
