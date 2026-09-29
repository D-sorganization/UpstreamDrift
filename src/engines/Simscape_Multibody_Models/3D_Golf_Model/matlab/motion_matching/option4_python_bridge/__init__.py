"""Option 4 — Python ↔ Simscape bridge.

This package wraps GolfSwing3D_Kinetic.slx via the MATLAB Engine API for
Python so that scipy/JAX-driven optimizers can drive the Simscape forward
simulation from Python.

See APPROACH.md, INTERFACES.md, and INSTALLATION.md for the full contract.

Lazy-import policy:
    Importing this package must NOT import ``matlab.engine``. The engine is
    a large, license-gated dependency; tests on machines without MATLAB must
    still be collectible. Concrete classes (``SimscapeAdapter``,
    ``fit_swing_scipy``) are imported on first use through the helpers below.
"""

from __future__ import annotations

__all__ = [
    "SimscapeAdapter",
    "SimOut",
    "ClubTarget",
    "fit_swing_scipy",
]


def __getattr__(name: str):  # pragma: no cover - thin import shim
    if name in {"SimscapeAdapter", "SimOut", "ClubTarget"}:
        from .simscape_adapter import ClubTarget, SimOut, SimscapeAdapter

        return {
            "SimscapeAdapter": SimscapeAdapter,
            "SimOut": SimOut,
            "ClubTarget": ClubTarget,
        }[name]
    if name == "fit_swing_scipy":
        from .fit_swing_python import fit_swing_scipy

        return fit_swing_scipy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
