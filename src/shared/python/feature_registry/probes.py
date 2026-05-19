"""Probe adapters for the feature registry.

This module is a thin shim that exposes the existing engine-probe
hierarchy in :mod:`src.shared.python.engine_core.engine_probes` plus
two new probes for features that lacked one (MediaPipe, PyChrono,
PyTorch, RL stack).

We deliberately do **not** duplicate probe logic. The registry's
contract is "given a feature name, return a uniform ``ProbeOutcome``;"
how that outcome was produced is the probe's business.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.engine_core.engine_probes import (
    DrakeProbe,
    EngineProbeResult,
    MuJoCoProbe,
    MyoSimProbe,
    OpenPoseProbe,
    OpenSimProbe,
    PendulumProbe,
    PinocchioProbe,
    ProbeStatus,
)


@dataclass(frozen=True)
class ProbeOutcome:
    """Registry-level probe outcome — uniform across all features.

    Wraps :class:`EngineProbeResult` so non-engine features (MediaPipe,
    PyTorch, RL stack) can share the same return type without forcing
    them through the engine-specific result schema.

    Attributes:
        available: Whether the feature is usable right now.
        version: Best-effort version string, ``None`` if unknown.
        message: Human-readable diagnostic; suitable for tooltips.
        missing: Names of missing sub-dependencies, when known.
    """

    available: bool
    version: str | None
    message: str
    missing: tuple[str, ...] = ()


def _from_engine_result(result: EngineProbeResult) -> ProbeOutcome:
    """Adapt an :class:`EngineProbeResult` into a :class:`ProbeOutcome`."""
    return ProbeOutcome(
        available=result.is_available(),
        version=result.version,
        message=result.diagnostic_message,
        missing=tuple(result.missing_dependencies),
    )


# ---------------------------------------------------------------------------
# Engine-probe wrappers (delegate to the existing probe classes)
# ---------------------------------------------------------------------------


def _probe_mujoco(suite_root: Path) -> ProbeOutcome:
    return _from_engine_result(MuJoCoProbe(suite_root).probe())


def _probe_drake(suite_root: Path) -> ProbeOutcome:
    return _from_engine_result(DrakeProbe(suite_root).probe())


def _probe_pinocchio(suite_root: Path) -> ProbeOutcome:
    return _from_engine_result(PinocchioProbe(suite_root).probe())


def _probe_opensim(suite_root: Path) -> ProbeOutcome:
    return _from_engine_result(OpenSimProbe(suite_root).probe())


def _probe_myosuite(suite_root: Path) -> ProbeOutcome:
    return _from_engine_result(MyoSimProbe(suite_root).probe())


def _probe_pendulum(suite_root: Path) -> ProbeOutcome:
    return _from_engine_result(PendulumProbe(suite_root).probe())


def _probe_openpose(suite_root: Path) -> ProbeOutcome:
    return _from_engine_result(OpenPoseProbe(suite_root).probe())


# ---------------------------------------------------------------------------
# New probes for features that previously had none.
# Kept self-contained here (rather than extending engine_probes.py) because
# they are not engines.
# ---------------------------------------------------------------------------


def _probe_mediapipe(_suite_root: Path) -> ProbeOutcome:
    try:
        import mediapipe as mp  # type: ignore[import-untyped]

        version = getattr(mp, "__version__", "unknown")
        return ProbeOutcome(
            available=True,
            version=version,
            message=f"MediaPipe {version} ready",
        )
    except ImportError:
        return ProbeOutcome(
            available=False,
            version=None,
            message=(
                "MediaPipe not installed. Install with: "
                "pip install 'upstream-drift[pose]'"
            ),
            missing=("mediapipe",),
        )


def _probe_chrono(_suite_root: Path) -> ProbeOutcome:
    try:
        import pychrono  # type: ignore[import-untyped]

        version = getattr(pychrono, "__version__", "unknown")
        return ProbeOutcome(
            available=True,
            version=version,
            message=f"PyChrono {version} ready",
        )
    except ImportError:
        return ProbeOutcome(
            available=False,
            version=None,
            message=(
                "PyChrono not installed. Conda-forge install: "
                "conda install -c projectchrono pychrono"
            ),
            missing=("pychrono",),
        )


def _probe_torch(_suite_root: Path) -> ProbeOutcome:
    try:
        import torch  # type: ignore[import-untyped]

        cuda_available = bool(getattr(torch.cuda, "is_available", lambda: False)())
        suffix = " (CUDA)" if cuda_available else " (CPU only)"
        return ProbeOutcome(
            available=True,
            version=torch.__version__,
            message=f"PyTorch {torch.__version__}{suffix}",
        )
    except ImportError:
        return ProbeOutcome(
            available=False,
            version=None,
            message=(
                "PyTorch not installed. CUDA wheels: "
                "pip install 'torch==2.8.0' "
                "--index-url https://download.pytorch.org/whl/cu124"
            ),
            missing=("torch",),
        )


def _probe_rl(_suite_root: Path) -> ProbeOutcome:
    missing: list[str] = []
    for module_name in ("gymnasium", "stable_baselines3"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        return ProbeOutcome(
            available=False,
            version=None,
            message=(
                f"RL stack incomplete; missing: {', '.join(missing)}. "
                "Install: pip install 'upstream-drift[rl]'"
            ),
            missing=tuple(missing),
        )
    return ProbeOutcome(
        available=True,
        version=None,
        message="RL stack (gymnasium + stable-baselines3) ready",
    )


# ---------------------------------------------------------------------------
# Probe registry — maps the ``Feature.probe_key`` to a callable.
# ---------------------------------------------------------------------------

ProbeCallable = Callable[[Path], ProbeOutcome]

PROBES: dict[str, ProbeCallable] = {
    "mujoco": _probe_mujoco,
    "drake": _probe_drake,
    "pinocchio": _probe_pinocchio,
    "opensim": _probe_opensim,
    "myosuite": _probe_myosuite,
    "pendulum": _probe_pendulum,
    "openpose": _probe_openpose,
    "mediapipe": _probe_mediapipe,
    "chrono": _probe_chrono,
    "torch": _probe_torch,
    "rl": _probe_rl,
}


__all__ = [
    "ProbeOutcome",
    "ProbeCallable",
    "PROBES",
    "ProbeStatus",
]
