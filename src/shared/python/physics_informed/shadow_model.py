"""Shadow model for non-intrusive observation of PINN residuals per swing phase.

Phase 3 of the PINNs epic (#5419). Provides:

- :class:`SwingPhase`: Enum for the three golf-swing phases.
- :class:`ShadowReport`: Dataclass holding peak residual torques per phase.
- :class:`ShadowModel`: Runs rigid simulation + PINN simultaneously in
  *observation mode*, recording peak residual torques per phase without
  modifying simulation state.

The model degrades gracefully: if JAX (or any optional dependency) is not
installed, :meth:`ShadowModel.observe` returns an empty :class:`ShadowReport`
rather than raising.

Part of epic #5419. Closes #5499.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.shared.python.physics_informed.mlp_residual import MlpResidual
    from src.shared.python.physics_informed.rigid_core import RigidCore

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and dataclasses
# =============================================================================


class SwingPhase(Enum):
    """Segmentation of a golf swing into three biomechanical phases."""

    TRANSITION = "transition"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"


@dataclass
class ShadowReport:
    """Container for peak residual torques recorded during shadow observation.

    Attributes:
        peak_residuals: Mapping from :attr:`SwingPhase.value` string to the
            peak (max abs) residual torque magnitude recorded in that phase.
            Empty when no frames were observed or when optional dependencies
            are unavailable.
    """

    peak_residuals: dict[str, float] = field(default_factory=dict)


# =============================================================================
# Phase classification helpers
# =============================================================================


def _classify_phase(frame_idx: int, total_frames: int) -> SwingPhase:
    """Return the SwingPhase for a given frame index.

    Frame distribution (fraction of total):
    - First 30 %  -> TRANSITION
    - Middle 40 % -> IMPACT
    - Last  30 %  -> FOLLOW_THROUGH

    DbC preconditions:
    - ``total_frames >= 1``
    - ``0 <= frame_idx < total_frames``

    Args:
        frame_idx:    Zero-based index of the frame.
        total_frames: Total number of frames in the sequence.

    Returns:
        The :class:`SwingPhase` for the given frame.
    """
    ratio = frame_idx / total_frames
    if ratio < 0.30:
        return SwingPhase.TRANSITION
    if ratio < 0.70:
        return SwingPhase.IMPACT
    return SwingPhase.FOLLOW_THROUGH


# =============================================================================
# ShadowModel
# =============================================================================


class ShadowModel:
    """Runs rigid simulation + PINN simultaneously in observation mode.

    Records peak residual torques per swing phase *without* modifying
    simulation state.  The "residual" is the MLP output -- what the rigid-body
    model alone would miss.

    Parameters
    ----------
    rigid_core:
        Configured :class:`~.rigid_core.RigidCore` instance (Pinocchio).
    mlp_residual:
        Configured :class:`~.mlp_residual.MlpResidual` instance (JAX/Equinox).

    Raises
    ------
    ValueError
        If ``rigid_core`` or ``mlp_residual`` is ``None``.
    """

    def __init__(
        self,
        rigid_core: RigidCore,
        mlp_residual: MlpResidual,
    ) -> None:
        """Compose rigid-body core and MLP residual for shadow observation.

        DbC preconditions:
        - ``rigid_core`` must not be ``None``.
        - ``mlp_residual`` must not be ``None``.

        Args:
            rigid_core:   Pinocchio-backed rigid-body torque calculator.
            mlp_residual: JAX/Equinox MLP residual torque predictor.

        Raises:
            ValueError: If either argument is ``None``.
        """
        if rigid_core is None:
            raise ValueError("rigid_core must not be None")
        if mlp_residual is None:
            raise ValueError("mlp_residual must not be None")

        self._rigid = rigid_core
        self._mlp = mlp_residual

        logger.debug("ShadowModel created in observation mode")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, frames: list[dict]) -> ShadowReport:
        """Run one pass over simulation frames, recording peak residuals.

        Each frame dict must contain:
        - ``q``   -- configuration vector (numpy array)
        - ``dq``  -- velocity vector (numpy array)
        - ``ddq`` -- acceleration vector (numpy array)

        The method classifies frames into phases by fractional index:
        - first 30 % -> TRANSITION
        - middle 40 % -> IMPACT
        - last 30 %  -> FOLLOW_THROUGH

        For each frame, the MLP residual (the part the rigid model misses)
        is computed as ``mlp(concat([q, dq, ddq]))``.  The peak
        ``max(abs(residual))`` is tracked per phase.

        This method is *observation-only*: it never modifies input frames
        or simulation state.

        Args:
            frames: List of frame dicts, each with keys ``q``, ``dq``, ``ddq``.

        Returns:
            :class:`ShadowReport` with ``peak_residuals`` keyed by
            :attr:`SwingPhase.value` strings.  Returns an empty report if
            ``frames`` is empty or if an :class:`ImportError` is raised by
            an optional dependency (graceful degradation without JAX).
        """
        if not frames:
            return ShadowReport()

        phase_peaks: dict[str, float] = {}

        try:
            total = len(frames)
            for idx, frame in enumerate(frames):
                phase = _classify_phase(idx, total)
                peak = self._compute_frame_peak(frame)
                key = phase.value
                if key not in phase_peaks or peak > phase_peaks[key]:
                    phase_peaks[key] = peak

        except ImportError as exc:
            logger.warning(
                "ShadowModel.observe: optional dependency unavailable (%s); "
                "returning empty ShadowReport",
                exc,
            )
            return ShadowReport()

        logger.debug(
            "ShadowModel.observe: %d frames -> peak_residuals=%s",
            len(frames),
            phase_peaks,
        )
        return ShadowReport(peak_residuals=phase_peaks)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_frame_peak(self, frame: dict) -> float:
        """Compute max(abs(mlp_residual)) for a single frame.

        The MLP receives concat([q, dq, ddq]) and produces a residual torque
        vector.  We then call rigid_core.compute_torques to ensure correctness
        (e.g. raise ImportError early if Pinocchio is missing).

        Args:
            frame: Dict with keys ``q``, ``dq``, ``ddq``.

        Returns:
            Scalar peak residual magnitude.

        Raises:
            ImportError: Propagated from rigid_core or mlp_residual if
                optional dependencies are missing.
        """
        q = np.asarray(frame["q"], dtype=np.float64)
        dq = np.asarray(frame["dq"], dtype=np.float64)
        ddq = np.asarray(frame["ddq"], dtype=np.float64)

        # Compute rigid torques to surface any ImportError early
        self._rigid.compute_torques(q, dq, ddq)

        # MLP residual: predict what rigid misses
        x = np.concatenate([q, dq, ddq])
        mlp_out = self._mlp(x)
        residuals = np.asarray(mlp_out, dtype=np.float64)

        return float(np.max(np.abs(residuals)))
