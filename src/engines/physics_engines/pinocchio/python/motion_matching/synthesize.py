"""TDD oracle for the Pinocchio motion-matching pipeline (issue #4121).

This module implements ``synthesize_target_from_coefficients`` -- the
canonical *forward* half of the optimiser oracle. Given a coefficient
vector ``theta``, it runs ``simulate_with_coefficients`` and packages
the result into a fully-validated :class:`ClubTarget` per
``CLUB_IK_SPEC.md``.

The schema mapping from the Pinocchio :class:`SimOut` to the canonical
``ClubTarget`` is the single source of truth for the parity surface:

* ``time``      <- ``SimOut.t``
* ``butt``      <- ``SimOut.grip_position``     (mid-hands frame)
* ``clubhead``  <- ``SimOut.clubhead_position`` (club-head frame)
* ``club_quat`` <- quaternion of ``SimOut.clubhead_rotation`` (Shepperd)
* ``impact_idx``<- 1-based ``argmax`` of clubhead linear speed (parity
  with the Simscape MATLAB oracle's ``detect_clubhead_impact``)
* ``source``    <- :class:`SourceProvenance` with ``format='synthetic'``,
  ``filename='synthetic'``, ``trial_id`` derived from the theta hash

CLAUDE.md gotcha (echoed): never call ``pin.computeTotalEnergy``. The
upstream simulator already obeys this; this wrapper does not touch
energy at all.

See also:
    * :mod:`...simulate` -- the forward simulator.
    * ``src.shared.python.motion_matching.club_target`` -- the canonical
      schema and validation rules.
    * ``src.shared.python.motion_matching.loaders.synthetic`` -- the
      shared dispatcher that delegates here when ``opts.engine ==
      'pinocchio'``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.motion_matching.club_target import (
    AlignOptions,
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.loaders._quaternion import rotmat_to_quat

from .simulate import COEFFS_PER_JOINT, SimOptions, SimOut, simulate_with_coefficients

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthesizeOptions:
    """Engine-specific options for :func:`synthesize_target_from_coefficients`.

    The :class:`AlignOptions` sample-rate / simulation-time fields drive
    the underlying :class:`SimOptions`; the remaining fields are
    Pinocchio-specific.

    Attributes:
        sim_options: Forward-simulator options. ``None`` -> defaults
            built from ``align``: ``dt = 1 / align.sample_rate_hz``,
            ``t_final = align.simulation_time_s``.
        align: ``AlignOptions`` for downstream resampling/provenance.
        subject_id: Provenance subject id. Defaults to ``"synthetic"``.
        trial_id: Provenance trial id. ``None`` -> ``f"theta_{sha[:8]}"``.
        initial_pose: Optional initial state forwarded to the simulator.
    """

    sim_options: SimOptions | None = None
    align: AlignOptions = AlignOptions()
    subject_id: str = "synthetic"
    trial_id: str | None = None
    initial_pose: dict[str, Any] | None = None


def _sim_options_from_align(align: AlignOptions) -> SimOptions:
    """Build a :class:`SimOptions` consistent with the canonical timegrid."""
    if align.sample_rate_hz <= 0:
        msg = f"align.sample_rate_hz must be positive, got {align.sample_rate_hz!r}"
        raise ValueError(msg)
    if align.simulation_time_s <= 0:
        msg = (
            f"align.simulation_time_s must be positive, got {align.simulation_time_s!r}"
        )
        raise ValueError(msg)
    return SimOptions(
        t_final=float(align.simulation_time_s),
        dt=1.0 / float(align.sample_rate_hz),
    )


def _theta_sha256(theta: npt.NDArray[np.float64]) -> str:
    """Hex sha256 of the canonical ``theta`` byte representation.

    Uses the ``float64`` byte image so the hash is reproducible across
    interpreters. Distinct ``theta`` always produce distinct hashes
    (collision probability negligible for our scale).
    """
    canonical = np.ascontiguousarray(theta, dtype=np.float64).tobytes()
    return hashlib.sha256(canonical).hexdigest()


def _impact_idx_from_clubhead(clubhead: npt.NDArray[np.float64]) -> int:
    """1-based index of peak clubhead linear speed (parity with MATLAB).

    Matches the postcondition in ``CLUB_IK_SPEC.md``: ``1 <= impact_idx
    <= N``. The reference MATLAB implementation uses the discrete
    derivative ``diff(clubhead, axis=0)`` and reports the speed
    argmax + 1; we replicate that convention so the oracle is bit-for-
    bit comparable across engines.
    """
    if clubhead.ndim != 2 or clubhead.shape[1] != 3:
        msg = f"clubhead must have shape (N, 3); got {clubhead.shape}"
        raise ValueError(msg)
    if clubhead.shape[0] < 2:
        msg = (
            f"clubhead needs >= 2 samples for impact detection; got {clubhead.shape[0]}"
        )
        raise ValueError(msg)
    diffs = np.asarray(np.diff(clubhead, axis=0), dtype=np.float64)
    # ``np.argmax`` returns the first occurrence (0-based) into ``diffs``;
    # the corresponding clubhead sample lies at index ``argmax + 1``.
    # That happens to keep us in [1, N-1], i.e. strictly inside the
    # interval, away from boundaries.
    # ⚡ Bolt: np.argmax(np.einsum(...)) avoids allocations & sqrt overhead
    return int(np.argmax(np.einsum("ij,ij->i", diffs, diffs))) + 1


def _sim_out_to_club_target(
    sim_out: SimOut,
    theta: npt.NDArray[np.float64],
    subject_id: str,
    trial_id: str,
) -> ClubTarget:
    """Pure adapter from :class:`SimOut` to :class:`ClubTarget`.

    Split out so the unit test can mock ``simulate_with_coefficients``
    and exercise the schema mapping without importing pinocchio.
    """
    quat = rotmat_to_quat(sim_out.clubhead_rotation)
    impact_idx = _impact_idx_from_clubhead(sim_out.clubhead_position)
    sha = _theta_sha256(theta)
    source = SourceProvenance(
        filename="synthetic",
        format="synthetic",
        subject_id=subject_id,
        trial_id=trial_id,
        sha256=sha,
    )
    return ClubTarget(
        time=np.asarray(sim_out.t, dtype=np.float64),
        butt=np.asarray(sim_out.grip_position, dtype=np.float64),
        clubhead=np.asarray(sim_out.clubhead_position, dtype=np.float64),
        club_quat=quat,
        impact_idx=int(impact_idx),
        source=source,
    )


def synthesize_target_from_coefficients(
    theta: npt.NDArray[np.float64],
    options: SynthesizeOptions | None = None,
) -> ClubTarget:
    """Forward-synthesize a :class:`ClubTarget` from a known ``theta``.

    This is the TDD oracle: any optimiser that cannot recover ``theta``
    from ``synthesize_target_from_coefficients(theta)`` is broken.

    Args:
        theta: Flat coefficient vector of shape ``(n_joints * 7,)`` in
            the same layout as :func:`simulate_with_coefficients`.
        options: Engine-specific options. ``None`` uses the defaults
            (1 kHz, 0.3 s, neutral initial pose, default gravity).

    Returns:
        Fully-validated :class:`ClubTarget`. The constructor enforces
        every postcondition from ``CLUB_IK_SPEC.md``.

    Raises:
        ValueError: if ``theta`` is non-finite, or if the simulator's
            output fails the ``ClubTarget`` validation rules (e.g. NaN
            position, non-unit quaternion).
        ImportError: if ``pinocchio`` is not installed (re-raised from
            the lazy import inside the simulator).
    """
    opts = options if options is not None else SynthesizeOptions()
    theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta_arr.size == 0:
        raise ValueError("theta must be non-empty")
    if theta_arr.size % COEFFS_PER_JOINT != 0:
        msg = (
            f"theta length {theta_arr.size} must be a multiple of "
            f"{COEFFS_PER_JOINT} (n_joints * coeffs_per_joint)"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta contains non-finite entries")

    sim_options = (
        opts.sim_options
        if opts.sim_options is not None
        else _sim_options_from_align(opts.align)
    )

    sim_out = simulate_with_coefficients(
        theta_arr,
        options=sim_options,
        initial_pose=opts.initial_pose,
    )

    sha = _theta_sha256(theta_arr)
    trial_id = opts.trial_id if opts.trial_id is not None else f"theta_{sha[:8]}"

    target = _sim_out_to_club_target(
        sim_out,
        theta=theta_arr,
        subject_id=opts.subject_id,
        trial_id=trial_id,
    )
    logger.debug(
        "synthesize_target_from_coefficients: theta sha256=%s, n_joints=%d, "
        "n_samples=%d, impact_idx=%d",
        sha,
        sim_out.meta.get("n_joints", -1),
        target.time.shape[0],
        target.impact_idx,
    )
    return target


__all__ = [
    "SynthesizeOptions",
    "synthesize_target_from_coefficients",
]
