"""MuJoCo backend for ``synthesize_target_from_coefficients``.

Implements issue #4122 (PARITY-MUJOCO-SYNTH): the engine-specific oracle
that runs the canonical MuJoCo forward-sim wrapper (``simulate.py``) with a
known ``theta`` and packages the resulting trajectory into a canonical
:class:`ClubTarget` per ``CLUB_IK_SPEC.md``.

Per ``MUJOCO_PARITY_SPEC.md`` §2.2 the oracle MUST round-trip exactly
through ``simulate_with_coefficients`` — this module provides a thin
adapter that maps :class:`SimOut` fields to the :class:`ClubTarget`
schema and never touches mujoco internals directly.

This file is intentionally short. Heavy lifting lives in
``simulate.py`` (issue #4113 / PR #4166).
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.club_target import (
    AlignOptions,
    ClubTarget,
    SourceProvenance,
)

from .simulate import SimOptions, SimOut, simulate_with_coefficients

logger = logging.getLogger(__name__)

__all__ = ["synthesize_target_from_coefficients"]


def _sim_options_from_align(opts: AlignOptions) -> SimOptions:
    """Translate :class:`AlignOptions` into :class:`SimOptions`.

    The cross-engine ``AlignOptions`` carries the simulation horizon and
    sample rate; everything else falls back to the MuJoCo defaults
    (full-body variant, polynomial torque, etc.). Engine-specific knobs
    (variant, dt overrides) are not exposed through ``AlignOptions`` —
    callers that need them must instantiate :class:`SimOptions` directly
    via the lower-level entry point.

    Args:
        opts: Cross-engine align options.

    Returns:
        :class:`SimOptions` with ``T_s`` and ``output_rate_hz`` taken
        from ``opts``; remaining fields default per :class:`SimOptions`.
    """
    return SimOptions(
        T_s=float(opts.simulation_time_s),
        output_rate_hz=float(opts.sample_rate_hz),
    )


def _detect_impact_index(
    time: NDArray[np.float64], clubhead: NDArray[np.float64]
) -> int:
    """Return the 1-based index of peak clubhead speed.

    Impact is conventionally identified as the frame at which
    ``||d r_clubhead / dt||`` is maximal — this matches the MATLAB
    oracle in ``synthesize_target_from_coefficients.m`` and the
    cross-engine spec in ``CROSS_ENGINE_PARITY_SPEC.md``.

    Args:
        time:     ``(N,)`` strictly increasing time grid.
        clubhead: ``(N, 3)`` clubhead position trajectory.

    Returns:
        ``int`` in ``[1, N]`` (1-based, to match :class:`ClubTarget`'s
        validation rule).

    Raises:
        ValueError: if ``time`` and ``clubhead`` row counts disagree or
            the trajectory has fewer than two samples (no derivative
            defined).
    """
    if time.shape[0] != clubhead.shape[0]:
        raise ValueError(
            f"time and clubhead row count mismatch: "
            f"{time.shape[0]} vs {clubhead.shape[0]}"
        )
    if time.shape[0] < 2:
        raise ValueError("need at least two samples to detect impact")
    # np.gradient gives a centred difference (forward/backward at the
    # endpoints), which is more robust than np.diff for picking out the
    # peak speed near the trajectory boundary.
    velocity = np.gradient(clubhead, time, axis=0)
    speed = np.sqrt(np.einsum("ij,ij->i", velocity, velocity))
    # +1 for the 1-based ClubTarget convention enforced by
    # _validate_clubtarget.
    return int(np.argmax(speed)) + 1


def _theta_sha256(theta: NDArray[np.float64]) -> str:
    """SHA-256 hex digest of ``theta``'s float64 byte representation.

    Reproducible across calls for identical input vectors so the
    provenance hash can be used as a cache key. Always cast through
    ``float64`` first so callers passing in ``float32`` or python lists
    get a stable hash.
    """
    canonical = np.ascontiguousarray(np.asarray(theta, dtype=np.float64).reshape(-1))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _normalise_quaternions(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Force unit-norm + non-negative w convention.

    Mirrors the MATLAB ``local_normalise_quat_rows`` helper. Any zero-norm
    rows (which would fail the postcondition in ``_validate_clubtarget``)
    are replaced with the identity quaternion ``[1, 0, 0, 0]``.

    Args:
        quat: ``(N, 4)`` quaternion rows in ``[w, x, y, z]`` order.

    Returns:
        ``(N, 4)`` unit quaternions with ``w >= 0``.
    """
    if quat.shape[1] != 4:
        raise ValueError(f"quat must have 4 columns, got {quat.shape}")
    out = np.asarray(quat, dtype=np.float64).copy()
    norms = np.sqrt(np.einsum("ij,ij->i", out, out))[:, np.newaxis]
    # Identity fallback for any degenerate row.
    bad = norms.reshape(-1) < 1.0e-12
    out[bad] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    norms = np.sqrt(np.einsum("ij,ij->i", out, out))[:, np.newaxis]
    out = out / norms
    flip = out[:, 0] < 0
    out[flip] = -out[flip]
    return out


def _provenance(theta: NDArray[np.float64], opts: AlignOptions) -> SourceProvenance:
    """Build the :class:`SourceProvenance` for a synthetic target.

    The ``subject_id`` is derived from the theta hash so the same theta
    always lands in the same logical bucket, and the ``trial_id`` carries
    the alignment mode for traceability.
    """
    sha = _theta_sha256(theta)
    return SourceProvenance(
        filename="synthetic",
        format="synthetic",
        subject_id=f"theta_seed_{sha[:8]}",
        trial_id=f"mujoco_align_{opts.time_alignment}",
        sha256=sha,
    )


def synthesize_target_from_coefficients(
    theta: NDArray[np.float64],
    opts: AlignOptions | None = None,
    *,
    sim_options: SimOptions | None = None,
    initial_pose: NDArray[np.float64] | None = None,
) -> ClubTarget:
    """Run the MuJoCo forward model with ``theta`` and return a ``ClubTarget``.

    This is the engine-specific implementation registered behind the
    shared dispatcher in ``shared/python/motion_matching/loaders/synthetic.py``
    (see :func:`register_mujoco_backend`). The cross-engine spec
    (CROSS_ENGINE_PARITY_SPEC.md §2.2) requires every backend to produce
    the same :class:`ClubTarget` schema; this thin wrapper enforces that.

    Args:
        theta: ``(n_joints * 7,)`` flat coefficient vector. Layout per
            joint is ``[A, B, C, D, E, F, G]`` matching the polynomial
            torque driver in ``simulate.py``.
        opts:  Cross-engine align options. ``None`` uses the
            :class:`AlignOptions` defaults.
        sim_options: Optional MuJoCo-specific overrides. When given, takes
            precedence over the values derived from ``opts``. Useful for
            picking the MJCF variant (``upper`` / ``full`` / ``advanced``)
            without leaking that knob into the cross-engine signature.
        initial_pose: Optional MuJoCo initial generalized coordinates passed
            through to :func:`simulate_with_coefficients`. ``None`` uses the
            MJCF default pose.

    Returns:
        Validated :class:`ClubTarget` whose trajectory rows mirror the
        underlying :class:`SimOut` (``butt = SimOut.grip``,
        ``clubhead = SimOut.clubhead``, ``club_quat = SimOut.club_quat``).

    Raises:
        ValueError: if ``theta`` is malformed, or the underlying simulator
            reports a non-finite trajectory.
        RuntimeError: propagated from :func:`simulate_with_coefficients` if
            the MJCF compilation or rollout fails.
    """
    if opts is None:
        opts = AlignOptions()
    if not isinstance(opts, AlignOptions):
        raise TypeError(
            f"opts must be an AlignOptions instance; got {type(opts).__name__}"
        )

    theta_arr = np.ascontiguousarray(np.asarray(theta, dtype=np.float64).reshape(-1))
    if theta_arr.size == 0:
        raise ValueError("theta must be non-empty")
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta must be finite (no NaN / Inf)")

    base_sim_opts = _sim_options_from_align(opts)
    if sim_options is not None:
        if not isinstance(sim_options, SimOptions):
            raise TypeError(
                "sim_options must be a SimOptions instance; got "
                f"{type(sim_options).__name__}"
            )
        # Caller override wins on every field.
        merged_sim_opts = sim_options
    else:
        merged_sim_opts = base_sim_opts

    logger.debug(
        "synthesize_target_from_coefficients(mujoco): theta shape %s, "
        "T_s=%s, output_rate_hz=%s",
        theta_arr.shape,
        merged_sim_opts.T_s,
        merged_sim_opts.output_rate_hz,
    )

    sim_out: SimOut = simulate_with_coefficients(
        theta_arr,
        merged_sim_opts,
        initial_pose=initial_pose,
    )

    if sim_out.solver_status not in {"ok", "success"}:
        raise RuntimeError(
            f"MuJoCo rollout did not converge: solver_status={sim_out.solver_status!r}"
        )

    return _build_target(sim_out, theta_arr, opts)


def _build_target(
    sim_out: SimOut,
    theta: NDArray[np.float64],
    opts: AlignOptions,
) -> ClubTarget:
    """Wrap a :class:`SimOut` into a validated :class:`ClubTarget`.

    Field mapping (per the issue body):

    * ``time``       <- ``SimOut.time``
    * ``butt``       <- ``SimOut.grip``     (mid-hands anchor; CLUB_IK_SPEC)
    * ``clubhead``   <- ``SimOut.clubhead``
    * ``club_quat``  <- ``SimOut.club_quat`` (re-normalised to be safe)
    * ``impact_idx`` <- argmax of clubhead speed
    """
    time = np.ascontiguousarray(np.asarray(sim_out.time, dtype=np.float64))
    butt = np.ascontiguousarray(np.asarray(sim_out.grip, dtype=np.float64))
    clubhead = np.ascontiguousarray(np.asarray(sim_out.clubhead, dtype=np.float64))
    club_quat = _normalise_quaternions(np.asarray(sim_out.club_quat, dtype=np.float64))
    impact_idx = _detect_impact_index(time, clubhead)
    source = _provenance(theta, opts)

    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=club_quat,
        impact_idx=impact_idx,
        source=source,
    )


def register_mujoco_backend() -> None:
    """Register this implementation behind the shared synthetic dispatcher.

    Importing this module is enough — the call is idempotent and lazy so
    GUI surfaces that never need MuJoCo don't pull it in transitively.
    """
    # Local import to avoid a circular dependency at module load time
    # (the shared dispatcher might in turn lazy-import this module).
    from src.shared.python.motion_matching.loaders import synthetic as _shared

    _shared.register_backend("mujoco", synthesize_target_from_coefficients)


# Auto-register on import. Safe because ``synthetic.register_backend`` is
# idempotent. The shared dispatcher uses ``opts.engine == "mujoco"`` to
# pick this implementation.
try:
    register_mujoco_backend()
except Exception:  # pragma: no cover - registration is best-effort
    # Never fail import on registration errors; the dispatcher will raise
    # a clear LookupError if the backend is needed but not registered.
    logger.exception("failed to register mujoco synthetic backend")
