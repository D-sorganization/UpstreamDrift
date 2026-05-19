"""Engine-agnostic TDD oracle.

Mirror of ``synthesize_target_from_coefficients.m``.

Per CROSS_ENGINE_PARITY_SPEC §2.1, "Engine-specific loaders are forbidden."
That extends to the TDD oracle: every engine-specific forward simulator is
exposed as an :class:`EngineSimulator` protocol that returns a :class:`SimOut`,
and this module's :func:`synthesize_target_from_coefficients` runs that
simulator and converts the result into a :class:`ClubTarget` -- the canonical
oracle defined in CLUB_IK_SPEC.md.

Rationale: optimizer code can be written once against this signature and
swapped between Simscape, Drake, MuJoCo, Pinocchio, ... by passing a
different simulator object.

Public API:
    EngineSimulator                       -- Protocol every engine must satisfy.
    SynthOptions                          -- mirrors ``default_synth_options.m``.
    synthesize_target_from_coefficients   -- engine-agnostic oracle.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .sim_out import SimOut
from .target import AlignOptions, ClubTarget, SourceProvenance
from .validators import must_be_finite_vector

logger = logging.getLogger(__name__)

__all__ = [
    "EngineSimulator",
    "SynthOptions",
    "THETA_BOUNDS",
    "synthesize_target_from_coefficients",
]

# Per-coefficient bounds matching ``generateRandomCoefficients.m``:
#   |A|, |B| <= 1000; |C|, |D| <= 500; |E|, |F| <= 100; |G| <= 25.
# Order is [A B C D E F G] per joint.
THETA_BOUNDS: tuple[float, ...] = (1000.0, 1000.0, 500.0, 500.0, 100.0, 100.0, 25.0)


@runtime_checkable
class EngineSimulator(Protocol):
    """Engine-agnostic forward-simulator interface.

    An :class:`EngineSimulator` is anything callable like
    ``sim_out = engine(theta, *, sample_rate_hz, simulation_time_s)`` that
    returns a :class:`SimOut`. This is the seam between this oracle and the
    per-engine implementation: Simscape, Drake, MuJoCo, Pinocchio each
    provide a class implementing this protocol.
    """

    def __call__(
        self,
        theta: NDArray[np.float64],
        *,
        sample_rate_hz: float,
        simulation_time_s: float,
    ) -> SimOut:
        """Return per-frame :class:`SimOut` for coefficient vector ``theta``."""


@dataclass(frozen=True)
class SynthOptions:
    """Mirror of ``default_synth_options.m``.

    Attributes:
        sample_rate_hz:    Output timegrid rate. Default 1 kHz.
        simulation_time_s: Total simulation duration. Default 0.3 s.
        subject_id:        Free-form provenance identifier.
        trial_id:          Free-form trial identifier.
        add_noise:         Whether to inject Gaussian position noise.
        noise_sigma_m:     Stdev of position noise in metres.
        noise_seed:        RNG seed for reproducible noise.
    """

    sample_rate_hz: float = 1000.0
    simulation_time_s: float = 0.3
    subject_id: str = "synthetic"
    trial_id: str = "synthetic"
    add_noise: bool = False
    noise_sigma_m: float = 0.0
    noise_seed: int = 0


def _validate_theta_bounds(theta: NDArray[np.float64]) -> None:
    """Enforce per-coefficient bounds; raises ``ValueError`` on violation."""
    n = theta.shape[0]
    if n == 0 or n % 7 != 0:
        raise ValueError(f"theta length must be a positive multiple of 7 (got {n})")
    n_joints = n // 7
    m = theta.reshape(n_joints, 7)
    for c, bound in enumerate(THETA_BOUNDS):
        if np.any(np.abs(m[:, c]) > bound + 1e-9):
            letter = chr(ord("A") + c)
            raise ValueError(f"coefficient {letter} exceeds +/-{bound:g}")


def _normalise_quat_rows(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Force unit-norm and ``q[:, 0] >= 0`` sign convention."""
    if q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError("club_quat must be a non-empty finite (N, 4) matrix")
    # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~3x faster than np.linalg.norm(..., axis=1)
    norms = np.sqrt(np.einsum("ij,ij->i", q, q))[:, np.newaxis]
    norms = np.where(norms == 0.0, 1.0, norms)
    out = q / norms
    flips = out[:, 0] < 0.0
    if np.any(flips):
        out = out.copy()
        out[flips] = -out[flips]
    return out


def _detect_impact_idx(time: NDArray[np.float64], clubhead: NDArray[np.float64]) -> int:
    """1-based argmax of ``||d r_clubhead/dt||`` -- matches MATLAB convention."""
    from .loaders._align import detect_impact_index

    return int(detect_impact_index(time, clubhead)) + 1


def synthesize_target_from_coefficients(
    theta: NDArray[np.float64],
    engine: EngineSimulator,
    opts: SynthOptions | None = None,
    *,
    align_opts: AlignOptions | None = None,
) -> ClubTarget:
    """Run ``engine`` on ``theta`` and return a canonical :class:`ClubTarget`.

    This is the engine-agnostic TDD oracle: any optimiser that cannot
    recover ``theta`` (within the spec'd RMSE) when given the result of
    this function as input is broken -- not the data.

    Args:
        theta:      Real, finite 1-D coefficient vector with ``len % 7 == 0``.
        engine:     Object satisfying :class:`EngineSimulator`.
        opts:       :class:`SynthOptions`; defaults match ``default_synth_options.m``.
        align_opts: Optional :class:`AlignOptions`; if provided, the impact
                    target time on the simulation grid is taken from
                    ``align_opts.impact_target_t_s``.

    Returns:
        Validated :class:`ClubTarget`.

    Raises:
        ValueError: If ``theta`` violates the bounds, the engine returns
            inconsistent shapes, or the resulting target fails the
            CLUB_IK_SPEC validation.
        TypeError: If ``engine`` doesn't satisfy :class:`EngineSimulator`.
    """
    options = opts if opts is not None else SynthOptions()
    if not callable(engine):
        raise TypeError(
            "engine must satisfy EngineSimulator (callable returning SimOut); "
            f"got {type(engine).__name__}"
        )
    theta = must_be_finite_vector(theta)
    _validate_theta_bounds(theta)

    sim_out = engine(
        theta,
        sample_rate_hz=options.sample_rate_hz,
        simulation_time_s=options.simulation_time_s,
    )
    if not isinstance(sim_out, SimOut):
        raise TypeError(f"engine must return SimOut; got {type(sim_out).__name__}")
    if sim_out.time is None:
        raise ValueError("engine SimOut.time is required for the oracle")

    time = np.asarray(sim_out.time, dtype=np.float64).reshape(-1)
    butt = np.asarray(sim_out.butt, dtype=np.float64)
    clubhead = np.asarray(sim_out.clubhead, dtype=np.float64)
    quat = _normalise_quat_rows(np.asarray(sim_out.club_quat, dtype=np.float64))

    if options.add_noise and options.noise_sigma_m > 0.0:
        rng = np.random.default_rng(options.noise_seed)
        sigma = float(options.noise_sigma_m)
        butt = butt + sigma * rng.standard_normal(butt.shape)
        clubhead = clubhead + sigma * rng.standard_normal(clubhead.shape)

    if align_opts is not None and align_opts.impact_target_t_s > 0:
        # Align to a known impact time on the engine's grid (best-effort).
        impact_idx = int(np.argmin(np.abs(time - align_opts.impact_target_t_s))) + 1
    else:
        impact_idx = _detect_impact_idx(time, clubhead)

    theta_hash = hashlib.sha256(theta.tobytes()).hexdigest()
    source = SourceProvenance(
        filename="",
        format="synthetic",
        subject_id=str(options.subject_id),
        trial_id=str(options.trial_id),
        sha256=theta_hash,
    )

    logger.debug(
        "Synthesised ClubTarget for theta of length %d (engine=%s, n=%d, impact=%d)",
        theta.shape[0],
        type(engine).__name__,
        time.shape[0],
        impact_idx,
    )
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=impact_idx,
        source=source,
    )
