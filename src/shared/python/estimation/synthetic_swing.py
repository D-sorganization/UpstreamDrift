"""Synthetic golf-swing benchmark with known torques and mocap-style corruption.

Generates a reproducible planar double-pendulum swing (shoulder + wrist) whose
ground-truth joint torques are known exactly, then corrupts the joint-angle
kinematics the way real motion capture is corrupted: Gaussian noise, gross
outlier spikes and an occlusion gap. Intended as a benchmark for
ZTCF-anchored kinematic matching (UpstreamDrift #11421).

Units: angles [rad], velocities [rad/s], torques [N*m], time [s].

Default swing (verified against the ``ode`` reference backend): start from
``q0 = (-2.0, -1.5)`` at rest and apply a 0.4 s smooth torque pulse with peaks
of 200 N*m (shoulder) and 40 N*m (wrist, delayed to 40 % of the pulse). The
shoulder sweeps ~290 deg and the wrist whips through, reaching peak speeds of
about 29 rad/s (shoulder) and 50 rad/s (wrist) with no blow-up. The more
modest 60/12 N*m peaks only reach ~9/5 rad/s on this model (too gentle).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.core.contracts import ensure, require

_DOF = 2
#: Duration of the default torque pulse [s]; the profile is zero outside it.
DEFAULT_SWING_TIME = 0.4


@dataclass(frozen=True)
class SwingTruth:
    """Clean ground-truth swing.

    Attributes:
        t: Sample times, shape ``(T,)`` [s].
        q: Joint angles ``[shoulder, wrist]``, shape ``(T, 2)`` [rad].
        v: Joint velocities, shape ``(T, 2)`` [rad/s].
        tau: Torque applied during step ``k``, shape ``(T, 2)`` [N*m]; the last
            row repeats the previous row (no step leaves the final sample).
        dt: Sample spacing [s].
    """

    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    tau: np.ndarray
    dt: float


@dataclass(frozen=True)
class CorruptedObservation:
    """Mocap-style corrupted joint angles.

    Attributes:
        q_observed: Noisy angles, shape ``(T, 2)`` [rad]; NaN rows where occluded.
        mask: Shape ``(T,)``; True where an observation is present.
        outlier_indices: Sorted sample indices carrying a gross spike.
        occluded_indices: Sorted sample indices inside the occlusion gap.
        noise_std: Standard deviation of the Gaussian noise [rad].
    """

    q_observed: np.ndarray
    mask: np.ndarray
    outlier_indices: np.ndarray
    occluded_indices: np.ndarray
    noise_std: float


def _bump(s: np.ndarray) -> np.ndarray:
    """C1 bump ``sin^2(pi s)`` on ``s in [0, 1]``, zero elsewhere."""
    inside = (s >= 0.0) & (s <= 1.0)
    return np.where(inside, np.sin(np.pi * np.clip(s, 0.0, 1.0)) ** 2, 0.0)


def smooth_swing_torque(
    t: np.ndarray,
    *,
    peak_shoulder: float = 200.0,
    peak_wrist: float = 40.0,
    swing_time: float = DEFAULT_SWING_TIME,
    wrist_delay: float = 0.4,
) -> np.ndarray:
    """Smooth (C1) downswing-like torque profile.

    The shoulder torque is ``peak_shoulder * sin^2(pi t / swing_time)`` (ramps up
    then down); the wrist torque is the same bump compressed into the final
    ``1 - wrist_delay`` fraction of the pulse (delayed release). Both vanish
    outside ``[0, swing_time]``.

    Args:
        t: Times, shape ``(T,)`` [s].
        peak_shoulder: Shoulder peak torque [N*m].
        peak_wrist: Wrist peak torque [N*m].
        swing_time: Pulse length [s], ``> 0``.
        wrist_delay: Fraction of the pulse before the wrist engages, in [0, 1).

    Returns:
        Torques, shape ``(T, 2)`` [N*m].

    Raises:
        ValueError: On non-finite inputs, ``swing_time <= 0`` or bad delay.
    """
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    require(bool(np.all(np.isfinite(t_arr))), "t must be finite", value=t)
    require(
        np.isfinite(swing_time) and swing_time > 0.0,
        f"swing_time must be positive and finite; got {swing_time!r}",
        value=swing_time,
    )
    require(
        np.isfinite(wrist_delay) and 0.0 <= wrist_delay < 1.0,
        f"wrist_delay must be in [0, 1); got {wrist_delay!r}",
        value=wrist_delay,
    )
    require(
        np.isfinite(peak_shoulder) and np.isfinite(peak_wrist),
        "peak torques must be finite",
    )
    s = t_arr / swing_time
    s_wrist = (s - wrist_delay) / (1.0 - wrist_delay)
    return np.stack([peak_shoulder * _bump(s), peak_wrist * _bump(s_wrist)], axis=1)


def _default_provider() -> Any:
    from src.shared.python.simulation_backends import GolfModelParams, make_backend

    return make_backend("ode", GolfModelParams.default())


def simulate_swing(
    *,
    duration: float = 0.4,
    dt: float = 0.002,
    q0: tuple[float, float] = (-2.0, -1.5),
    v0: tuple[float, float] = (0.0, 0.0),
    torque_fn: Callable[[np.ndarray], np.ndarray] = smooth_swing_torque,
    provider_factory: Callable[[], Any] | None = None,
) -> SwingTruth:
    """Simulate the clean ground-truth swing on the reference backend.

    Args:
        duration: Total simulated time [s]; must exceed ``dt``.
        dt: Integration/sample step [s], ``> 0``.
        q0: Initial ``[shoulder, wrist]`` angles [rad].
        v0: Initial velocities [rad/s].
        torque_fn: Maps times ``(T,)`` to torques ``(T, 2)`` [N*m].
        provider_factory: Zero-arg callable returning a backend with
            ``reset``/``rollout``; defaults to the ``ode`` reference backend.

    Returns:
        :class:`SwingTruth` with ``T = round(duration / dt) + 1`` samples.

    Raises:
        ValueError: On invalid ``dt``/``duration``/initial state or torque shape.
    """
    from src.shared.python.simulation_backends import SimState

    require(np.isfinite(dt) and dt > 0.0, f"dt must be positive and finite; got {dt!r}")
    require(
        np.isfinite(duration) and duration > dt,
        f"duration must be finite and > dt; got {duration!r}",
        value=duration,
    )
    q0_arr = np.asarray(q0, dtype=float)
    v0_arr = np.asarray(v0, dtype=float)
    require(q0_arr.shape == (_DOF,), "q0 must have 2 entries", value=q0)
    require(v0_arr.shape == (_DOF,), "v0 must have 2 entries", value=v0)
    require(
        bool(np.all(np.isfinite(q0_arr)) and np.all(np.isfinite(v0_arr))),
        "q0 and v0 must be finite",
    )

    steps = int(round(duration / dt))
    t = np.arange(steps + 1, dtype=float) * dt
    tau = np.asarray(torque_fn(t), dtype=float)
    require(
        tau.shape == (steps + 1, _DOF),
        f"torque_fn must return shape {(steps + 1, _DOF)}; got {tau.shape}",
        value=tau.shape,
    )
    require(bool(np.all(np.isfinite(tau))), "torque_fn returned non-finite values")
    tau = tau.copy()
    tau[-1] = tau[-2]

    backend = (provider_factory or _default_provider)()
    backend.reset(SimState(q=q0_arr, v=v0_arr, time=0.0))
    trace = backend.rollout(tau[:-1], steps, dt)
    q = np.asarray(trace.q, dtype=float)
    v = np.asarray(trace.v, dtype=float)
    ensure(
        bool(np.all(np.isfinite(q)) and np.all(np.isfinite(v))),
        "simulated swing diverged (non-finite state)",
    )
    return SwingTruth(t=t, q=q, v=v, tau=tau, dt=float(dt))


def corrupt_observations(
    truth: SwingTruth,
    *,
    noise_std: float = 0.002,
    outlier_fraction: float = 0.03,
    outlier_magnitude: float = 0.15,
    occlusion: tuple[int, int] | None = (60, 75),
    seed: int = 0,
) -> CorruptedObservation:
    """Corrupt clean angles with noise, gross outliers and an occlusion gap.

    Gaussian noise (``noise_std``) is added to every sample. Then
    ``round(outlier_fraction * T)`` samples (never index 0/1, never inside the
    occlusion) receive a spike of ``outlier_magnitude * U(1, 1.5)`` rad with
    random sign on one random joint. Occluded rows are set to NaN.

    Args:
        truth: Clean swing.
        noise_std: Noise standard deviation [rad], ``>= 0``.
        outlier_fraction: Fraction of samples spiked, in ``[0, 0.5)``.
        outlier_magnitude: Spike size [rad], ``>= 0``.
        occlusion: Half-open sample range ``[start, stop)`` or ``None``.
        seed: Seed for ``np.random.default_rng`` (fully deterministic).

    Returns:
        :class:`CorruptedObservation`.

    Raises:
        ValueError: On invalid parameters or malformed ``truth``.
    """
    q = np.asarray(truth.q, dtype=float)
    require(q.ndim == 2 and q.shape[1] == _DOF, "truth.q must be (T, 2)")
    n = q.shape[0]
    require(
        np.isfinite(noise_std) and noise_std >= 0.0,
        f"noise_std must be >= 0 and finite; got {noise_std!r}",
        value=noise_std,
    )
    require(
        np.isfinite(outlier_fraction) and 0.0 <= outlier_fraction < 0.5,
        f"outlier_fraction must be in [0, 0.5); got {outlier_fraction!r}",
        value=outlier_fraction,
    )
    require(
        np.isfinite(outlier_magnitude) and outlier_magnitude >= 0.0,
        f"outlier_magnitude must be >= 0 and finite; got {outlier_magnitude!r}",
        value=outlier_magnitude,
    )
    if occlusion is None:
        occluded = np.zeros(0, dtype=int)
    else:
        start, stop = occlusion
        require(
            0 <= start < stop <= n,
            f"occlusion must satisfy 0 <= start < stop <= {n}; got {occlusion!r}",
            value=occlusion,
        )
        occluded = np.arange(start, stop, dtype=int)

    rng = np.random.default_rng(seed)
    observed = q + rng.normal(0.0, noise_std, size=q.shape)

    allowed = np.setdiff1d(np.arange(2, n), occluded)
    count = min(int(round(outlier_fraction * n)), allowed.size)
    outliers = np.sort(rng.choice(allowed, size=count, replace=False)).astype(int)
    joints = rng.integers(0, _DOF, size=count)
    spikes = (
        rng.choice([-1.0, 1.0], size=count)
        * outlier_magnitude
        * rng.uniform(1.0, 1.5, size=count)
    )
    observed[outliers, joints] += spikes

    mask = np.ones(n, dtype=bool)
    mask[occluded] = False
    observed[~mask] = np.nan

    ensure(
        bool(np.array_equal(np.flatnonzero(~mask), occluded)),
        "mask must be False exactly at occluded indices",
    )
    ensure(bool(np.all(np.isfinite(observed[mask]))), "present samples must be finite")
    ensure(
        not np.intersect1d(outliers, occluded).size and not np.any(outliers < 2),
        "outliers must avoid occlusion and the first two samples",
    )
    return CorruptedObservation(
        q_observed=observed,
        mask=mask,
        outlier_indices=outliers,
        occluded_indices=occluded,
        noise_std=float(noise_std),
    )
