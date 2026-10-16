"""Signal-dependent muscle noise pushed to the joint torque (NSX-94, #12723).

Mechanism M1 of the variability branch, applied where it physically lives
(hypothesis ``H-NOISE-SIGNAL-DEPENDENT``): the force noise of muscle ``k`` grows
with its command,

    S_k(a) = sigma_0,k^2 + (kappa_k F_max,k a_k)^2,

the declared law of :class:`DisturbanceSpec` (``covariance + diag((gain u)^2)``)
with ``gain_k = kappa_k F_max,k`` and the signal ``u = a`` the activation.  The
issue's ``sigma proportional to a F_max`` is this law with ``sigma_0 = 0``.

Units and convention.  ``S_k`` is a two-sided spectral-density level in
``N^2 s`` (the :class:`DisturbanceSpec` convention), so ``kappa`` has units
``sqrt(s)``: the noise amplitude per unit of ``a F_max`` per root hertz.  The
per-step covariance of a white draw is ``S_k / dt`` (V6: amplitudes are
timestep-consistent).  The injection ``stage`` is ``ACTIVATION``: the noise
enters at the muscle command, upstream of the moment-arm map, and its unit is
muscle force in newtons.

Pushing to the joint: ``tau = R^T f`` is linear, so the joint-torque covariance
of independent muscle noises is ``R^T diag(S) R`` (``n x n``).  Muscles acting
on the same coordinate with the same sign add variance; the cross terms between
agonist and antagonist vanish only because their noises are independent.  This
is why raising antagonist activation at fixed net torque cannot lower the
muscle-noise variance (it is a sum of non-negative terms increasing in each
``a_k``).

Non-claims: ``kappa`` and ``sigma_0`` are declared hypotheses; no value is
measured on a golfer and none is adopted from the literature here.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.muscle_frame import MuscleFrame
from src.shared.python.perturbation.disturbance import (
    DisturbanceSpec,
    InjectionStage,
    Spectrum,
)

__all__ = ["joint_torque_covariance", "muscle_noise_spec"]


def muscle_noise_spec(
    frame: MuscleFrame,
    *,
    kappa: float | ArrayLike,
    sigma0: float | ArrayLike = 0.0,
    name: str = "muscle-force-noise",
) -> DisturbanceSpec:
    """Declared signal-dependent white force noise, one channel per muscle.

    ``kappa`` (``sqrt(s)``) and ``sigma0`` (``N sqrt(s)``) are scalars or
    per-muscle arrays.  Preconditions (``ValueError``): ``kappa`` finite and
    ``>= 0``, ``sigma0`` finite and ``>= 0``.  Postcondition: the spec's signal
    law is active with bounds ``[0, 1]`` per channel.
    """
    m = frame.n_muscles
    kap = np.broadcast_to(np.asarray(kappa, dtype=np.float64), (m,)).copy()
    sig = np.broadcast_to(np.asarray(sigma0, dtype=np.float64), (m,)).copy()
    if not (np.all(np.isfinite(kap)) and np.all(kap >= 0.0)):
        raise ValueError("kappa must be finite and non-negative")
    if not (np.all(np.isfinite(sig)) and np.all(sig >= 0.0)):
        raise ValueError("sigma0 must be finite and non-negative")
    return DisturbanceSpec(
        name=name,
        stage=InjectionStage.ACTIVATION,
        channels=frame.names,
        units="N",
        covariance=np.diag(sig**2),
        spectrum=Spectrum.WHITE,
        signal_gain=tuple(float(g) for g in kap * frame.max_isometric_force),
        signal_bounds=((0.0, 1.0),) * m,
    )


def joint_torque_covariance(
    frame: MuscleFrame,
    spec: DisturbanceSpec,
    activation: ArrayLike,
    *,
    dt: float,
) -> Array:
    """Per-step joint-torque covariance ``R^T diag(S(a) / dt) R``, ``(n, n)`` N^2 m^2.

    Preconditions (``ValueError``): ``spec.channels == frame.names`` (a spec
    built for another frame is refused), ``0 <= activation <= 1`` (enforced by
    the spec's signal bounds), ``dt > 0``.  Postcondition: symmetric positive
    semi-definite, non-decreasing in every activation, equal to ``0`` at
    ``a = 0`` when ``sigma0 = 0``.
    """
    if spec.channels != frame.names:
        raise ValueError(
            f"spec channels {spec.channels} do not match the frame's muscles {frame.names}"
        )
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError(f"dt must be positive and finite, got {dt}")
    a = np.asarray(activation, dtype=np.float64)
    cov = spec.discrete_covariance(dt, a)
    out = frame.moment_arms.T @ cov @ frame.moment_arms
    return np.asarray(0.5 * (out + out.T))
