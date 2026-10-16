"""Shared muscle-level fixtures for the NSX-94 tests (#12723).

* :func:`elbow_frame` -- the NSX-73 three-muscle Hill elbow
  (``create_elbow_muscle_system``): constant moment arms ``[0.04, 0.03, -0.035]``
  m, isometric forces per unit activation ``[1000, 800, 1200]`` N, zero passive
  force at optimal fibre length.  Rank one, nullity two.
* :func:`pair_frame` -- an ideal antagonistic pair with equal ``F_max`` and
  moment arms ``+r`` / ``-r``.  Its co-contraction sub-fibre is a single point,
  which makes every closed form below exact.
* :class:`OneJoint` -- a one-coordinate rigid body (inertia ``I``, viscous
  damping ``B``) in the model-function view the NSX-84 routine differences.

The numbers are invented fixture values that demonstrate a mechanism; none is a
physiological or golfer measurement.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.biomechanics.hill_muscle import MuscleState
from src.shared.python.biomechanics.multi_muscle import create_elbow_muscle_system
from src.shared.python.loop_nullspace.evidence import InputProvenance, ModelProvenance
from src.shared.python.loop_nullspace.muscle_frame import MuscleFrame, TendonKind

MODEL = ModelProvenance("muscle-fixture", "1")
INPUT = InputProvenance("fixture", "nsx94-tests")

PAIR_ARM = 0.04  # m
PAIR_FORCE = 1000.0  # N per unit activation
PAIR_RHO = 50.0  # 1/m, declared force-proportional SRS slope (fixture value)
PAIR_KAPPA = 0.05  # sqrt(s), declared signal-dependent noise slope
EXTERNAL_PSD = 1.0  # N^2 m^2 s
INERTIA = 0.1  # kg m^2
DAMPING = 4.0  # N m s / rad

_STATES = {"biceps": (0.15, 0.0), "brachialis": (0.12, 0.0), "triceps": (0.18, 0.0)}


def elbow_frame() -> MuscleFrame:
    system = create_elbow_muscle_system(enable_rust=False)
    names = tuple(system.muscle_names)
    muscles = {**system.agonist.muscles, **system.antagonist.muscles}
    arms = {**system.agonist.attachments, **system.antagonist.attachments}
    f_unit = np.array(
        [
            muscles[n].compute_force(
                MuscleState(activation=1.0, l_CE=_STATES[n][0], v_CE=0.0, l_MT=0.0)
            )
            for n in names
        ]
    )
    r = np.array([[arms[n].moment_arm] for n in names])
    f_max = np.array([muscles[n].params.F_max for n in names])
    return MuscleFrame(
        names=names,
        coordinates=("elbow_flexion",),
        moment_arms=r,
        activation_grid=np.array([0.0, 1.0]),
        force_grid=np.array([np.zeros(3), f_unit]),
        max_isometric_force=f_max,
        tendon=TendonKind.RIGID,
        engine="fixture",
        model_provenance=MODEL,
        input_provenance=INPUT,
    )


def pair_frame() -> MuscleFrame:
    return MuscleFrame(
        names=("flexor", "extensor"),
        coordinates=("joint",),
        moment_arms=np.array([[PAIR_ARM], [-PAIR_ARM]]),
        activation_grid=np.array([0.0, 1.0]),
        force_grid=np.array([[0.0, 0.0], [PAIR_FORCE, PAIR_FORCE]]),
        max_isometric_force=np.array([PAIR_FORCE, PAIR_FORCE]),
        tendon=TendonKind.RIGID,
        engine="fixture",
        model_provenance=MODEL,
        input_provenance=INPUT,
    )


class OneJoint:
    """``M = I``, ``h = B v``; no closure rows, so ``constrained = free``."""

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return np.array([[INERTIA]])

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.array([DAMPING * float(v[0])])

    def closure_jacobian(self, q: np.ndarray) -> np.ndarray:
        return np.zeros((0, 1))

    def closure_drift(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.zeros(0)
