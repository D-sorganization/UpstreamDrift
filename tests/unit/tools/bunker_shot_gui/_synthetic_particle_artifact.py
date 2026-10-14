"""SYNTHETIC TEST FIXTURE -- a particle trajectory no solver produced.

Issue #9545 (folded-in #9693). No production backend writes the slice-6
particle-trajectory artifact yet: the only grain-resolving 3-D tier
(Chrono, F3) needs ``pychrono``, which is not a declared dependency, and
the MPM tier is 2-D plane strain with no grain identity. The viewer is
therefore tested against this hand-built artifact, written through the
real :class:`~bunkershot3d.fields.trajectory.ParticleTrajectoryWriter` so
the reader verifies it exactly as it would a solver's.

Nothing here is physics. The motions are straight lines chosen to make
identity, lateral asymmetry and loss observable; the provenance names the
fixture so a picture rendered from it cannot be mistaken for a run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bunkershot3d.fields.schema import GrainRoster, ParticleKind
from bunkershot3d.fields.standing import FieldProvenance
from bunkershot3d.fields.trajectory import ContactEvent, ParticleTrajectoryWriter
from bunkershot3d.solvers import EnvelopeStatus, FidelityTier

SYNTHETIC_SOLVER_NAME = "SYNTHETIC TEST FIXTURE (tests/unit/tools/bunker_shot_gui)"

#: Deliberately unsorted, so an index-for-id mix-up cannot pass by luck.
GRAIN_IDS = np.array([31, 2, 47, 8, 15, 23], dtype=np.int64)
RADII_M = np.array([2.0e-4, 2.4e-4, 2.2e-4, 2.6e-4, 2.1e-4, 2.5e-4])

#: Irregular retained cadence, so frame index and time are not proportional.
FRAME_TIMES_S = (0.0, 1.0e-4, 2.5e-4, 3.0e-4, 5.0e-4)

_START_M = np.array(
    [
        [-1.0e-3, 0.0, 2.0e-3],
        [1.0e-3, 0.0, 2.0e-3],
        [0.0, -2.0e-3, 1.0e-3],
        [0.0, 2.0e-3, 1.0e-3],
        [3.0e-3, 1.0e-3, 0.5e-3],
        [-3.0e-3, -1.0e-3, 0.5e-3],
    ]
)
#: Grains 31 and 2 run head-on along x and cross between frames 2 and 3;
#: every grain has its own lateral (y) velocity, so the fixture is not a
#: plane repeated across the width.
_VELOCITY_M_S = np.array(
    [
        [4.0, 0.5, 0.0],
        [-4.0, -1.5, 0.0],
        [0.0, 3.0, 1.0],
        [0.0, -2.0, 0.5],
        [-1.0, 0.0, 2.0],
        [1.0, 4.0, 0.0],
    ]
)

#: Grain 47 (row 2) enters at frame 1; grain 23 (row 5) escapes at frame 3.
_ENTERS_AT = {2: 1}
_LOST_FROM = {5: 3}


def grain_position(row: int, time_s: float) -> np.ndarray:
    """Where the fixture puts the grain in ``row`` at ``time_s``."""
    return _START_M[row] + _VELOCITY_M_S[row] * time_s


def _masks(frame: int) -> tuple[np.ndarray, np.ndarray]:
    active = np.ones(GRAIN_IDS.size, dtype=bool)
    lost = np.zeros(GRAIN_IDS.size, dtype=bool)
    for row, first in _ENTERS_AT.items():
        active[row] = frame >= first
    for row, first in _LOST_FROM.items():
        lost[row] = frame >= first
        active[row] = active[row] and not lost[row]
    return active, lost


def synthetic_provenance() -> FieldProvenance:
    """Provenance that names the fixture rather than a solver."""
    return FieldProvenance(
        fidelity_tier=FidelityTier.F3,
        envelope_status=EnvelopeStatus.BEYOND_VALIDATION,
        solver_name=SYNTHETIC_SOLVER_NAME,
        kinematics="synthetic straight-line test motion",
        peak_speed_m_s=4.0,
        reasons=("synthetic test fixture; not a solver run",),
    )


def write_synthetic_artifact(
    path: Path, *, kind: ParticleKind = ParticleKind.DISCRETE_GRAIN
) -> Path:
    """Write and finalise the synthetic fixture at ``path``."""
    roster = GrainRoster(grain_ids=GRAIN_IDS, radius_m=RADII_M, kind=kind)
    with ParticleTrajectoryWriter(path, roster, synthetic_provenance()) as writer:
        for frame, time_s in enumerate(FRAME_TIMES_S):
            positions = np.array(
                [grain_position(row, time_s) for row in range(GRAIN_IDS.size)]
            )
            active, lost = _masks(frame)
            writer.append_frame(time_s, positions, _VELOCITY_M_S, active, lost)
        writer.append_contact(
            ContactEvent(
                time_s=FRAME_TIMES_S[2],
                grain_id=31,
                body_id=0,
                point_m=grain_position(0, FRAME_TIMES_S[2]),
                force_n=np.full(3, np.nan),
            )
        )
    return path
