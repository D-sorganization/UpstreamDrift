"""Shared 16-segment :class:`SubjectAnthropometrics` fixture for adapters."""

from __future__ import annotations

import numpy as np
import pytest

from anthropometrics import SegmentProperties, SubjectAnthropometrics

# The 16 canonical body segments — head, torso, pelvis, plus left/right
# upper-arm, forearm, hand, thigh, shank, foot.
_SEGMENT_DEFS: list[dict[str, object]] = [
    {"name": "head", "body_part_id": "head", "length_m": 0.20, "mass_kg": 4.5},
    {"name": "torso", "body_part_id": "torso", "length_m": 0.50, "mass_kg": 28.0},
    {"name": "pelvis", "body_part_id": "pelvis", "length_m": 0.15, "mass_kg": 8.5},
    {
        "name": "upper_arm_left",
        "body_part_id": "upper_arm_left",
        "length_m": 0.30,
        "mass_kg": 2.1,
    },
    {
        "name": "upper_arm_right",
        "body_part_id": "upper_arm_right",
        "length_m": 0.30,
        "mass_kg": 2.1,
    },
    {
        "name": "forearm_left",
        "body_part_id": "forearm_left",
        "length_m": 0.27,
        "mass_kg": 1.3,
    },
    {
        "name": "forearm_right",
        "body_part_id": "forearm_right",
        "length_m": 0.27,
        "mass_kg": 1.3,
    },
    {
        "name": "hand_left",
        "body_part_id": "hand_left",
        "length_m": 0.18,
        "mass_kg": 0.45,
    },
    {
        "name": "hand_right",
        "body_part_id": "hand_right",
        "length_m": 0.18,
        "mass_kg": 0.45,
    },
    {
        "name": "thigh_left",
        "body_part_id": "thigh_left",
        "length_m": 0.42,
        "mass_kg": 8.6,
    },
    {
        "name": "thigh_right",
        "body_part_id": "thigh_right",
        "length_m": 0.42,
        "mass_kg": 8.6,
    },
    {
        "name": "shank_left",
        "body_part_id": "shank_left",
        "length_m": 0.43,
        "mass_kg": 3.4,
    },
    {
        "name": "shank_right",
        "body_part_id": "shank_right",
        "length_m": 0.43,
        "mass_kg": 3.4,
    },
    {
        "name": "foot_left",
        "body_part_id": "foot_left",
        "length_m": 0.25,
        "mass_kg": 1.0,
    },
    {
        "name": "foot_right",
        "body_part_id": "foot_right",
        "length_m": 0.25,
        "mass_kg": 1.0,
    },
    {
        "name": "neck",
        "body_part_id": "neck",
        "length_m": 0.10,
        "mass_kg": 1.2,
    },
]


def _build_segment(idx: int, defn: dict[str, object]) -> SegmentProperties:
    """Construct a physically-realisable :class:`SegmentProperties`."""
    length_m = float(defn["length_m"])  # type: ignore[arg-type]
    mass_kg = float(defn["mass_kg"])  # type: ignore[arg-type]
    # Use a small but distinct off-diagonal pattern per segment so the
    # adapters are exercised on fully populated tensors. The diagonal
    # dominates so the triangle inequality and PD invariants hold.
    diag = mass_kg * length_m**2 / 12.0
    # Slightly perturb each axis so principal moments differ.
    ixx = diag * (1.0 + 0.10 * (idx % 3))
    iyy = diag * (1.0 + 0.07 * ((idx + 1) % 3))
    izz = diag * (1.0 + 0.03 * ((idx + 2) % 3))
    off = diag * 0.01
    tensor = np.array(
        [
            [ixx, off, -off * 0.5],
            [off, iyy, off * 0.3],
            [-off * 0.5, off * 0.3, izz],
        ],
        dtype=float,
    )
    com = np.array([0.001 * idx, -0.002 * (idx % 4), 0.4 * length_m], dtype=float)
    return SegmentProperties(
        name=str(defn["name"]),
        body_part_id=str(defn["body_part_id"]),
        length_m=length_m,
        proximal_marker=f"{defn['name']}_prox" if idx % 2 == 0 else None,
        distal_marker=f"{defn['name']}_dist" if idx % 3 == 0 else None,
        mass_kg=mass_kg,
        com_xyz_m=com,
        inertia_tensor=tensor,
        source_method="de_leva_1996",
        source_subject_height_m=1.80,
        source_subject_mass_kg=75.0,
    )


@pytest.fixture
def sixteen_segment_subject() -> SubjectAnthropometrics:
    """A representative 16-segment :class:`SubjectAnthropometrics`."""
    segments = tuple(
        (str(defn["name"]), _build_segment(i, defn))
        for i, defn in enumerate(_SEGMENT_DEFS)
    )
    return SubjectAnthropometrics(
        subject_id="test_subject_001",
        height_m=1.80,
        mass_kg=75.0,
        segments=segments,
        source_method="de_leva_1996",
        age_years=32.5,
        sex="M",
    )
