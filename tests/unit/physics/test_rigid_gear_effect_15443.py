"""Rigid-contact gear effect for an off-centre strike (#15443).

The legacy gear term is a heuristic (``gear_effect_factor * offset * speed *
scale``) documented as unqualified. The rigid-contact law resolves the
off-centre strike with the shared finite-inertia contact solver instead: the
gear increment is the ball spin of the offset strike minus that of the centred
strike, for a flat face. These tests pin its physical properties.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.physics.impact_model.rigid_gear import (
    face_frame,
    rigid_gear_spin,
)
from src.shared.python.physics.impact_model.solver import ImpactSolverAPI
from src.shared.python.physics.impact_model.types import (
    ImpactModelType,
    ImpactParameters,
    PreImpactState,
    RigidGearHead,
)
from src.shared.python.physics.impact_model.utils import compute_gear_effect_spin

pytestmark = pytest.mark.unit
RPM = 60.0 / (2.0 * math.pi)
DRIVER = RigidGearHead(
    inertia_diagonal_kg_m2=(0.00025, 0.00040, 0.00045), cg_depth_m=0.035
)
# A driver strike measured on a matched swing: lofted, slightly closed face.
NORMAL = np.array([0.9802, -0.0302, 0.1955])
VELOCITY = np.array([51.1063, -0.4876, 4.2059])


def _pre(offset, normal=NORMAL, velocity=VELOCITY) -> PreImpactState:
    return PreImpactState(
        clubhead_velocity=np.asarray(velocity, dtype=float),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=np.asarray(normal, dtype=float) / np.linalg.norm(normal),
        ball_position=np.zeros(3),
        ball_velocity=np.zeros(3),
        ball_angular_velocity=np.zeros(3),
        impact_offset=None if offset is None else np.asarray(offset, dtype=float),
    )


def _gear(offset, head=DRIVER, **kw) -> np.ndarray:
    return rigid_gear_spin(_pre(offset, **kw), head, cor=0.83, friction=0.4)


def test_face_frame_is_right_handed_with_heel_to_the_left_of_the_target():
    normal, heel, up = face_frame(np.array([1.0, 0.0, 0.0]))
    assert heel == pytest.approx([0.0, 1.0, 0.0])
    assert up == pytest.approx([0.0, 0.0, 1.0])
    assert np.cross(normal, heel) == pytest.approx(up)


def test_face_frame_rejects_a_vertical_normal():
    with pytest.raises(ValueError, match="vertical"):
        face_frame(np.array([0.0, 0.0, 1.0]))


def test_a_centred_strike_has_no_gear_spin():
    assert _gear((0.0, 0.0)) == pytest.approx(np.zeros(3), abs=1e-9)


def test_heel_strike_spins_like_the_legacy_sign_convention():
    """Toe adds +Z spin (draw), heel -Z (fade), as the legacy heuristic."""
    heel = _gear((-0.0034, 0.0))
    toe = _gear((0.0034, 0.0))
    legacy = compute_gear_effect_spin(np.array([-0.0034, 0.0]), VELOCITY, NORMAL)
    assert heel[2] < 0.0 and legacy[2] < 0.0
    assert toe[2] > 0.0


def test_square_face_toe_and_heel_gear_are_antisymmetric():
    square = np.array([1.0, 0.0, 0.0])
    along = np.array([50.0, 0.0, 0.0])
    toe = _gear((0.01, 0.0), normal=square, velocity=along)
    heel = _gear((-0.01, 0.0), normal=square, velocity=along)
    assert toe[2] == pytest.approx(-heel[2], rel=1e-6)


def test_no_cg_depth_means_no_side_spin_from_a_toe_strike():
    """Gear effect needs the CG behind the face; a face-plane CG has no lever."""
    flat = RigidGearHead(
        inertia_diagonal_kg_m2=DRIVER.inertia_diagonal_kg_m2, cg_depth_m=0.0
    )
    square = np.array([1.0, 0.0, 0.0])
    along = np.array([50.0, 0.0, 0.0])
    toe = _gear((0.01, 0.0), head=flat, normal=square, velocity=along)
    assert toe[2] == pytest.approx(0.0, abs=1e-6)


def test_a_higher_yaw_inertia_head_has_less_gear_spin():
    stiff = RigidGearHead(
        inertia_diagonal_kg_m2=(0.00025, 0.00040, 0.00090), cg_depth_m=0.035
    )
    assert abs(_gear((-0.0034, 0.0), head=stiff)[2]) < abs(_gear((-0.0034, 0.0))[2])


def test_matched_driver_heel_strike_gear_is_about_half_the_legacy_heuristic():
    """Regression on the shared solver: -143 rpm side spin versus legacy -291 rpm."""
    gear = _gear((-0.0034, 0.0014)) * RPM
    legacy = (
        compute_gear_effect_spin(np.array([-0.0034, 0.0014]), VELOCITY, NORMAL) * RPM
    )
    assert gear[2] == pytest.approx(-143.0, abs=2.0)
    assert legacy[2] == pytest.approx(-291.0, abs=2.0)


def test_head_rejects_unphysical_values():
    with pytest.raises(ValueError):
        RigidGearHead(inertia_diagonal_kg_m2=(0.0, 0.0004, 0.00045), cg_depth_m=0.03)
    with pytest.raises(ValueError):
        RigidGearHead(
            inertia_diagonal_kg_m2=(0.00025, 0.0004, 0.00045), cg_depth_m=-0.01
        )


def test_solver_uses_the_rigid_law_only_when_a_head_is_given():
    """Same base impact; only the added gear term differs between the laws."""
    pre = _pre((-0.0034, 0.0014))
    legacy = ImpactSolverAPI(ImpactModelType.RIGID_BODY, ImpactParameters())
    rigid = ImpactSolverAPI(
        ImpactModelType.RIGID_BODY, ImpactParameters(rigid_gear_head=DRIVER)
    )
    legacy_spin = legacy.solve_pre_impact_state(0.0, pre).ball_angular_velocity
    rigid_spin = rigid.solve_pre_impact_state(0.0, pre).ball_angular_velocity
    legacy_term = compute_gear_effect_spin(
        np.asarray(pre.impact_offset), VELOCITY, pre.clubhead_orientation
    )
    assert rigid_spin - legacy_spin == pytest.approx(
        _gear((-0.0034, 0.0014)) - legacy_term, abs=1e-9
    )
