"""Contact-point kinematics and normal effective mass for rigid-body impact."""

from __future__ import annotations

import math

import numpy as np

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
)

from .types import PreImpactState


def unit_face_normal(orientation: np.ndarray) -> np.ndarray:
    raw = np.asarray(orientation, dtype=float).reshape(-1)
    norm = float(math.hypot(*raw)) if raw.size > 0 else 0.0
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return raw / norm


def impact_offset_vector_3d(pre_state: PreImpactState) -> np.ndarray:
    """2D face offset as a CG-to-contact lever arm in world coordinates."""
    if pre_state.impact_offset is None:
        return np.zeros(3)
    offset = np.asarray(pre_state.impact_offset, dtype=float).reshape(-1)
    if offset.size == 0:
        return np.zeros(3)
    n_unit = unit_face_normal(pre_state.clubhead_orientation)
    up = np.array([0.0, 0.0, 1.0])
    horizontal_axis = np.cross(up, n_unit)
    h_norm = float(math.hypot(*horizontal_axis))
    horizontal_axis = (
        np.array([0.0, 1.0, 0.0]) if h_norm <= 1e-12 else horizontal_axis / h_norm
    )
    vertical_axis = np.cross(n_unit, horizontal_axis)
    v_norm = float(math.hypot(*vertical_axis))
    if v_norm > 1e-12:
        vertical_axis = vertical_axis / v_norm
    v_offset = float(offset[1]) if offset.size > 1 else 0.0
    return float(offset[0]) * horizontal_axis + v_offset * vertical_axis


def contact_point_velocity(
    center_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    lever_arm_cg_to_contact: np.ndarray,
) -> np.ndarray:
    """Rigid-body velocity at a contact point: v_c = v_cm + omega x r."""
    v_cm = np.asarray(center_velocity, dtype=float)
    omega = np.asarray(angular_velocity, dtype=float)
    r = np.asarray(lever_arm_cg_to_contact, dtype=float)
    return v_cm + np.cross(omega, r)


def ball_contact_lever_arm(face_normal: np.ndarray) -> np.ndarray:
    """Ball CG-to-surface lever arm along the inward clubface normal."""
    n_unit = unit_face_normal(face_normal)
    return -float(GOLF_BALL_RADIUS_M) * n_unit


def normal_effective_inverse_mass(
    mass: float,
    moi: float,
    lever_arm: np.ndarray,
    face_normal: np.ndarray,
) -> float:
    """Inverse effective mass along face_normal including scalar rotational inertia."""
    inv = 1.0 / float(mass)
    if moi > 0.0:
        n_unit = unit_face_normal(face_normal)
        r = np.asarray(lever_arm, dtype=float)
        rxn = np.cross(r, n_unit)
        inv += float(np.dot(rxn, rxn)) / float(moi)
    return inv


def reduced_normal_effective_mass(
    pre_state: PreImpactState,
    face_normal: np.ndarray,
    *,
    ball_mass: float = GOLF_BALL_MASS_KG,
    ball_moi: float = GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
) -> float:
    """Reduced effective mass for normal impulse (club and ball contact points)."""
    n = unit_face_normal(face_normal)
    r_club = impact_offset_vector_3d(pre_state)
    r_ball = ball_contact_lever_arm(n)
    inv_club = normal_effective_inverse_mass(
        float(pre_state.clubhead_mass),
        float(pre_state.clubhead_moi),
        r_club,
        n,
    )
    inv_ball = normal_effective_inverse_mass(ball_mass, ball_moi, r_ball, n)
    return 1.0 / (inv_club + inv_ball)


def compute_effective_club_mass(pre_state: PreImpactState) -> float:
    """Scalar normal-direction effective mass of the club at the contact point."""
    if pre_state is None:
        raise ValueError("pre_state must be provided")
    m_club = float(pre_state.clubhead_mass)
    n_unit = unit_face_normal(pre_state.clubhead_orientation)
    inv = normal_effective_inverse_mass(
        m_club,
        float(pre_state.clubhead_moi),
        impact_offset_vector_3d(pre_state),
        n_unit,
    )
    return 1.0 / inv


def pre_impact_contact_relative_velocity(pre_state: PreImpactState) -> np.ndarray:
    """Club-minus-ball relative velocity at the mutual contact point."""
    n_unit = unit_face_normal(pre_state.clubhead_orientation)
    r_club = impact_offset_vector_3d(pre_state)
    r_ball = ball_contact_lever_arm(n_unit)
    v_club = contact_point_velocity(
        pre_state.clubhead_velocity,
        pre_state.clubhead_angular_velocity,
        r_club,
    )
    v_ball = contact_point_velocity(
        pre_state.ball_velocity,
        pre_state.ball_angular_velocity,
        r_ball,
    )
    return v_club - v_ball
