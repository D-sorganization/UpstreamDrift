"""Contracts and pure math for per-segment angular momentum (#15397)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz.segment_angular_momentum import (
    REFERENCE_POINT,
    UNITS_ANGMOM,
    SegmentAngularMomentumFrame,
    SegmentGroupCatalog,
    body_angular_momentum_about_point,
    build_segment_angular_momentum_frame,
    orbital_fraction,
    read_segment_angular_momentum_frame,
    spin_angular_momentum_world,
)

pytestmark = pytest.mark.unit


def test_spinning_rod_about_com_matches_inertial_spin_term() -> None:
    inertia = np.array([0.1, 2.0, 2.0])
    omega = np.array([0.0, 0.0, 12.0])
    rot = np.eye(3)
    com = np.array([0.3, 0.0, 1.0])
    spin = spin_angular_momentum_world(inertia, rot, omega)
    contrib, spin_only = body_angular_momentum_about_point(
        com_world=com,
        com_vel_world=np.zeros(3),
        ang_vel_world=omega,
        inertia_principal=inertia,
        rot_body_to_world=rot,
        mass=2.0,
        reference_com_world=com,
        reference_com_vel_world=np.zeros(3),
    )
    np.testing.assert_allclose(spin, [0.0, 0.0, 24.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(contrib, spin_only, rtol=0, atol=1e-12)
    np.testing.assert_allclose(contrib, spin, rtol=0, atol=1e-12)


def test_displaced_com_with_relative_velocity_uses_com_offset() -> None:
    """Omitting r_C must change H when r_C, r_i, and v_i−v_C are all nonzero (#15400)."""
    r_c = np.array([0.2, 0.0, 0.0])
    v_c = np.array([1.0, 0.5, 0.0])
    r_i = np.array([0.5, 0.3, 0.1])
    v_i = np.array([2.0, -0.2, 0.4])
    kw = {
        "com_world": r_i,
        "com_vel_world": v_i,
        "ang_vel_world": np.zeros(3),
        "inertia_principal": np.ones(3) * 0.1,
        "rot_body_to_world": np.eye(3),
        "mass": 2.0,
        "reference_com_vel_world": v_c,
    }
    about_com, _ = body_angular_momentum_about_point(
        reference_com_world=r_c,
        **kw,
    )
    wrong_ref, _ = body_angular_momentum_about_point(
        reference_com_world=np.zeros(3),
        **kw,
    )
    np.testing.assert_allclose(
        about_com,
        np.cross(r_i - r_c, 2.0 * (v_i - v_c)),
        rtol=0,
        atol=1e-12,
    )
    assert not np.allclose(about_com, wrong_ref, rtol=0, atol=1e-9)


def test_offset_translating_point_mass_orbital_term() -> None:
    r_c = np.zeros(3)
    v_c = np.array([2.0, 0.0, 0.0])
    r_i = np.array([0.0, 0.5, 0.0])
    v_i = v_c.copy()
    contrib, spin_only = body_angular_momentum_about_point(
        com_world=r_i,
        com_vel_world=v_i,
        ang_vel_world=np.zeros(3),
        inertia_principal=np.zeros(3),
        rot_body_to_world=np.eye(3),
        mass=3.0,
        reference_com_world=r_c,
        reference_com_vel_world=v_c,
    )
    expected = np.cross(r_i - r_c, 3.0 * (v_i - v_c))
    np.testing.assert_allclose(contrib, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(spin_only, 0.0, rtol=0, atol=1e-12)


def test_frame_metadata_and_group_partition() -> None:
    catalog = SegmentGroupCatalog(
        legs=frozenset({"thigh"}),
        pelvis=frozenset({"pelvis"}),
        torso=frozenset({"torso"}),
        arms=frozenset({"arm"}),
        club=frozenset({"club"}),
    )
    vectors = {
        name: np.array([float(i), 0.0, 0.0])
        for i, name in enumerate(("pelvis", "torso", "thigh", "arm", "club"), start=1)
    }
    frame = build_segment_angular_momentum_frame(
        time_s=0.5,
        source="fixture",
        contributions=vectors,
        spin_only={name: np.zeros(3) for name in vectors},
        catalog=catalog,
    )
    assert frame.reference_point == REFERENCE_POINT
    assert frame.frame == "world"
    assert frame.units == UNITS_ANGMOM
    assert frame.groups_nm_s["legs"] == (3.0, 0.0, 0.0)
    assert frame.whole_body_nm_s == (15.0, 0.0, 0.0)
    group_sum = sum(
        np.asarray(frame.groups_nm_s[name])
        for name in catalog.as_mapping()
        if frame.groups_nm_s[name] is not None
    )
    np.testing.assert_allclose(group_sum, frame.whole_body_nm_s, rtol=0, atol=1e-12)


def test_sign_flip_negates_angular_momentum() -> None:
    inertia = np.array([0.2, 0.2, 1.5])
    omega = np.array([0.0, 5.0, 0.0])
    com = np.array([1.0, 0.0, 0.0])
    kw = {
        "com_world": com,
        "com_vel_world": np.zeros(3),
        "inertia_principal": inertia,
        "rot_body_to_world": np.eye(3),
        "mass": 1.0,
        "reference_com_world": com,
        "reference_com_vel_world": np.zeros(3),
    }
    pos, _ = body_angular_momentum_about_point(ang_vel_world=omega, **kw)
    neg, _ = body_angular_momentum_about_point(ang_vel_world=-omega, **kw)
    np.testing.assert_allclose(pos, -neg, rtol=0, atol=1e-12)


def test_orbital_fraction_is_recorded_not_hard_coded() -> None:
    inertia = np.array([0.05, 0.05, 0.8])
    omega = np.array([0.0, 0.0, 8.0])
    com = np.array([0.4, 0.0, 0.0])
    ref = np.zeros(3)
    contrib, spin = body_angular_momentum_about_point(
        com_world=com,
        com_vel_world=np.array([1.5, 0.0, 0.0]),
        ang_vel_world=omega,
        inertia_principal=inertia,
        rot_body_to_world=np.eye(3),
        mass=2.0,
        reference_com_world=ref,
        reference_com_vel_world=np.zeros(3),
    )
    share = orbital_fraction(contrib, spin)
    assert share is not None
    assert 0.0 <= share <= 1.0
    assert share != pytest.approx(0.8, abs=0.05)  # unverified coaching claim


def test_provider_reader_validates_timestamp() -> None:
    class Provider:
        def get_segment_angular_momentum(self):
            return SegmentAngularMomentumFrame(
                time_s=1.0,
                reference_point=REFERENCE_POINT,
                frame="world",
                units=UNITS_ANGMOM,
                source="fixture",
                contribution_nm_s={"a": (1.0, 0.0, 0.0)},
                spin_only_nm_s={"a": (0.5, 0.0, 0.0)},
                groups_nm_s={
                    "legs": None,
                    "pelvis": None,
                    "torso": None,
                    "arms": None,
                    "club": None,
                },
                whole_body_nm_s=(1.0, 0.0, 0.0),
            )

    assert read_segment_angular_momentum_frame(Provider(), 1.0) is not None
    assert read_segment_angular_momentum_frame(Provider(), 2.0) is None
    assert read_segment_angular_momentum_frame(object(), 1.0) is None
