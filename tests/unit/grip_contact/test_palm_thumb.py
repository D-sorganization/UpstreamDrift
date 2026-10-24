"""Allocated palm-on-thumb hand-to-hand force (HRG-H1, #13189). Reporting only."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_contact.palm_thumb import (
    METHOD,
    PalmThumbForce,
    allocate_palm_thumb,
    unavailable_palm_thumb,
)

pytestmark = pytest.mark.unit
LAMBDA_TOL = 1e-10
THIRD_LAW_TOL = 1e-9
G = 9.80665
POINT = np.array([0.30, -0.10, 0.20])
SITE = np.array([0.29, -0.12, 0.18])
NORMAL = np.array([0.0, 1.0, 0.0])  # thumb -> palm
MU = 0.9


def _alloc(force, torque, normal=NORMAL, mu=MU):
    return allocate_palm_thumb(
        force_on_lead_n=force,
        torque_on_lead_nm=torque,
        reference_point_m=SITE,
        patch_point_m=POINT,
        patch_normal=normal,
        static_friction=mu,
    )


def test_method_label_is_allocation() -> None:
    out = _alloc([0.0, -100.0, 0.0], [1.0, 2.0, 3.0])
    assert METHOD == "allocation" and out.method == "allocation"


def test_normal_is_the_compressive_part_and_tangential_the_rest() -> None:
    # The palm pushes the thumb along -normal; add a sliding component.
    out = _alloc([30.0, -100.0, 40.0], [0.0, 0.0, 0.0])
    assert out.available and not out.unloaded
    assert out.normal_force_n == pytest.approx(100.0)
    assert out.tangential_force_n == pytest.approx(50.0)
    assert out.friction_ratio == pytest.approx(50.0 / (MU * 100.0))
    assert out.share_of_closure_force == pytest.approx(100.0 / np.hypot(50, 100))


def test_tension_is_flagged_unloaded_not_reported() -> None:
    out = _alloc([0.0, 80.0, 5.0], [0.0, 0.0, 0.0])
    assert out.unloaded and out.normal_force_n == 0.0
    assert out.friction_ratio is None and "unloaded" in out.friction_reason


def test_patch_force_and_couple_reproduce_the_closure_wrench() -> None:
    rng = np.random.default_rng(13189)
    for _ in range(20):
        f, m = rng.normal(0, 50, 3), rng.normal(0, 5, 3)
        out = _alloc(f, m)
        back = np.array(out.force_on_lead_n)
        moment_back = np.array(out.couple_nm) + np.cross(POINT - SITE, back)
        assert np.abs(back - f).max() <= LAMBDA_TOL
        assert np.abs(moment_back - m).max() <= LAMBDA_TOL


def test_newtons_third_law_between_trail_hand_and_lead_thumb() -> None:
    out = _alloc([3.0, -40.0, 7.0], [0.5, -0.2, 0.9])
    assert (
        np.abs(np.add(out.force_on_lead_n, out.force_on_trail_n)).max() <= THIRD_LAW_TOL
    )
    assert np.abs(np.add(out.couple_nm, out.couple_on_trail_nm)).max() <= THIRD_LAW_TOL


def test_unavailable_is_none_with_a_reason_never_zero() -> None:
    out = unavailable_palm_thumb("no efforts")
    assert not out.available and out.reason == "no efforts"
    for name in (
        "normal_force_n",
        "tangential_force_n",
        "friction_ratio",
        "share_of_closure_force",
        "force_on_lead_n",
        "couple_nm",
    ):
        assert getattr(out, name) is None, name
    assert out.unloaded is None
    with pytest.raises(ValueError, match="reason"):
        unavailable_palm_thumb("  ")


@pytest.mark.parametrize("k", range(12))
def test_static_two_body_hold_at_twelve_attitudes(k: int) -> None:
    """A club of mass m held at rest: the allocated force carries its weight."""
    rng = np.random.default_rng(k)
    mass = 0.45 + 0.01 * k
    weight = mass * G
    com = rng.normal(0.0, 0.3, 3)
    # Equilibrium of the club: the hand force is m g up and its couple about the
    # closure site is (com - site) x F, so the net moment about the COM vanishes.
    force = np.array([0.0, 0.0, weight])
    torque_site = np.cross(com - SITE, force)
    attitude = _rotation(k)
    normal = attitude @ NORMAL
    out = _alloc(force, torque_site, normal=normal)
    # Force residual
    assert np.abs(np.array(out.force_on_lead_n) - force).max() <= 1e-6 * weight
    # Moment residual about the COM: F at the patch plus the kept couple.
    about_com = np.cross(POINT - com, out.force_on_lead_n) + np.array(out.couple_nm)
    assert np.abs(about_com).max() <= 1e-6 * weight
    assert out.unloaded or out.normal_force_n >= 0.0


def _rotation(k: int) -> np.ndarray:
    a = 2.0 * np.pi * k / 12.0
    c, s = np.cos(a), np.sin(a)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    b = 0.4 * (k % 5)
    cb, sb = np.cos(b), np.sin(b)
    rx = np.array([[1.0, 0, 0], [0, cb, -sb], [0, sb, cb]])
    return rz @ rx


def test_preconditions_are_checked() -> None:
    ok = ([0.0, -1.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="unit"):
        _alloc(*ok, normal=np.array([0.0, 2.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        _alloc([np.nan, 0.0, 0.0], ok[1])
    with pytest.raises(ValueError, match="static_friction"):
        _alloc(*ok, mu=-0.1)
    with pytest.raises(ValueError, match="shape"):
        _alloc([1.0, 2.0], ok[1])
    assert isinstance(_alloc(*ok), PalmThumbForce)
