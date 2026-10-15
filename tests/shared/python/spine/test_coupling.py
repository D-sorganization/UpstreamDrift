"""Tests for spine coupling laws (SPN-2a, issue #12741)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.myofullbody.couplings import JointCoupling
from src.shared.python.spine.coupling import (
    CouplingLaw,
    LinearCouplingLaw,
    StateDependentCouplingLaw,
    from_joint_coupling,
    level_rotations,
    regional_fraction,
)
from src.shared.python.spine.levels import SpinalLevel, levels_between
from src.shared.python.spine.segmental import SegmentalPose

pytestmark = pytest.mark.unit

CITE = "Invented fixture law, doi:10.0000/test table 1"
AXIAL = 2
ALL = levels_between(SpinalLevel.L5_S1, SpinalLevel.T12_L1)  # 6 levels
LUMBAR = levels_between(SpinalLevel.L5_S1, SpinalLevel.L1_L2)  # 5 levels


def _axial_law() -> LinearCouplingLaw:
    """Diagonal law; the 5 lumbar AR gains sum to 0.17, T12_L1 holds 0.83."""
    gains = {lv: 0.034 for lv in LUMBAR}
    gains[SpinalLevel.T12_L1] = 0.83
    mats = {}
    for lv, g in gains.items():
        m = np.zeros((3, 1))
        m[AXIAL, 0] = g
        mats[lv] = m
    return LinearCouplingLaw("axial", "1", mats, ("trunk_axial",), CITE)


def _cross_law() -> LinearCouplingLaw:
    """Axial rotation induces lateral bending; sign flips at L5_S1."""
    mats = {}
    for lv in levels_between(SpinalLevel.L5_S1, SpinalLevel.L4_L5):
        m = np.zeros((3, 2))
        m[2, 0] = 0.1
        m[1, 0] = -0.2 if lv is SpinalLevel.L5_S1 else 0.2
        m[0, 1] = 0.5
        mats[lv] = m
    return LinearCouplingLaw("cross", "2", mats, ("axial", "flex"), CITE)


def _state_law() -> StateDependentCouplingLaw:
    bps = np.array([0.0, 0.5, 1.0])
    lv = levels_between(SpinalLevel.L5_S1, SpinalLevel.L4_L5)
    mats = {}
    for k, level in enumerate(lv):
        arr = np.zeros((3, 3, 2))
        arr[:, 0, 0] = [1.0 + k, 0.1, 0.0]
        arr[:, 1, 0] = [2.0 + k, 0.2, 0.3]
        arr[:, 2, 0] = [2.5 + k, -0.4, 0.3]
        arr[:, :, 1] = np.array([[0.0, 0.3, 0.1]] * 3).T
        mats[level] = arr
    return StateDependentCouplingLaw(
        "rhythm", "1", ("flex", "twist"), "flex", bps, mats, CITE
    )


def _fd(law: CouplingLaw, phi: np.ndarray, h: float = 1e-6) -> dict:
    out = {lv: np.zeros((3, phi.size)) for lv in law.expand(phi)}
    for j in range(phi.size):
        e = np.zeros(phi.size)
        e[j] = h
        plus, minus = law.expand(phi + e), law.expand(phi - e)
        for lv in out:
            out[lv][:, j] = (plus[lv] - minus[lv]) / (2 * h)
    return out


def test_regional_fraction_lumbar_axial() -> None:
    law = _axial_law()
    frac = regional_fraction(law, LUMBAR, AXIAL, np.array([0.3]))
    assert frac == pytest.approx(0.17, abs=1e-12)


def test_regional_fraction_zero_denominator() -> None:
    with pytest.raises(ValueError, match="denominator"):
        regional_fraction(_axial_law(), LUMBAR, AXIAL, np.array([0.0]))


def test_regional_fraction_preconditions() -> None:
    law = _axial_law()
    with pytest.raises(ValueError, match="axis"):
        regional_fraction(law, LUMBAR, 3, np.array([0.3]))
    with pytest.raises(ValueError, match="not in law"):
        regional_fraction(law, [SpinalLevel.T1_T2], AXIAL, np.array([0.3]))
    with pytest.raises(ValueError, match="empty"):
        regional_fraction(law, [], AXIAL, np.array([0.3]))


def test_cross_axis_sign_flip() -> None:
    law = _cross_law()
    out = law.expand(np.array([0.2, 0.1]))
    np.testing.assert_allclose(out[SpinalLevel.L5_S1], [0.05, 0.1 * 2 * 0.1 * 0 - 0.04, 0.02])
    np.testing.assert_allclose(out[SpinalLevel.L4_L5], [0.05, 0.04, 0.02])
    assert out[SpinalLevel.L5_S1][1] == -out[SpinalLevel.L4_L5][1]


def test_linear_jacobian_matches_finite_difference() -> None:
    law = _cross_law()
    phi = np.array([0.3, -0.2])
    jac = law.jacobian(phi)
    fd = _fd(law, phi)
    for lv in jac:
        np.testing.assert_allclose(jac[lv], fd[lv], atol=1e-7)


@pytest.mark.parametrize("flex", [0.1, 0.3, 0.7, 0.9, 1.4, -0.3])
def test_state_dependent_jacobian_matches_finite_difference(flex: float) -> None:
    law = _state_law()
    phi = np.array([flex, 0.4])
    jac = law.jacobian(phi)
    fd = _fd(law, phi)
    for lv in jac:
        assert jac[lv].shape == (3, 2)
        np.testing.assert_allclose(jac[lv], fd[lv], atol=1e-7)


def test_state_dependent_interpolates() -> None:
    law = _state_law()
    out = law.expand(np.array([0.25, 0.0]))
    # C(0.25) = mean of first two breakpoint columns, times flex 0.25
    np.testing.assert_allclose(
        out[SpinalLevel.L5_S1], 0.25 * np.array([1.5, 0.15, 0.15])
    )


def test_protocol_members() -> None:
    for law in (_axial_law(), _state_law()):
        assert law.law_id and law.version
        assert isinstance(law.independent_names(), tuple)


def test_level_rotations_builds_pose() -> None:
    law = _cross_law()
    phi = np.array([0.2, 0.1])
    pose = level_rotations(law, phi)
    assert isinstance(pose, SegmentalPose)
    assert pose.levels == (SpinalLevel.L5_S1, SpinalLevel.L4_L5)
    expected = law.expand(phi)
    for tr in pose.transforms:
        np.testing.assert_allclose(tr.isb_angles(), expected[tr.level], atol=1e-12)


def test_level_rotations_neutral() -> None:
    law = _cross_law()
    neutral = np.eye(4)
    neutral[:3, 3] = [0.0, 0.03, 0.0]
    pose = level_rotations(law, np.array([0.0, 0.0]), neutral={SpinalLevel.L5_S1: neutral})
    np.testing.assert_allclose(pose.transforms[0].neutral, neutral)
    np.testing.assert_allclose(pose.transforms[1].neutral, np.eye(4))
    with pytest.raises(ValueError, match="neutral"):
        level_rotations(law, np.array([0.0, 0.0]), neutral={SpinalLevel.T1_T2: neutral})


def test_phi_preconditions() -> None:
    law = _cross_law()
    with pytest.raises(ValueError, match="shape"):
        law.expand(np.array([0.1]))
    with pytest.raises(ValueError, match="finite"):
        law.expand(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="shape"):
        law.jacobian(np.zeros(3))


def test_linear_construction_preconditions() -> None:
    good = {SpinalLevel.L5_S1: np.zeros((3, 1)), SpinalLevel.L4_L5: np.zeros((3, 1))}

    def mk(**kw):
        args = dict(
            law_id="x", version="1", matrices=good, independent=("a",), citation=CITE
        )
        args.update(kw)
        return LinearCouplingLaw(**args)

    mk()
    with pytest.raises(ValueError, match="shape"):
        mk(matrices={SpinalLevel.L5_S1: np.zeros((3, 2))})
    with pytest.raises(ValueError, match="shape"):
        mk(matrices={SpinalLevel.L5_S1: np.zeros((2, 1))})
    bad = np.zeros((3, 1))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        mk(matrices={SpinalLevel.L5_S1: bad})
    with pytest.raises(ValueError, match="citation"):
        mk(citation="  ")
    with pytest.raises(TypeError, match="citation"):
        mk(citation=None)
    with pytest.raises(TypeError, match="SpinalLevel"):
        mk(matrices={"L5_S1": np.zeros((3, 1))})
    with pytest.raises(ValueError, match="contiguous"):
        mk(matrices={SpinalLevel.L5_S1: np.zeros((3, 1)), SpinalLevel.L3_L4: np.zeros((3, 1))})
    with pytest.raises(ValueError, match="law_id"):
        mk(law_id="")
    with pytest.raises(ValueError, match="independent"):
        mk(independent=())
    with pytest.raises(ValueError, match="unique"):
        mk(matrices={SpinalLevel.L5_S1: np.zeros((3, 2))}, independent=("a", "a"))
    with pytest.raises(ValueError, match="at least one"):
        mk(matrices={})


def test_linear_matrices_read_only_and_copied() -> None:
    src = np.zeros((3, 1))
    law = LinearCouplingLaw("x", "1", {SpinalLevel.L5_S1: src}, ("a",), CITE)
    src[0, 0] = 5.0
    assert law.matrices[SpinalLevel.L5_S1][0, 0] == 0.0
    with pytest.raises(ValueError):
        law.matrices[SpinalLevel.L5_S1][0, 0] = 1.0


def test_state_dependent_preconditions() -> None:
    lv = {SpinalLevel.L5_S1: np.zeros((3, 3, 2))}

    def mk(**kw):
        args = dict(
            law_id="r",
            version="1",
            independent=("flex", "twist"),
            scheduling="flex",
            breakpoints=np.array([0.0, 0.5, 1.0]),
            matrices=lv,
            citation=CITE,
        )
        args.update(kw)
        return StateDependentCouplingLaw(**args)

    mk()
    with pytest.raises(ValueError, match="scheduling"):
        mk(scheduling="nope")
    with pytest.raises(ValueError, match="increasing"):
        mk(breakpoints=np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="breakpoints"):
        mk(breakpoints=np.array([0.0]))
    with pytest.raises(ValueError, match="shape"):
        mk(matrices={SpinalLevel.L5_S1: np.zeros((2, 3, 2))})
    with pytest.raises(ValueError, match="citation"):
        mk(citation="")
    with pytest.raises(ValueError, match="contiguous"):
        mk(matrices={**lv, SpinalLevel.L3_L4: np.zeros((3, 3, 2))})


# ---- from_joint_coupling ----------------------------------------------------


def _jc() -> JointCoupling:
    """q0 independent; q1, q2 dependent on q0 (poly); q3 dependent on q1."""
    poly = np.array(
        [
            [0.1, 2.0, 0.5, 0.0, 0.0],
            [-0.05, -1.0, 0.0, 0.3, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    return JointCoupling(
        qpos0=np.array([0.05, 0.02, -0.01, 0.0]),
        dependent=(1, 2, 3),
        driver=(0, 0, 1),
        poly=poly,
    )


def test_from_joint_coupling_reproduces_expand() -> None:
    jc = _jc()
    level_map = {
        0: "flex",
        1: (SpinalLevel.L5_S1, 0),
        2: (SpinalLevel.L4_L5, 1),
    }
    law = from_joint_coupling(jc, level_map)
    assert law.independent_names() == ("flex",)
    for x in (-0.4, 0.0, 0.3, 0.9):
        q = jc.expand(np.array([x, 0.0, 0.0, 0.0]))
        out = law.expand(np.array([x]))
        assert out[SpinalLevel.L5_S1][0] == pytest.approx(q[1], abs=1e-12)
        assert out[SpinalLevel.L4_L5][1] == pytest.approx(q[2], abs=1e-12)
        assert out[SpinalLevel.L5_S1][1] == 0.0


def test_from_joint_coupling_jacobian() -> None:
    law = from_joint_coupling(
        _jc(), {0: "flex", 1: (SpinalLevel.L5_S1, 0), 2: (SpinalLevel.L4_L5, 1)}
    )
    phi = np.array([0.35])
    jac = law.jacobian(phi)
    fd = _fd(law, phi)
    for lv in jac:
        np.testing.assert_allclose(jac[lv], fd[lv], atol=1e-7)


def test_from_joint_coupling_rejections() -> None:
    jc = _jc()
    ok = {0: "flex", 1: (SpinalLevel.L5_S1, 0)}
    from_joint_coupling(jc, ok)
    # q3 is driven by a dependent coordinate (chain): not a single independent driver
    with pytest.raises(ValueError, match="driver"):
        from_joint_coupling(jc, {**ok, 3: (SpinalLevel.L4_L5, 0)})
    # driver address not mapped to an independent name
    with pytest.raises(ValueError, match="unmapped"):
        from_joint_coupling(jc, {1: (SpinalLevel.L5_S1, 0)})
    # map refers to a coordinate that is neither dependent nor a driver
    with pytest.raises(ValueError, match="unmapped"):
        from_joint_coupling(jc, {**ok, 9: (SpinalLevel.L4_L5, 0)})
    # two dependents on the same (level, axis)
    with pytest.raises(ValueError, match="duplicate"):
        from_joint_coupling(jc, {**ok, 2: (SpinalLevel.L5_S1, 0)})
    with pytest.raises(ValueError, match="axis"):
        from_joint_coupling(jc, {**ok, 2: (SpinalLevel.L4_L5, 4)})
    with pytest.raises(ValueError, match="contiguous"):
        from_joint_coupling(jc, {**ok, 2: (SpinalLevel.L3_L4, 0)})
    with pytest.raises(TypeError):
        from_joint_coupling("jc", ok)  # type: ignore[arg-type]


def test_from_joint_coupling_multi_driver() -> None:
    jc = JointCoupling(
        qpos0=np.zeros(3),
        dependent=(2,),
        driver=(0,),
        poly=np.array([[0.0, 1.0, 0.0, 0.0, 0.0]]),
    )
    # same level, two axes driven by different drivers is allowed (cross-axis)
    jc2 = JointCoupling(
        qpos0=np.zeros(4),
        dependent=(2, 3),
        driver=(0, 1),
        poly=np.array([[0.0, 1.0, 0.0, 0.0, 0.0]] * 2),
    )
    law = from_joint_coupling(
        jc2,
        {0: "a", 1: "b", 2: (SpinalLevel.L5_S1, 0), 3: (SpinalLevel.L5_S1, 1)},
    )
    np.testing.assert_allclose(
        law.expand(np.array([0.2, 0.3]))[SpinalLevel.L5_S1], [0.2, 0.3, 0.0]
    )
    # an independent name given twice (two drivers for one name) is rejected
    with pytest.raises(ValueError, match="driver"):
        from_joint_coupling(
            jc2,
            {0: "a", 1: "a", 2: (SpinalLevel.L5_S1, 0), 3: (SpinalLevel.L5_S1, 1)},
        )
    del jc
