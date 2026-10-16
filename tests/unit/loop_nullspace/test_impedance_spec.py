"""NSX-93 (#12722): the declared joint-impedance value type.

Impedance is a declared hypothesis (``H-DECLARED-JOINT-IMPEDANCE``, rule V5):
nothing here estimates a stiffness, and the only evidence the type carries is
the id of the hypothesis it stands for.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.impedance_spec import (
    HYPOTHESIS_ID,
    ImpedanceWindow,
    JointImpedance,
)

pytestmark = pytest.mark.unit

N = 3


def _window(**kw) -> ImpedanceWindow:
    args = {
        "t_start_s": 0.1,
        "t_end_s": 0.3,
        "stiffness": np.array([50.0, 20.0, 10.0]),
        "damping": np.array([2.0, 1.0, 0.5]),
    }
    args.update(kw)
    return ImpedanceWindow(**args)


def test_the_hypothesis_id_is_the_registry_id() -> None:
    assert HYPOTHESIS_ID == "H-DECLARED-JOINT-IMPEDANCE"
    assert JointImpedance((_window(),)).hypotheses == (HYPOTHESIS_ID,)


def test_torque_is_restoring_about_the_reference() -> None:
    """``tau = K (q_ref - q) + D (v_ref - v)``: it opposes the deviation."""
    imp = JointImpedance((_window(),))
    e = np.array([0.1, -0.2, 0.05])  # q_ref - q
    de = np.array([-0.3, 0.4, 0.0])
    tau = imp.torque(e, de, 0.2)
    np.testing.assert_allclose(tau, [50 * 0.1 - 2 * 0.3, -20 * 0.2 + 0.4, 10 * 0.05])
    # a stiffness-only deflection: the torque points back to the reference
    pure = imp.torque(e, np.zeros(N), 0.2)
    assert np.all(pure * e > 0.0)
    # damping alone is dissipative: power on the deviation rate is not positive
    damp = imp.torque(np.zeros(N), de, 0.2)
    assert np.all(damp * de >= 0.0)  # tau_D . (v_ref - v) >= 0 -> dissipates


def test_outside_every_window_there_is_no_impedance() -> None:
    imp = JointImpedance((_window(),))
    for t in (0.0, 0.0999, 0.3, 1.0):
        np.testing.assert_array_equal(imp.stiffness_at(t), np.zeros((N, N)))
        np.testing.assert_array_equal(imp.damping_at(t), np.zeros((N, N)))
        np.testing.assert_array_equal(
            imp.torque(np.ones(N), np.ones(N), t), np.zeros(N)
        )


def test_raised_cosine_ramp_is_continuous_and_bounded() -> None:
    imp = JointImpedance((_window(ramp_s=0.05),))
    ts = np.linspace(0.0, 0.4, 801)
    k = np.array([imp.stiffness_at(t)[0, 0] for t in ts])
    assert k.max() == pytest.approx(50.0)
    assert k.min() == 0.0
    assert np.max(np.abs(np.diff(k))) < 50.0 * 0.06  # no jump: smooth edge
    # half-way up the ramp the weight is exactly one half
    assert imp.stiffness_at(0.125)[0, 0] == pytest.approx(25.0)
    # flat plateau
    assert imp.stiffness_at(0.2)[0, 0] == pytest.approx(50.0)


def test_biarticular_coupling_is_an_off_diagonal_psd_term() -> None:
    k = np.array([[30.0, 10.0, 0.0], [10.0, 20.0, 0.0], [0.0, 0.0, 5.0]])
    imp = JointImpedance((_window(stiffness=k, damping=np.zeros((N, N))),))
    tau = imp.torque(np.array([1.0, 0.0, 0.0]), np.zeros(N), 0.2)
    np.testing.assert_allclose(tau, [30.0, 10.0, 0.0])  # cross-joint torque


def test_overlapping_windows_add() -> None:
    a = _window()
    b = _window(t_start_s=0.2, t_end_s=0.4)
    imp = JointImpedance((a, b))
    assert imp.stiffness_at(0.25)[0, 0] == pytest.approx(100.0)
    assert imp.stiffness_at(0.15)[0, 0] == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"stiffness": np.array([1.0, -1.0, 1.0])}, "stiffness"),
        ({"damping": np.array([1.0, -1.0, 1.0])}, "damping"),
        ({"stiffness": np.eye(N) * np.nan}, "finite"),
        ({"t_end_s": 0.05}, "t_end_s"),
        ({"ramp_s": -0.01}, "ramp_s"),
        ({"ramp_s": 0.2}, "ramp_s"),
        ({"stiffness": np.ones((N, N))[:2]}, "stiffness"),
    ],
)
def test_invalid_windows_are_rejected(kw, match) -> None:
    with pytest.raises(ValueError, match=match):
        _window(**kw)


def test_matrix_stiffness_must_be_symmetric_psd() -> None:
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="PSD"):
        ImpedanceWindow(0.0, 1.0, indefinite, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="symmetric"):
        ImpedanceWindow(0.0, 1.0, np.array([[1.0, 2.0], [0.0, 1.0]]), np.zeros((2, 2)))


def test_windows_must_agree_on_the_joint_count() -> None:
    other = ImpedanceWindow(0.0, 1.0, np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="joints"):
        JointImpedance((_window(), other))
    with pytest.raises(ValueError, match="at least one"):
        JointImpedance(())


def test_zero_impedance_is_flagged_and_exact() -> None:
    imp = JointImpedance.zero(N)
    assert imp.is_zero and imp.n_actuated == N
    np.testing.assert_array_equal(imp.torque(np.ones(N), np.ones(N), 0.2), np.zeros(N))
    assert not JointImpedance((_window(),)).is_zero


def test_noise_inflation_is_declared_and_scales_with_stiffness() -> None:
    """``sigma_eff = sigma (1 + kappa_i K_ii(t))`` - a declared penalty, not a fit."""
    kappa = np.array([0.01, 0.0, 0.02])
    imp = JointImpedance((_window(),), noise_inflation=kappa)
    np.testing.assert_allclose(imp.noise_scale_at(0.2), [1.5, 1.0, 1.2])
    np.testing.assert_array_equal(imp.noise_scale_at(0.0), np.ones(N))
    plain = JointImpedance((_window(),))
    np.testing.assert_array_equal(plain.noise_scale_at(0.2), np.ones(N))
    series = imp.noise_scale_series(np.array([0.0, 0.2]))
    assert series.shape == (2, N)
    with pytest.raises(ValueError, match="noise_inflation"):
        JointImpedance((_window(),), noise_inflation=np.array([-0.1, 0.0, 0.0]))
    with pytest.raises(ValueError, match="noise_inflation"):
        JointImpedance((_window(),), noise_inflation=np.ones(2))


def test_gain_matrices_over_time() -> None:
    imp = JointImpedance((_window(),))
    times = np.array([0.0, 0.2, 0.5])
    k, d = imp.gain_series(times)
    assert k.shape == (3, N, N) and d.shape == (3, N, N)
    np.testing.assert_allclose(np.diag(k[1]), [50, 20, 10])
    assert np.all(k[0] == 0.0) and np.all(d[2] == 0.0)
