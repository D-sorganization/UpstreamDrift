"""Characterization tests for the pendulum ``run_simulation`` runners (#12844).

Pin the numerical output and keyword surface of the three runners so the
SimulationConfig parameter-object refactor stays behaviour-preserving.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pendulum_simulator.physics import (
    JointLimits,
    JointLimitsNDOF,
    PendulumParams,
    TorqueClamp,
)
from src.shared.python.pendulum_simulator.physics_golfer import GolferParams
from src.shared.python.pendulum_simulator.physics_triple import TriplePendulumParams
from src.shared.python.pendulum_simulator.simulation import run_simulation as run_d
from src.shared.python.pendulum_simulator.simulation_golfer import (
    run_simulation as run_g,
)
from src.shared.python.pendulum_simulator.simulation_triple import (
    run_simulation as run_t,
)

DP = PendulumParams(m1=5.0, m2=0.3, L1=0.65, L2=1.1, mClub=0.2)
TP = TriplePendulumParams(m1=5.0, m2=0.3, m3=0.05, L1=0.65, L2=1.1, L3=0.1)
GP = GolferParams(
    m_hub=0.01,
    m_r_upper=2.0,
    m_r_fore=1.5,
    m_l_upper=2.0,
    m_l_fore=1.5,
    m_club=0.3,
    L_hub=0.05,
    L_r_upper=0.35,
    L_r_fore=0.30,
    L_l_upper=0.35,
    L_l_fore=0.30,
    L_club=1.0,
    d_rs=0.2,
    d_ls=0.2,
    grip_right=0.2,
    grip_left=0.25,
)
Y2 = np.array([0.3, 0.2, 0.1, 0.0])
Y3 = np.array([0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
YG = np.zeros(16)
YG[:8] = 0.05


def _td(_t: float) -> tuple[float, float]:
    return 3.0, -1.0


def _tt(_t: float) -> tuple[float, float, float]:
    return 2.0, -1.0, 0.5


def _tg(_t: float) -> np.ndarray:
    return np.full(7, 1.0)


def _lim3() -> JointLimitsNDOF:
    return JointLimitsNDOF(
        angle_min=np.full(3, -0.25),
        angle_max=np.full(3, 0.25),
        stiffness=100.0,
        damping=1.0,
    )


def _lim7() -> JointLimitsNDOF:
    return JointLimitsNDOF(
        angle_min=np.full(7, -0.02),
        angle_max=np.full(7, 0.02),
        stiffness=100.0,
        damping=1.0,
    )


def _check(result: object, key: str) -> None:
    steps, final = GOLD[key]
    assert result.n_steps == steps  # type: ignore[attr-defined]
    np.testing.assert_allclose(
        result.states[-1],
        np.array(final),
        rtol=1e-9,
        atol=1e-12,  # type: ignore[attr-defined]
    )


GOLD = {
    "d_default": (
        21,
        [
            0.27948836471800453,
            0.1468837956432432,
            -0.2966236904588776,
            -0.5249184503439089,
        ],
    ),
    "d_lim_clamp": (
        21,
        [
            0.2546341926043619,
            0.20173087173440254,
            -0.5308162354441579,
            -0.00844674996563027,
        ],
    ),
    "d_method": (
        21,
        [
            0.2794883647198277,
            0.14688379564122253,
            -0.29662369046298215,
            -0.524918450337651,
        ],
    ),
    "t_default": (
        21,
        [
            0.2537453380119333,
            0.11773018720124517,
            21.59674659912161,
            -0.42768517707722403,
            1.800055206803366,
            218.27961096331762,
        ],
    ),
    "t_limits": (
        21,
        [
            0.2231641163576798,
            0.1897056141492645,
            0.2676047327087128,
            -0.6251547397552166,
            -0.28499455560540105,
            -0.0002387524195401922,
        ],
    ),
    "t_tl": (
        21,
        [
            0.24480471566995382,
            0.13159262121230272,
            21.5889643514091,
            -0.5104602044814022,
            1.9178006572199182,
            218.1338565140631,
        ],
    ),
    "t_clamp": (
        21,
        [
            0.23332678401485196,
            0.17134515387169547,
            21.489583156449527,
            -0.6091218002962996,
            2.1511115686095943,
            216.29825798531064,
        ],
    ),
    "t_both": (
        21,
        [
            0.24480471566995382,
            0.13159262121230272,
            21.5889643514091,
            -0.5104602044814022,
            1.9178006572199182,
            218.1338565140631,
        ],
    ),
    "t_tol": (
        21,
        [
            0.25374533881749417,
            0.11773016034495871,
            21.596742837950725,
            -0.42768503673121555,
            1.8000497738503782,
            218.27951305359636,
        ],
    ),
    "g_default": (
        6,
        [
            0.26078206706961815,
            -0.37149353183936956,
            -0.129143279542255,
            0.050000000000000024,
            0.1322191080479653,
            0.11052309701254043,
            0.050000000000000024,
            0.04510347464643935,
            0.14720852461424216,
            -0.3462014993424067,
            0.5792742667737343,
            9.3319916071534e-17,
            -0.7784491328493952,
            1.4716889708409298,
            3.090374731418943e-16,
            -0.193488742487905,
        ],
    ),
    "g_limits": (
        6,
        [
            0.11782567214749068,
            -0.23002839316256093,
            -0.11506990851742065,
            0.04999999999999998,
            0.23233117931355093,
            0.09680488628064948,
            0.04999999999999998,
            0.018789933993664576,
            -4.094570695883679,
            4.0635864882886015,
            0.9675473150477897,
            -1.170779206841302e-15,
            1.943831504394057,
            0.7125654941421734,
            -3.2009638285806596e-16,
            -1.1944138615545143,
        ],
    ),
    "g_tl": (
        6,
        [
            0.25873611141718245,
            -0.36771157409367855,
            -0.13623750057687334,
            0.049999999999999996,
            0.13647719719090753,
            0.1029939699499952,
            0.05000000000000002,
            0.04634488472130931,
            0.06378417091399316,
            -0.19530352506364945,
            0.3017748180485371,
            -7.978512338118288e-16,
            -0.6095440557593967,
            1.1769064733363652,
            2.722069774325603e-16,
            -0.1446762814656178,
        ],
    ),
    "g_clamp": (
        6,
        [
            0.25873611141718245,
            -0.36771157409367855,
            -0.13623750057687334,
            0.049999999999999996,
            0.13647719719090753,
            0.1029939699499952,
            0.05000000000000002,
            0.04634488472130931,
            0.06378417091399316,
            -0.19530352506364945,
            0.3017748180485371,
            -7.978512338118288e-16,
            -0.6095440557593967,
            1.1769064733363652,
            2.722069774325603e-16,
            -0.1446762814656178,
        ],
    ),
    "g_both": (
        6,
        [
            0.2595541304523784,
            -0.3692232646117913,
            -0.1334006291859254,
            0.05000000000000003,
            0.13477361762862955,
            0.10600583760809844,
            0.050000000000000024,
            0.045848562465843685,
            0.0971271334350921,
            -0.2555809098236043,
            0.41271669965377056,
            4.60719174079995e-16,
            -0.6771370954751434,
            1.2948433794149503,
            5.014807290499445e-16,
            -0.16418359978554903,
        ],
    ),
    "g_ab": (
        6,
        [
            0.2607820668194342,
            -0.3714935314319901,
            -0.12914328006818884,
            0.04999999999999996,
            0.13221910964747874,
            0.11052309385817732,
            0.05,
            0.04510347466617437,
            0.14720853953285593,
            -0.34620151437189817,
            0.5792742688904564,
            -1.3094149165359848e-15,
            -0.778449145776049,
            1.4716889745504795,
            1.0113188036064482e-16,
            -0.1934887429765603,
        ],
    ),
}

CASES = [
    ("d_default", lambda: run_d(DP, Y2, 0.2, _td, dt=0.01)),
    (
        "d_lim_clamp",
        lambda: run_d(
            DP, Y2, 0.2, _td, dt=0.01, limits=JointLimits(), clamp=TorqueClamp(1.0, 0.5)
        ),
    ),
    ("d_method", lambda: run_d(DP, Y2, 0.2, _td, 0.01, "DOP853")),
    ("t_default", lambda: run_t(TP, Y3, 0.2, _tt, dt=0.01)),
    ("t_limits", lambda: run_t(TP, Y3, 0.2, _tt, dt=0.01, limits=_lim3())),
    (
        "t_tl",
        lambda: run_t(TP, Y3, 0.2, _tt, dt=0.01, torque_limits=np.full(3, 1.0)),
    ),
    ("t_clamp", lambda: run_t(TP, Y3, 0.2, _tt, dt=0.01, clamp=np.full(3, 0.5))),
    (
        "t_both",
        lambda: run_t(
            TP,
            Y3,
            0.2,
            _tt,
            dt=0.01,
            clamp=np.full(3, 0.5),
            torque_limits=np.full(3, 1.0),
        ),
    ),
    (
        "t_tol",
        lambda: run_t(TP, Y3, 0.2, _tt, dt=0.01, method="RK45", rtol=1e-9, atol=1e-11),
    ),
    ("g_default", lambda: run_g(GP, YG, 0.05, _tg, dt=0.01)),
    ("g_limits", lambda: run_g(GP, YG, 0.05, _tg, dt=0.01, limits=_lim7())),
    (
        "g_tl",
        lambda: run_g(GP, YG, 0.05, _tg, dt=0.01, torque_limits=np.full(7, 0.5)),
    ),
    ("g_clamp", lambda: run_g(GP, YG, 0.05, _tg, dt=0.01, clamp=np.full(7, 0.5))),
    (
        "g_both",
        lambda: run_g(
            GP,
            YG,
            0.05,
            _tg,
            dt=0.01,
            clamp=np.full(7, 0.5),
            torque_limits=np.full(7, 0.7),
        ),
    ),
    (
        "g_ab",
        lambda: run_g(
            GP,
            YG,
            0.05,
            _tg,
            dt=0.01,
            alpha=2.0,
            beta=3.0,
            rtol=1e-8,
            atol=1e-10,
            method="RK45",
        ),
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("key", "call"), CASES, ids=[c[0] for c in CASES])
def test_runner_output_is_pinned(key: str, call) -> None:
    _check(call(), key)


@pytest.mark.unit
@pytest.mark.parametrize(
    "runner,args",
    [
        (run_d, (DP, Y2, 0.2, _td)),
        (run_t, (TP, Y3, 0.2, _tt)),
        (run_g, (GP, YG, 0.05, _tg)),
    ],
    ids=["double", "triple", "golfer"],
)
def test_runner_rejects_unknown_keyword(runner, args) -> None:
    with pytest.raises(TypeError):
        runner(*args, dt=0.01, bogus_option=1)


@pytest.mark.unit
@pytest.mark.parametrize(
    "runner,args",
    [(run_d, (DP, Y2, 0.2, _td)), (run_t, (TP, Y3, 0.2, _tt))],
    ids=["double", "triple"],
)
def test_runner_rejects_dt_not_below_t_end(runner, args) -> None:
    with pytest.raises(ValueError):
        runner(*args, dt=0.5)
