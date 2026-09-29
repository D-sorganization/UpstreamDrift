"""Unit tests for ``cost.compute_cost``.

Mirrors the MATLAB reference implementation at
``src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/
shared/compute_cost.m``. The ``test_python_matches_matlab_compute_cost`` test
is the contract that the Python mirror is faithful: rather than running
MATLAB, ``J`` is computed analytically from the cost-function formulas (see
``COST_FUNCTION_SPEC.md``) for two cases whose answers are derivable in
closed form to ``1e-12``.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching.final_cost import (
    CostBreakdown,
    CostOptions,
    SimOutput,
    compute_cost,
)

from ._fixtures import make_target


def _zero_kinetics(n: int, n_joints: int = 3) -> dict[str, np.ndarray]:
    """Time/tau/omega payload that integrates to zero work."""
    return {
        "time": np.linspace(0.0, 0.3, n),
        "tau": np.zeros((n, n_joints)),
        "omega": np.zeros((n, n_joints)),
    }


def _matching_sim(target, **overrides) -> SimOutput:
    """SimOutput that reproduces ``target`` exactly (residuals = 0)."""
    n = target.time.shape[0]
    base = dict(
        butt=target.butt.copy(),
        clubhead=target.clubhead.copy(),
        club_quat=target.club_quat.copy(),
        **_zero_kinetics(n),
    )
    base.update(overrides)
    return SimOutput(**base)


def _const_sim_fn(sim_out: SimOutput):
    """Wrap a SimOutput as a sim_fn that ignores theta."""
    return lambda _theta: sim_out


def test_zero_difference_yields_lambda_times_work() -> None:
    target = make_target(n=51)
    n = target.time.shape[0]
    # Non-zero work: tau=2, omega=3, 2 joints, T=0.3 s -> W=2*|2*3|*0.3 = 3.6
    tau = np.full((n, 2), 2.0)
    omega = np.full((n, 2), 3.0)
    sim = _matching_sim(target, tau=tau, omega=omega)
    j, terms = compute_cost(np.zeros(7), target, _const_sim_fn(sim), CostOptions())
    expected_w = 2.0 * abs(2.0 * 3.0) * 0.3  # = 3.6
    assert abs(terms.regularizer - 1e-4 * expected_w) < 1e-15
    assert terms.position == 0.0
    assert terms.orientation == 0.0
    assert terms.impact_anchor == 0.0
    assert abs(j - 1e-4 * expected_w) < 1e-15


def test_position_term_units_metres_squared() -> None:
    target = make_target(n=10)
    sim = _matching_sim(target)
    # Offset every butt frame by 1 mm in x: per-frame ||db||^2 = 1e-6 m^2
    new_butt = sim.butt + np.array([1e-3, 0.0, 0.0])
    sim2 = SimOutput(
        butt=new_butt,
        clubhead=sim.clubhead,
        club_quat=sim.club_quat,
        time=sim.time,
        tau=sim.tau,
        omega=sim.omega,
    )
    opts = CostOptions(w_anchor_impact=0.0)
    _, terms = compute_cost(np.zeros(7), target, _const_sim_fn(sim2), opts)
    assert abs(terms.position - 1.0 * 1e-6) < 1e-18


def test_orientation_term_geodesic() -> None:
    """``q`` and ``-q`` represent the same rotation -> zero orientation term."""
    target = make_target(n=10)
    sim = _matching_sim(target)
    flipped = SimOutput(
        butt=sim.butt,
        clubhead=sim.clubhead,
        club_quat=-sim.club_quat,
        time=sim.time,
        tau=sim.tau,
        omega=sim.omega,
    )
    _, terms = compute_cost(np.zeros(7), target, _const_sim_fn(flipped), CostOptions())
    assert terms.orientation < 1e-20


def test_impact_anchor_amplifies() -> None:
    target = make_target(n=11)
    sim = _matching_sim(target)
    # 1 mm clubhead offset only at impact frame
    new_ch = sim.clubhead.copy()
    k = target.impact_idx
    new_ch[k - 1] = new_ch[k - 1] + np.array([1e-3, 0.0, 0.0])
    sim2 = SimOutput(
        butt=sim.butt,
        clubhead=new_ch,
        club_quat=sim.club_quat,
        time=sim.time,
        tau=sim.tau,
        omega=sim.omega,
    )
    opts = CostOptions(w_anchor_impact=10.0, w_position=1.0)
    _, terms = compute_cost(np.zeros(7), target, _const_sim_fn(sim2), opts)
    # anchor term = 10 * (1e-3)^2 = 1e-5
    assert abs(terms.impact_anchor - 1e-5) < 1e-18
    # position term = (1/N) * (1e-3)^2 = 1e-6 / 11
    assert abs(terms.position - (1e-6 / 11)) < 1e-18
    # anchor strictly dominates position by factor of ~110
    assert terms.impact_anchor > terms.position * 100


@pytest.mark.parametrize(
    "name",
    ["total_work", "peak_power", "torque_l2", "coeff_l2"],
)
def test_regularizer_choice_switch(name: str) -> None:
    target = make_target(n=21)
    n = target.time.shape[0]
    tau = np.full((n, 2), 1.5)
    omega = np.full((n, 2), 2.0)
    sim = _matching_sim(target, tau=tau, omega=omega)
    theta = np.array([1.0, 2.0, 3.0])
    opts = CostOptions(regularizer=name)  # type: ignore[arg-type]
    _, terms = compute_cost(theta, target, _const_sim_fn(sim), opts)

    if name == "total_work":
        # 2 joints * |1.5*2.0| * 0.3 = 1.8
        expected = 1e-4 * 1.8
    elif name == "peak_power":
        # max over t of sum(|tau*omega|, axis=1) = 2 * 3 = 6
        expected = 1e-4 * 6.0
    elif name == "torque_l2":
        # trapz over [0, 0.3] of (2 * 1.5^2) = 4.5 * 0.3 = 1.35
        expected = 1e-4 * 1.35
    else:  # coeff_l2
        expected = 1e-4 * (1.0 + 4.0 + 9.0)
    assert abs(terms.regularizer - expected) < 1e-15


def test_terms_sum_to_total() -> None:
    target = make_target(n=15)
    sim = _matching_sim(target)
    j, terms = compute_cost(np.zeros(7), target, _const_sim_fn(sim), CostOptions())
    s = (
        terms.position
        + terms.orientation
        + terms.impact_anchor
        + terms.body_marker
        + terms.regularizer
    )
    assert abs(s - terms.total) < 1e-15
    assert abs(terms.total - j) < 1e-15
    assert isinstance(terms, CostBreakdown)


def test_invalid_target_dataclass_rejected_by_precondition() -> None:
    target = make_target(n=10)
    sim = _matching_sim(target)
    not_a_target = {"butt": target.butt}  # plain dict, not ClubTarget
    from src.shared.python.core.contracts.exceptions import PreconditionError

    with pytest.raises((PreconditionError, TypeError, ValueError)):
        compute_cost(np.zeros(7), not_a_target, _const_sim_fn(sim))  # type: ignore[arg-type]


def test_precondition_rejects_nan_theta() -> None:
    target = make_target(n=10)
    sim = _matching_sim(target)
    from src.shared.python.core.contracts.exceptions import PreconditionError

    with pytest.raises((PreconditionError, ValueError)):
        compute_cost(np.array([1.0, np.nan, 0.0]), target, _const_sim_fn(sim))


def test_python_matches_matlab_compute_cost() -> None:
    """Cross-check Python output against analytically-derived MATLAB values.

    Because we cannot execute MATLAB locally, ``J`` is computed by hand
    from the formulas in ``COST_FUNCTION_SPEC.md`` for two cases:

    Case 1: sim_out reproduces target exactly except for ``W_total > 0``.
            All position/orientation/anchor terms collapse to zero, so
            ``J == lambda * W_total``, derivable to machine precision.

    Case 2: a single 1 mm clubhead offset on one non-impact frame, with
            zero kinetics. Closed form: ``J = w_p * (1 mm)^2 / N``.

    Both cases must match the Python implementation to 1e-12.
    """
    target = make_target(n=31)
    n = target.time.shape[0]

    # --- Case 1: residuals zero, W_total > 0 ---
    tau1 = np.full((n, 4), 0.7)
    omega1 = np.full((n, 4), -1.25)  # negative omega exercises abs()
    sim1 = _matching_sim(target, tau=tau1, omega=omega1)
    opts1 = CostOptions()  # lambda_=1e-4, total_work
    # W_total = 4 joints * |0.7*1.25| * 0.3 = 4 * 0.875 * 0.3 = 1.05
    expected_w = 4.0 * abs(0.7 * 1.25) * 0.3
    expected_j_1 = 1e-4 * expected_w
    j1, terms1 = compute_cost(np.zeros(7), target, _const_sim_fn(sim1), opts1)
    delta_1 = abs(j1 - expected_j_1)
    assert delta_1 < 1e-12, f"case 1 delta={delta_1:.2e}"
    assert terms1.position == 0.0
    assert terms1.orientation == 0.0
    assert terms1.impact_anchor == 0.0

    # --- Case 2: single 1 mm clubhead offset on a non-impact frame ---
    sim2_base = _matching_sim(target)
    new_ch = sim2_base.clubhead.copy()
    # pick a frame that is NOT impact
    offset_idx = 0 if target.impact_idx != 1 else n - 1
    new_ch[offset_idx] = new_ch[offset_idx] + np.array([1e-3, 0.0, 0.0])
    sim2 = SimOutput(
        butt=sim2_base.butt,
        clubhead=new_ch,
        club_quat=sim2_base.club_quat,
        time=sim2_base.time,
        tau=sim2_base.tau,
        omega=sim2_base.omega,
    )
    opts2 = CostOptions(lambda_=0.0, w_anchor_impact=0.0)
    j2, terms2 = compute_cost(np.zeros(7), target, _const_sim_fn(sim2), opts2)
    expected_j_2 = 1.0 * (1e-3) ** 2 / n  # w_p * (1mm)^2 / N
    delta_2 = abs(j2 - expected_j_2)
    assert delta_2 < 1e-12, f"case 2 delta={delta_2:.2e}"
    assert terms2.impact_anchor == 0.0
    assert terms2.regularizer == 0.0


class DummyBody:
    def __init__(self, marker_xyz):
        self.marker_xyz = marker_xyz


class DummyMultiTarget:
    def __init__(self, club, marker_xyz):
        self.club = club
        self.body = DummyBody(marker_xyz)
        self.time = club.time


def test_body_marker_term() -> None:
    from src.shared.python.motion_matching.cost import CostOptions, compute_cost

    target_club = make_target(n=10)
    target_markers = np.zeros((10, 2, 3))
    target = DummyMultiTarget(target_club, target_markers)

    sim_markers = np.zeros((10, 2, 3))
    sim_markers[:, 0, 0] = 1e-3
    sim = _matching_sim(target_club)
    # create a new sim_out with marker_xyz
    # Since SimOutput is frozen, we do it carefully:
    sim_dict = {
        "butt": sim.butt,
        "clubhead": sim.clubhead,
        "club_quat": sim.club_quat,
        "time": sim.time,
        "tau": sim.tau,
        "omega": sim.omega,
        "marker_xyz": sim_markers,
    }
    from src.shared.python.motion_matching.cost import SimOutput

    sim = SimOutput(**sim_dict)

    opts = CostOptions(w_body_marker=2.0)
    _, terms = compute_cost(np.zeros(7), target, _const_sim_fn(sim), opts)
    assert abs(terms.body_marker - 1e-6) < 1e-15
