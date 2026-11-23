"""Conditional earliest sphere-clearance bracket diagnostics."""

import math

import pytest

from src.shared.python.motion_matching import contact_event

pytestmark = pytest.mark.unit


def test_adaptive_scan_finds_narrow_crossing_missed_by_coarse_endpoints():
    def clearance(t):
        return abs(t - 0.314159) - 0.002

    result = contact_event.scan_first_clearance_bracket(
        clearance,
        duration_s=1.0,
        max_clearance_rate_mps=1.0,
        time_tolerance_s=1e-5,
        max_evaluations=1000,
    )
    assert result.status == "conditional_bracket"
    assert result.start_s <= 0.312159 <= result.end_s
    assert result.end_s - result.start_s <= 1e-5


def test_clear_and_insufficient_resolution_are_distinct():
    clear = contact_event.scan_first_clearance_bracket(
        lambda t: 1.0,
        duration_s=1.0,
        max_clearance_rate_mps=1.0,
        time_tolerance_s=1e-4,
        max_evaluations=100,
    )
    assert clear.status == "conditionally_clear"
    uncertain = contact_event.scan_first_clearance_bracket(
        lambda t: abs(t - 0.5) + 1e-9,
        duration_s=1.0,
        max_clearance_rate_mps=3.0,
        time_tolerance_s=1e-3,
        max_evaluations=100,
    )
    assert uncertain.status == "unresolved"


def test_earliest_unresolved_prevents_returning_later_crossing():
    def clearance(t):
        if t < 0.5:
            return abs(t - 0.25) + 1e-9
        return 0.75 - t

    result = contact_event.scan_first_clearance_bracket(
        clearance,
        duration_s=1.0,
        max_clearance_rate_mps=3.0,
        time_tolerance_s=1e-3,
        max_evaluations=1000,
    )
    assert result.status == "unresolved"
    assert result.start_s <= 0.25 <= result.end_s


@pytest.mark.parametrize("func", [lambda t: math.nan, lambda t: 2 * t + 1])
def test_bad_clearance_or_contradicted_bound_fails_closed(func):
    with pytest.raises(ValueError):
        contact_event.scan_first_clearance_bracket(
            func,
            duration_s=1.0,
            max_clearance_rate_mps=1.0,
            time_tolerance_s=1e-4,
            max_evaluations=100,
        )


def test_scanner_brackets_real_triangle_contact_between_coarse_samples():
    import numpy as np

    face = np.array([[[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]])
    center = np.array([1.0, 0.0, 0.0])
    radius = 0.01

    def clearance(t):
        triangles = face + np.array([4.0 * t, 0.0, 0.0])
        data = contact_event._surface_sample(triangles, center, radius)
        return float(np.linalg.norm(center - data[3]) - radius)

    result = contact_event.scan_first_clearance_bracket(
        clearance,
        duration_s=1.0,
        max_clearance_rate_mps=4.0,
        time_tolerance_s=1e-6,
        max_evaluations=1000,
    )
    assert result.status == "conditional_bracket"
    assert result.start_s <= 0.2475 <= result.end_s
    assert clearance(result.start_s) > 0.0
    assert clearance(result.end_s) <= 0.0


def test_budget_exhaustion_and_initial_overlap_stay_unresolved():
    budget = contact_event.scan_first_clearance_bracket(
        lambda t: abs(t - 0.4) - 0.01,
        duration_s=1.0,
        max_clearance_rate_mps=1.0,
        time_tolerance_s=1e-6,
        max_evaluations=2,
    )
    assert budget.status == "unresolved"
    assert budget.evaluations == 2
    overlap = contact_event.scan_first_clearance_bracket(
        lambda t: t - 0.1,
        duration_s=1.0,
        max_clearance_rate_mps=1.0,
        time_tolerance_s=1e-6,
        max_evaluations=100,
    )
    assert overlap.status == "unresolved"


def test_integrated_adaptive_finder_discovers_narrow_finite_face_contact():
    import numpy as np

    face = np.array([[[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]])
    result = contact_event.find_contact_event(
        lambda t: (np.array([4.0 * t]), np.array([4.0])),
        lambda q: face + np.array([q[0], 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        0.01,
        duration_s=1.0,
        time_tolerance_s=1e-7,
        face_point_velocity_world=lambda q, v, point: np.array([4.0, 0.0, 0.0]),
        max_clearance_rate_mps=4.0,
        coverage_max_evaluations=1000,
    )
    assert result.event is not None
    assert result.event.time_s == pytest.approx(0.2475, abs=1e-7)
    assert result.coverage_status == "conditional_bracket_candidate"


def test_integrated_adaptive_finder_fails_closed_on_budget_exhaustion():
    import numpy as np

    face = np.array([[[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]])
    result = contact_event.find_contact_event(
        lambda t: (np.array([t]), np.array([1.0])),
        lambda q: face + np.array([q[0], 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        0.5,
        duration_s=1.0,
        face_point_velocity_world=lambda q, v, point: np.array([1.0, 0.0, 0.0]),
        max_clearance_rate_mps=1.0,
        coverage_max_evaluations=2,
    )
    assert result.event is None
    assert result.coverage_status == "unresolved_coverage"
    assert math.isfinite(result.closest_approach_penalty_m)


def test_adaptive_finder_on_fixed_lattice_mujoco():
    import numpy as np

    mujoco = pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.same_input.integrator import zoh_rk4_step

    model = mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 0 0"/>
        <worldbody><body><joint type="slide" axis="1 0 0"/>
        <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
        </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)

    class Plant:
        def acceleration(self, q, v, force):
            data.qpos[:], data.qvel[:], data.qfrc_applied[:] = q, v, force
            mujoco.mj_forward(model, data)
            return data.qacc.copy()

    face = np.array([[[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]])

    def advance(t):
        q, v = np.zeros(1), np.ones(1)
        fixed = 0.001
        count = int(t / fixed)
        for _ in range(count):
            q, v = zoh_rk4_step(Plant(), q, v, np.zeros(1), fixed)
        if t > count * fixed:
            q, v = zoh_rk4_step(Plant(), q, v, np.zeros(1), t - count * fixed)
        return q, v

    result = contact_event.find_contact_event(
        advance,
        lambda q: face + np.array([q[0], 0.0, 0.0]),
        np.array([0.03, 0.0, 0.0]),
        0.01,
        duration_s=0.04,
        time_tolerance_s=1e-7,
        max_clearance_rate_mps=1.0,
        coverage_max_evaluations=1000,
        face_point_velocity_world=lambda q, v, point: np.array([v[0], 0.0, 0.0]),
    )
    assert result.event is not None
    assert result.event.time_s == pytest.approx(0.02, abs=1e-7)
    assert result.coverage_status == "conditional_bracket_candidate"


def test_integrated_finder_does_not_leap_over_unresolved_earlier_interval():
    import numpy as np

    face = np.array([[[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]])

    def advance(t):
        gap = abs(t - 0.25) + 1e-9 if t < 0.5 else 0.75 - t
        speed = (1.0 if t < 0.25 else -1.0) if t < 0.5 else 1.0
        return np.array([0.9 - gap]), np.array([speed])

    result = contact_event.find_contact_event(
        advance,
        lambda q: face + np.array([q[0], 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        0.1,
        duration_s=1.0,
        time_tolerance_s=1e-4,
        max_clearance_rate_mps=3.0,
        coverage_max_evaluations=1000,
        face_point_velocity_world=lambda q, v, point: np.array([v[0], 0.0, 0.0]),
    )
    assert result.event is None
    assert result.coverage_status == "unresolved_coverage"
