"""#13112: replay receipts reproduce; post-impact divergence is measured and
the ZMP filter's centre-of-mass correction is bounded."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import dynamics_filter as dfilt
from src.shared.python.motion_matching.pipeline import replay_sensitivity as rs

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


# --- (a) reproducibility ----------------------------------------------------


@pytest.fixture(scope="module")
def tracked_reference():  # noqa: ANN201
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs

    spec_bytes = SPEC.read_bytes()

    def build() -> fs.FullBodySimulator:
        return fs.FullBodySimulator(NativeMujocoFullBodyModel(spec_bytes))

    sim = build()
    t = np.linspace(0.0, 0.03, 4)
    q0 = np.zeros(sim.nv)
    q0[2] = 2.0  # airborne: the test is about the integrator, not contact
    reference = np.repeat(q0[None, :], t.size, axis=0)
    reference[:, 10] += 0.2 * t
    return build, fs, t, reference


def _run(build, fs, t, reference):  # noqa: ANN001, ANN202
    sim = build()
    controller = fs.tracking_controller(sim, t, reference, omega_rad_s=30.0)
    return sim.run(
        reference[0], np.zeros(sim.nv), controller, duration_s=float(t[-1]), dt_s=1e-3
    )


def test_tracked_replay_is_bit_identical_across_fresh_simulators(
    tracked_reference,  # noqa: ANN001
) -> None:
    """Same inputs, separately built plants: the record reproduces bitwise.

    Run-to-run differences between committed and regenerated receipts
    therefore come from changed inputs (code or document drift, or a
    different host's floating-point summation order) amplified by the
    after-impact sensitivity, not from randomness in the replay.
    """
    build, fs, t, reference = tracked_reference
    a = _run(build, fs, t, reference)
    b = _run(build, fs, t, reference)
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.v, b.v)
    np.testing.assert_array_equal(a.tau, b.tau)


def test_compare_receipts_ignores_wall_clock_and_flags_real_changes() -> None:
    a = {
        "elapsed_s": 244.9,
        "dynamics": {"marker_rms_m": 0.09933, "passes": [{"x": 1.0}]},
        "label": "driver",
        "nan": float("nan"),
    }
    b = {
        "elapsed_s": 301.2,
        "dynamics": {"marker_rms_m": 0.09933, "passes": [{"x": 1.0}]},
        "label": "driver",
        "nan": float("nan"),
    }
    assert rs.compare_receipts(a, b) == []
    b["dynamics"]["marker_rms_m"] = 0.09933 + 5e-5
    assert rs.compare_receipts(a, b) == ["dynamics.marker_rms_m"]
    assert rs.compare_receipts(a, b, atol=rs.REPRODUCIBILITY_TOLERANCE_M) == []
    b["dynamics"]["passes"].append({"x": 2.0})
    assert "dynamics.passes#len" in rs.compare_receipts(a, b, atol=1.0)


# --- sensitivity characterisation ------------------------------------------


def test_growth_rate_recovers_a_known_exponent() -> None:
    t = np.linspace(0.0, 1.0, 201)
    sep = 1e-6 * np.exp(12.0 * t)
    assert rs.finite_time_growth_rate(t, sep, (0.0, 1.0)) == pytest.approx(12.0)
    # Round-off samples are skipped, not fitted as log(0).
    assert rs.finite_time_growth_rate(t, np.zeros_like(t), (0.0, 1.0)) is None
    with pytest.raises(ValueError):
        rs.finite_time_growth_rate(t, sep, (0.5, 0.5))


def test_divergence_report_separates_tracked_and_chaotic_phases() -> None:
    """Tracking contracts before impact; the follow-through amplifies."""
    t = np.linspace(0.0, 1.8, 649)
    impact = 1.2
    eps = 1e-6
    sep = np.where(
        t <= impact, eps * np.exp(-3.0 * t), eps * np.exp(-3.6 + 20.0 * (t - impact))
    )
    report = rs.divergence_report(t, sep, impact, eps, post_window_s=0.1)
    assert report["growth_rate_address_to_impact_per_s"] == pytest.approx(-3.0)
    assert report["growth_rate_after_impact_per_s"] == pytest.approx(20.0, rel=1e-3)
    assert report["e_folding_time_after_impact_s"] == pytest.approx(0.05, rel=1e-3)
    assert report["amplification_address_to_impact"] == pytest.approx(1.0)
    # The bounded window amplifies far less than the whole after-impact phase.
    assert (
        report["amplification_post_impact_window"]
        < 1e-3 * report["amplification_after_impact"]
    )


def test_replay_sensitivity_probe_is_seeded_and_reports_the_growth() -> None:
    t = np.linspace(0.0, 1.0, 101)
    q_track = np.zeros((101, 3))
    gain = np.where(t <= 0.5, 1.0, np.exp(10.0 * (t - 0.5)))

    def run_replay(reference: np.ndarray) -> np.ndarray:
        return reference * gain[:, None]

    def markers(q: np.ndarray) -> np.ndarray:
        return np.array([q])

    nominal = run_replay(q_track)
    a = rs.replay_sensitivity(run_replay, markers, t, q_track, nominal, 0.5)
    b = rs.replay_sensitivity(run_replay, markers, t, q_track, nominal, 0.5)
    assert rs.compare_receipts(a, b) == []
    assert a["growth_rate_after_impact_per_s"] == pytest.approx(10.0, rel=1e-6)
    assert a["perturbation_seed"] == rs.PERTURBATION_SEED


def test_post_impact_window_rms_uses_only_the_bounded_window() -> None:
    t = np.linspace(0.0, 1.0, 11)
    errors = np.where(t[:, None] <= 0.65, 0.01, 1.0) * np.ones((11, 2))
    valid = np.ones((11, 2), dtype=bool)
    block = rs.post_impact_window_rms(t, errors, valid, 0.5, window_s=0.15)
    assert block["post_impact_window_frames"] == 1  # t = 0.6
    assert block["fd_rms_post_impact_window_m"] == pytest.approx(0.01)


# --- (b) bounded centre-of-mass correction ----------------------------------

#: Double-stance footprint (m): two 0.26 m feet 0.40 m apart (centre to
#: centre); its narrowest width is one foot length.
FEET = np.array(
    [
        [-0.13, -0.25],
        [0.13, -0.25],
        [0.13, -0.15],
        [-0.13, -0.15],
        [-0.13, 0.15],
        [0.13, 0.15],
        [0.13, 0.25],
        [-0.13, 0.25],
    ]
)


def test_support_min_width_is_the_narrowest_base_dimension() -> None:
    assert dfilt.support_min_width(FEET) == pytest.approx(0.26)
    one_foot = FEET[:4]
    assert dfilt.support_min_width(one_foot) == pytest.approx(0.10)
    assert dfilt.support_min_width(np.array([[0.0, 0.0], [1.0, 0.0]])) == 0.0


def test_com_shift_cannot_exceed_the_base_of_support() -> None:
    """The pre-torsion 7-iron diverged to a 0.83-1.04 m shift (#13112).

    A frame whose reference ZMP is metres outside the feet (vertical
    reaction barely above the unloaded threshold) asks the cart-table solve
    for a metre-scale shift. The bound keeps every frame's correction within
    the base of support's narrowest width and leaves in-range frames alone.
    """
    times = np.linspace(0.0, 1.0, 101)
    zmp = np.zeros((101, 2))
    target = zmp.copy()
    target[40:70, 0] = 1.0  # a sustained 1 m gap
    target[20:30, 1] = 0.03  # an ordinary 3 cm correction
    shift = dfilt.cart_table_shift(zmp, target, 0.9, times)
    assert np.abs(shift).max() > 0.5  # the unbounded solve diverges
    limit = dfilt.support_min_width(FEET)
    bounded, clipped = dfilt.bound_com_shift(shift, limit)
    norms = np.sqrt(np.einsum("ij,ij->i", bounded, bounded))
    assert norms.max() <= limit + 1e-12
    assert clipped.any()
    assert not clipped[20:30].any()
    np.testing.assert_array_equal(bounded[~clipped], shift[~clipped])
    # Direction is kept on the clipped frames.
    k = int(np.argmax(np.sqrt(np.einsum("ij,ij->i", shift, shift))))
    np.testing.assert_allclose(
        bounded[k] / np.linalg.norm(bounded[k]), shift[k] / np.linalg.norm(shift[k])
    )
    with pytest.raises(ValueError):
        dfilt.bound_com_shift(shift, -1.0)
