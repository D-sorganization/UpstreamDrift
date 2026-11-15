"""Unit tests for GM-3 Cheetham X-factor stretch and kinematic sequence (#14426)."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.analysis.swing_metrics import SwingMetricsMixin
from src.shared.python.biomechanics.metric_definitions.xfactor_kinematic_sequence import (
    KINEMATIC_SEQUENCE_CHEETHAM_ENTRY,
    XFACTOR_STRETCH_CHEETHAM_ENTRY,
    VelocityComponentKind,
    compute_cheetham_kinematic_sequence,
    compute_cheetham_xfactor_stretch,
    compute_xfactor_velocity,
)

pytestmark = pytest.mark.unit


def test_cheetham_xfactor_stretch_computation_and_no_stretch() -> None:
    t = np.linspace(0.0, 1.0, 101)
    xf = np.full(101, 20.0)
    top_idx, max_idx, impact_idx = 40, 60, 80
    xf[top_idx], xf[max_idx], xf[impact_idx] = 30.0, 42.0, 15.0

    r = compute_cheetham_xfactor_stretch(xf, t, top_idx=top_idx, impact_idx=impact_idx)
    assert r.x_factor_at_top_deg == pytest.approx(30.0)
    assert r.max_downswing_x_factor_deg == pytest.approx(42.0)
    assert r.stretch_deg == pytest.approx(12.0) and r.stretch_pct == pytest.approx(40.0)
    assert r.top_time_s == pytest.approx(
        t[top_idx]
    ) and r.max_downswing_time_s == pytest.approx(t[max_idx])

    # Monotonic downswing: stretch is 0
    r0 = compute_cheetham_xfactor_stretch(
        np.linspace(35.0, 10.0, 50), np.linspace(0, 1, 50), top_idx=0
    )
    assert r0.stretch_deg == 0.0 and r0.stretch_pct == 0.0


def test_xfactor_velocity_and_deprecation_alias() -> None:
    t = np.linspace(0.0, 1.0, 101)
    xf = 30.0 * np.sin(2.0 * np.pi * t)
    vel, peak = compute_xfactor_velocity(xf, t)
    assert vel.size == t.size and peak > 0.0

    class DummyContext(SwingMetricsMixin):
        def __init__(self) -> None:
            self.times, self.dt = t, float(t[1] - t[0])

        def compute_x_factor(self, s: int, h: int) -> np.ndarray:
            return xf

    ctx = DummyContext()
    vel_res = ctx.compute_x_factor_velocity(0, 1)
    assert vel_res is not None and vel_res[1] == pytest.approx(peak, rel=1e-4)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        legacy_res = ctx.compute_x_factor_stretch(0, 1)
        assert len(w) == 1 and issubclass(w[0].category, DeprecationWarning)
        assert legacy_res is not None and legacy_res[1] == pytest.approx(vel_res[1])


def test_kinematic_sequence_evaluation_and_gains() -> None:
    t = np.linspace(0.0, 1.0, 101)

    def _peak(tp: float, amp: float) -> np.ndarray:
        return amp * np.exp(-((t - tp) ** 2) / 0.005)

    vels = {
        "pelvis": _peak(0.6, 300.0),
        "thorax": _peak(0.7, 600.0),
        "lead_arm": _peak(0.8, 900.0),
        "club": _peak(0.9, 1800.0),
    }

    res = compute_cheetham_kinematic_sequence(
        vels, t, impact_time_s=0.95, velocity_component=VelocityComponentKind.axial
    )
    assert (
        res.velocity_component == VelocityComponentKind.axial and res.is_valid_sequence
    )
    assert res.filter_descriptor.method == "butterworth_4th_order_zero_lag"
    assert res.sequence_order == ["pelvis", "thorax", "lead_arm", "club"]
    assert res.peaks["pelvis"].peak_time_rel_impact_s == pytest.approx(-0.35, abs=0.01)
    assert res.peaks["club"].peak_time_rel_impact_s == pytest.approx(-0.05, abs=0.01)

    gains = {f"{x.proximal_segment}->{x.distal_segment}": x for x in res.gains}
    assert gains["pelvis->thorax"].speed_gain_diff_deg_s == pytest.approx(
        300.0, abs=1.0
    )
    assert gains["pelvis->thorax"].speed_gain_ratio == pytest.approx(2.0, rel=0.05)
    assert gains["thorax->lead_arm"].speed_gain_diff_deg_s == pytest.approx(
        300.0, abs=1.0
    )
    assert gains["lead_arm->club"].speed_gain_diff_deg_s == pytest.approx(
        900.0, abs=1.0
    )


def test_no_frame_index_fallback_missing_time_raises() -> None:
    vels = {"pelvis": np.array([10.0, 20.0, 30.0])}
    with pytest.raises(ValueError, match="Missing timestamps"):
        compute_cheetham_kinematic_sequence(vels, None, impact_time_s=1.0)

    from src.api.models.requests import AnalysisRequest
    from src.api.services.analysis_service import AnalysisService

    svc = AnalysisService(MagicMock())
    req = AnalysisRequest(
        analysis_type="swing_sequence", data_source="simulation", parameters={}, data={}
    )
    with pytest.raises(ValueError, match="Missing timestamps"):
        svc._resolve_sequence_times(req, vels, {})


def test_sampling_rate_invariance() -> None:
    t_120, t_240 = np.linspace(0.2, 0.7, 61), np.linspace(0.2, 0.7, 121)
    r120 = compute_cheetham_xfactor_stretch(
        30.0 + 15.0 * np.sin(3.0 * np.pi * (t_120 - 0.2)), t_120, top_idx=0
    )
    r240 = compute_cheetham_xfactor_stretch(
        30.0 + 15.0 * np.sin(3.0 * np.pi * (t_240 - 0.2)), t_240, top_idx=0
    )
    assert r120.stretch_deg == pytest.approx(r240.stretch_deg, abs=0.05)
    assert r120.stretch_pct == pytest.approx(r240.stretch_pct, abs=0.2)

    def _v_dict(t: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "pelvis": 350.0 * np.exp(-((t - 0.45) ** 2) / 0.004),
            "thorax": 700.0 * np.exp(-((t - 0.55) ** 2) / 0.004),
        }

    k120 = compute_cheetham_kinematic_sequence(
        _v_dict(t_120), t_120, impact_time_s=0.65
    )
    k240 = compute_cheetham_kinematic_sequence(
        _v_dict(t_240), t_240, impact_time_s=0.65
    )
    assert k120.peaks["pelvis"].peak_time_rel_impact_s == pytest.approx(
        k240.peaks["pelvis"].peak_time_rel_impact_s, abs=0.01
    )
    assert k120.gains[0].speed_gain_diff_deg_s == pytest.approx(
        k240.gains[0].speed_gain_diff_deg_s, abs=0.5
    )


def test_metric_registry_entries_valid() -> None:
    assert XFACTOR_STRETCH_CHEETHAM_ENTRY.metric_id == "x_factor.stretch.cheetham_2001"
    assert "Cheetham et al. (2001)" in XFACTOR_STRETCH_CHEETHAM_ENTRY.cites
    assert (
        KINEMATIC_SEQUENCE_CHEETHAM_ENTRY.metric_id
        == "kinematic_sequence.cheetham_2008"
    )
    assert "Cheetham et al. (2008)" in KINEMATIC_SEQUENCE_CHEETHAM_ENTRY.cites
