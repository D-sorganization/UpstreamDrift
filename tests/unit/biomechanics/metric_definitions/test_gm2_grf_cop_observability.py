"""Unit tests for GM-2: GRF and COP observability, peaks and normalisation (#14425)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.biomechanics.metric_definitions.grf_cop_observability import (
    STANDARD_GRAVITY_M_S2,
    GRFChannelObservability,
    GRFSourceAttribution,
    compute_cop_target_kinematics,
    compute_event_anchored_peaks,
    compute_lead_fraction_crossing,
    format_channel_value,
    normalize_vertical_force,
    sanitize_channel_value,
    validate_unobserved_channel,
)
from src.shared.python.biomechanics.metric_definitions.schema import (
    NormalisationKind,
    SourceKind,
)
from src.shared.python.contracts import ContractViolationError

pytestmark = pytest.mark.unit


def test_unobserved_channel_contracts() -> None:
    obs = GRFChannelObservability()
    assert not obs.shear_observed and not obs.free_moment_observed
    assert obs.vertical_force_observed and obs.cop_observed
    assert validate_unobserved_channel(float("nan"), observed=False)
    assert validate_unobserved_channel(None, observed=False)
    with pytest.raises(ContractViolationError):
        validate_unobserved_channel(0.0, observed=False)
    with pytest.raises(ContractViolationError):
        validate_unobserved_channel(12.3, observed=False)
    assert format_channel_value(float("nan"), observed=False) == "NaN"
    assert format_channel_value(0.0, observed=False) == "NaN"
    assert format_channel_value(10.5, observed=True) == "10.50 N"
    assert math.isnan(sanitize_channel_value(0.0, observed=False))
    assert sanitize_channel_value(42.0, observed=True) == 42.0


def test_source_attribution_contracts() -> None:
    for eng in ["mujoco", "drake", "opensim"]:
        assert (
            GRFSourceAttribution(source=SourceKind.modelled, engine=eng).engine == eng
        )
    with pytest.raises(ContractViolationError):
        GRFSourceAttribution(source=SourceKind.modelled, engine=None)
    with pytest.raises(ContractViolationError):
        GRFSourceAttribution(source=SourceKind.modelled, engine="unknown")
    assert GRFSourceAttribution(source=SourceKind.measured, engine=None).engine is None
    with pytest.raises(ContractViolationError):
        GRFSourceAttribution(source=SourceKind.measured, engine="mujoco")


def test_normalisation_and_nan_propagation() -> None:
    bw = 80.0 * STANDARD_GRAVITY_M_S2
    scaled = normalize_vertical_force([0.0, bw, 2.0 * bw, np.nan], body_mass_kg=80.0)
    assert pytest.approx(scaled[0]) == 0.0 and pytest.approx(scaled[1]) == 100.0
    assert pytest.approx(scaled[2]) == 200.0 and math.isnan(scaled[3])
    raw = normalize_vertical_force(
        [150.0, np.nan], body_mass_kg=80.0, normalisation=NormalisationKind.none
    )
    assert pytest.approx(raw[0]) == 150.0 and math.isnan(raw[1])
    with pytest.raises(ContractViolationError):
        normalize_vertical_force([100.0], body_mass_kg=0.0)


def test_event_anchored_peaks() -> None:
    times = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    bw = 80.0 * STANDARD_GRAVITY_M_S2
    fz_l = np.array([0.2, 0.3, 0.5, 1.0, 1.5, 1.2, 0.8]) * bw
    fz_t = np.array([0.9, 1.2, 1.0, 0.6, 0.3, 0.1, 0.0]) * bw
    p = compute_event_anchored_peaks(
        times, fz_l, fz_t, t_top_s=1.0, t_impact_s=1.3, body_mass_kg=80.0
    )
    assert (
        pytest.approx(p.lead.peak_value) == 150.0
        and pytest.approx(p.lead.peak_time_s) == 1.2
    )
    assert (
        pytest.approx(p.lead.dt_top_s) == 0.2
        and pytest.approx(p.lead.dt_impact_s) == -0.1
    )
    assert (
        pytest.approx(p.trail.peak_value) == 120.0
        and pytest.approx(p.trail.peak_time_s) == 0.9
    )
    assert (
        pytest.approx(p.trail.dt_top_s) == -0.1
        and pytest.approx(p.trail.dt_impact_s) == -0.4
    )
    p_nan = compute_event_anchored_peaks(
        [0.0, 1.0],
        [np.nan, np.nan],
        [np.nan, np.nan],
        t_top_s=0.5,
        t_impact_s=0.8,
        body_mass_kg=75.0,
    )
    assert math.isnan(p_nan.lead.peak_value) and math.isnan(p_nan.lead.dt_top_s)


def test_cop_target_kinematics_and_crossing() -> None:
    times = np.array([0.0, 0.5, 1.0])
    kin = compute_cop_target_kinematics(
        times, [[0.0, 0.0], [0.1, 0.0], [0.25, 0.0]], target_direction=(1.0, 0.0, 0.0)
    )
    assert (
        pytest.approx(kin.range_m) == 0.25
        and pytest.approx(kin.peak_velocity_m_s, abs=1e-3) == 0.3
    )
    assert kin.mean_velocity_m_s > 0.0
    kin_nan = compute_cop_target_kinematics(times, [np.nan, np.nan, np.nan])
    assert math.isnan(kin_nan.range_m) and math.isnan(kin_nan.peak_velocity_m_s)

    times_cross = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fz_l = np.array([200.0, 300.0, 400.0, 600.0, 800.0, 900.0])
    fz_t = np.array([800.0, 700.0, 600.0, 400.0, 200.0, 100.0])
    c = compute_lead_fraction_crossing(
        times_cross, fz_l, fz_t, t_top_s=0.4, t_impact_s=0.8, threshold=0.5
    )
    assert c.crossed is True and pytest.approx(c.crossing_time_s) == 0.5
    assert pytest.approx(c.dt_top_s) == 0.1 and pytest.approx(c.dt_impact_s) == -0.3
    no_c = compute_lead_fraction_crossing(
        [0.0, 0.5, 1.0],
        [100.0, 100.0, 100.0],
        [900.0, 900.0, 900.0],
        t_top_s=0.4,
        t_impact_s=0.8,
    )
    assert (
        no_c.crossed is False
        and math.isnan(no_c.crossing_time_s)
        and math.isnan(no_c.dt_top_s)
    )
