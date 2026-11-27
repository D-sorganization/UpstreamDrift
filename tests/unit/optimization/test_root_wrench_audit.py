"""WP-0a (#15808): 6-component root-wrench residual audit contract."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.optimization.root_wrench_audit import (
    COMPONENTS,
    audit_root_wrench,
)

N = 200
T = np.linspace(0.0, 1.0, N)
BW_N = 800.0
HEIGHT_M = 1.8
IMPACT_S = 0.8


def _audit(wrench, causes=None, **kw):
    """Run the audit on the shared time base with optional overrides."""
    return audit_root_wrench(
        times_s=T,
        root_wrench=wrench,
        body_weight_n=BW_N,
        height_m=HEIGHT_M,
        impact_time_s=IMPACT_S,
        impact_window_s=0.05,
        causes=causes or {},
        **kw,
    )


def test_components_are_six_force_then_moment() -> None:
    """The residual is force then moment, in axis order."""
    assert COMPONENTS == ("fx", "fy", "fz", "mx", "my", "mz")


def test_zero_residual_passes_all_components() -> None:
    """A zero wrench is admitted with no refit."""
    report = _audit(np.zeros((N, 6)))
    assert report.admitted
    assert all(c.passes for c in report.components)
    assert report.recommendation == "none"


def test_force_gate_is_two_percent_body_weight_outside_impact() -> None:
    """Force components fail above 2% body weight and pass below it."""
    w = np.zeros((N, 6))
    w[:, 2] = 0.03 * BW_N  # 3% BW vertical force: fails
    w[:, 0] = 0.01 * BW_N  # 1% BW: passes
    report = _audit(w)
    by = {c.name: c for c in report.components}
    assert not by["fz"].passes and by["fx"].passes
    assert by["fz"].max_normalized == pytest.approx(0.03)
    assert not report.admitted


def test_moment_gate_normalised_by_bw_times_height() -> None:
    """Moment components are gated by 2% of body weight times height."""
    w = np.zeros((N, 6))
    w[:, 5] = 0.019 * BW_N * HEIGHT_M
    assert _audit(w).components[5].passes
    w[:, 5] = 0.021 * BW_N * HEIGHT_M
    assert not _audit(w).components[5].passes


def test_impact_window_excluded_from_gate_but_reported() -> None:
    """Impact-window spikes are reported and excluded from the gate."""
    w = np.zeros((N, 6))
    w[np.abs(T - IMPACT_S) <= 0.05, 1] = 5.0 * BW_N  # huge spike at impact only
    report = _audit(w)
    fy = report.components[1]
    assert fy.passes
    assert fy.impact_peak_normalized == pytest.approx(5.0)
    assert fy.n_gated_frames < N


def test_dominant_cause_attributed_per_component() -> None:
    """The cause that matches a failing component is the refit target."""
    slip = np.sin(np.linspace(0, 6 * np.pi, N))
    w = np.zeros((N, 6))
    w[:, 0] = 0.05 * BW_N * slip
    rng = np.random.default_rng(0)
    report = _audit(
        w,
        causes={"foot_slip": slip, "grip_weld_gap": rng.standard_normal(N)},
    )
    fx = report.components[0]
    assert fx.dominant_cause == "foot_slip"
    assert fx.cause_correlation["foot_slip"] == pytest.approx(1.0, abs=1e-9)
    assert report.recommendation == "refit:foot_slip"


def test_unattributed_when_no_cause_correlates() -> None:
    """A failing component with no matching cause stays unattributed."""
    rng = np.random.default_rng(1)
    w = np.zeros((N, 6))
    w[:, 3] = 0.1 * BW_N * HEIGHT_M * rng.standard_normal(N)
    report = _audit(w, causes={"foot_slip": rng.standard_normal(N)})
    assert report.components[3].dominant_cause == "unattributed"
    assert report.recommendation == "refit:unattributed"


def test_constant_cause_has_zero_correlation_not_nan() -> None:
    """A constant cause has correlation 0, not NaN."""
    w = np.zeros((N, 6))
    w[:, 0] = np.arange(N, dtype=float)
    report = _audit(w, causes={"foot_slip": np.ones(N)})
    assert report.components[0].cause_correlation["foot_slip"] == 0.0


def test_to_dict_is_json_serialisable_and_machine_readable() -> None:
    """The report dict round-trips through JSON with the schema gates."""
    report = _audit(np.zeros((N, 6)), causes={"foot_slip": np.zeros(N)})
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["schema"] == "root_wrench_audit/v1"
    assert [c["name"] for c in payload["components"]] == list(COMPONENTS)
    assert payload["gates"] == {"force_bw": 0.02, "moment_bw_height": 0.02}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"root_wrench": np.zeros((N, 5))}, "shape"),
        ({"root_wrench": np.full((N, 6), np.nan)}, "finite"),
        ({"body_weight_n": 0.0}, "body_weight_n"),
        ({"height_m": -1.0}, "height_m"),
        ({"impact_window_s": -0.1}, "impact_window_s"),
        ({"causes": {"x": np.zeros(N - 1)}}, "causes"),
        ({"times_s": T[::-1]}, "increasing"),
    ],
)
def test_preconditions_rejected(kwargs, match) -> None:
    """Malformed inputs raise ValueError before a verdict is produced."""
    base = {
        "times_s": T,
        "root_wrench": np.zeros((N, 6)),
        "body_weight_n": BW_N,
        "height_m": HEIGHT_M,
        "impact_time_s": IMPACT_S,
        "impact_window_s": 0.05,
        "causes": {},
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        audit_root_wrench(**base)


def test_all_frames_in_impact_window_is_rejected() -> None:
    """An impact window that covers every frame is not auditable."""
    with pytest.raises(ValueError, match="no frames"):
        audit_root_wrench(
            times_s=T,
            root_wrench=np.zeros((N, 6)),
            body_weight_n=BW_N,
            height_m=HEIGHT_M,
            impact_time_s=0.5,
            impact_window_s=10.0,
            causes={},
        )
