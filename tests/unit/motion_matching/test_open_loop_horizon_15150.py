"""Open-loop replay horizon backward from impact (#15150 D2).

A synthetic replay callable with a known divergence curve stands in for the
engine: the error grows with the lead time between the replay start and
impact, so the right horizon is known in closed form.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.shared.python.motion_matching import open_loop_horizon as olh

pytestmark = pytest.mark.unit

IMPACT_S = 1.0
REFERENCE = olh.ImpactDelivery(face_angle_deg=2.0, path_deg=-3.0, head_speed_mps=45.0)
# Binary-exact values so tolerance edges are not blurred by round-off.
SETTINGS = olh.HorizonSettings(
    face_tolerance_deg=0.5,
    path_tolerance_deg=0.5,
    head_speed_tolerance_mps=0.25,
    step_s=0.0625,
    max_horizon_s=0.5,
)


def _divergent(rate_per_s: float, field: str = "face_angle_deg"):
    """Replay whose ``field`` error is ``rate * lead`` (linear divergence)."""

    def replay(start_s: float) -> olh.ImpactDelivery:
        error = rate_per_s * (IMPACT_S - start_s)
        return replace(REFERENCE, **{field: getattr(REFERENCE, field) + error})

    return replay


def test_finds_the_horizon_of_a_known_divergence_curve() -> None:
    # Face error 2 deg/s: 0.5 deg is reached at a lead of 0.25 s exactly.
    result = olh.search_horizon(
        _divergent(2.0), REFERENCE, impact_s=IMPACT_S, earliest_s=0.0, settings=SETTINGS
    )
    assert result.status == "accepted"
    assert result.contiguous
    assert result.horizon_s == pytest.approx(0.25)
    assert result.earliest_start_s == pytest.approx(0.75)
    assert len(result.candidates) == 8
    leads = [c.lead_s for c in result.candidates]
    assert leads == sorted(leads)
    assert [c.passed for c in result.candidates] == [True] * 4 + [False] * 4
    assert result.candidates[-1].face_error_deg == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "rate", "horizon"),
    [
        ("path_deg", 4.0, 0.125),
        ("head_speed_mps", -1.0, 0.25),
    ],
)
def test_each_metric_bounds_the_horizon(field: str, rate: float, horizon: float):
    result = olh.search_horizon(
        _divergent(rate, field),
        REFERENCE,
        impact_s=IMPACT_S,
        earliest_s=0.0,
        settings=SETTINGS,
    )
    assert result.horizon_s == pytest.approx(horizon)


def test_error_exactly_at_tolerance_passes_and_just_above_fails() -> None:
    def replay(start_s: float) -> olh.ImpactDelivery:
        lead = IMPACT_S - start_s
        bump = 0.5 if lead <= 0.1875 else 0.5 + 2.0**-20
        return replace(REFERENCE, face_angle_deg=REFERENCE.face_angle_deg - bump)

    result = olh.search_horizon(
        replay, REFERENCE, impact_s=IMPACT_S, earliest_s=0.0, settings=SETTINGS
    )
    assert result.horizon_s == pytest.approx(0.1875)
    assert result.candidates[2].face_error_deg == 0.5


def test_face_error_wraps_across_180_degrees() -> None:
    reference = replace(REFERENCE, face_angle_deg=179.875)

    def replay(_start_s: float) -> olh.ImpactDelivery:
        return replace(reference, face_angle_deg=-179.875)

    result = olh.search_horizon(
        replay, reference, impact_s=IMPACT_S, earliest_s=0.0, settings=SETTINGS
    )
    assert result.candidates[0].face_error_deg == pytest.approx(0.25)
    assert result.horizon_s == pytest.approx(0.5)


def test_no_candidate_passes() -> None:
    result = olh.search_horizon(
        _divergent(100.0),
        REFERENCE,
        impact_s=IMPACT_S,
        earliest_s=0.0,
        settings=SETTINGS,
    )
    assert result.status == "none_accepted"
    assert result.horizon_s == 0.0
    assert result.earliest_start_s is None


def test_non_contiguous_window_is_reported_not_picked() -> None:
    # Fails at a lead of 0.1875 s only, passes either side of it.
    def replay(start_s: float) -> olh.ImpactDelivery:
        bad = abs((IMPACT_S - start_s) - 0.1875) < 1e-9
        return replace(REFERENCE, face_angle_deg=REFERENCE.face_angle_deg + bad)

    result = olh.search_horizon(
        replay, REFERENCE, impact_s=IMPACT_S, earliest_s=0.0, settings=SETTINGS
    )
    assert result.status == "non_contiguous"
    assert not result.contiguous
    assert result.earliest_start_s is None
    assert result.horizon_s == pytest.approx(0.125)
    assert result.passes_after_failure_s == pytest.approx(
        (0.25, 0.3125, 0.375, 0.4375, 0.5)
    )
    summary = result.summary()
    json.dumps(summary)
    assert summary["status"] == "non_contiguous"
    assert len(summary["candidates"]) == 8


def test_candidates_stop_at_the_earliest_available_start() -> None:
    result = olh.search_horizon(
        _divergent(0.0), REFERENCE, impact_s=IMPACT_S, earliest_s=0.8, settings=SETTINGS
    )
    assert [c.lead_s for c in result.candidates] == pytest.approx(
        [0.0625, 0.125, 0.1875]
    )
    assert result.horizon_s == pytest.approx(0.1875)


def test_contract_errors() -> None:
    kwargs = {"impact_s": IMPACT_S, "earliest_s": 0.0, "settings": SETTINGS}
    with pytest.raises(TypeError, match="callable"):
        olh.search_horizon("not callable", REFERENCE, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ImpactDelivery"):
        olh.search_horizon(_divergent(1.0), (1, 2, 3), **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="before impact"):
        olh.search_horizon(
            _divergent(1.0), REFERENCE, impact_s=1.0, earliest_s=1.0, settings=SETTINGS
        )
    with pytest.raises(ValueError, match="before impact"):
        olh.search_horizon(
            _divergent(1.0), REFERENCE, impact_s=0.96, earliest_s=0.9, settings=SETTINGS
        )
    with pytest.raises(TypeError, match="ImpactDelivery"):
        olh.search_horizon(lambda _t: None, REFERENCE, **kwargs)  # type: ignore[arg-type,return-value]
    with pytest.raises(ValueError, match="finite"):
        olh.ImpactDelivery(
            face_angle_deg=float("nan"), path_deg=0.0, head_speed_mps=1.0
        )
    with pytest.raises(ValueError, match="step_s"):
        replace(SETTINGS, step_s=0.0)
    with pytest.raises(ValueError, match="max_horizon_s"):
        replace(SETTINGS, max_horizon_s=0.03)


def test_settings_file_loads_and_rejects_missing_fields(tmp_path) -> None:
    settings = olh.load_settings()
    assert settings.step_s < settings.max_horizon_s
    bad = tmp_path / "s.json"
    bad.write_text(json.dumps({"step_s": 0.01}), encoding="utf-8")
    with pytest.raises(ValueError, match="miss"):
        olh.load_settings(bad)


# --- diverged replays count as failed candidates ------------------------------


def _diverges_beyond(lead_limit_s: float):
    """Exact replay that diverges for every lead past ``lead_limit_s``."""

    def replay(start_s: float) -> olh.ImpactDelivery:
        if IMPACT_S - start_s > lead_limit_s:
            raise olh.ReplayDiverged(f"nonfinite state from {start_s} s")
        return REFERENCE

    return replay


def test_diverged_replay_is_a_failed_candidate_with_a_reason() -> None:
    result = olh.search_horizon(
        _diverges_beyond(0.25),
        REFERENCE,
        impact_s=IMPACT_S,
        earliest_s=0.0,
        settings=SETTINGS,
    )
    assert result.status == "accepted"
    assert result.horizon_s == pytest.approx(0.25)
    assert len(result.candidates) == 8
    diverged = result.candidates[4:]
    assert not any(c.passed for c in diverged)
    assert all("nonfinite state" in (c.failure or "") for c in diverged)
    assert all(c.face_error_deg is None for c in diverged)
    assert all(c.failure is None for c in result.candidates[:4])
    summary = result.summary()
    json.dumps(summary, allow_nan=False)
    assert summary["diverged_leads_s"] == [c.lead_s for c in diverged]


def test_divergence_after_a_failure_does_not_break_the_contiguity_check() -> None:
    def replay(start_s: float) -> olh.ImpactDelivery:
        lead = IMPACT_S - start_s
        if lead > 0.375:
            raise olh.ReplayDiverged("integrator error")
        bump = 0.0 if lead <= 0.125 else 1.0
        return replace(REFERENCE, path_deg=REFERENCE.path_deg + bump)

    result = olh.search_horizon(
        replay, REFERENCE, impact_s=IMPACT_S, earliest_s=0.0, settings=SETTINGS
    )
    assert result.status == "accepted"
    assert result.horizon_s == pytest.approx(0.125)


def test_other_replay_errors_still_propagate() -> None:
    def replay(start_s: float) -> olh.ImpactDelivery:
        raise ValueError("off the step grid")

    with pytest.raises(ValueError, match="grid"):
        olh.search_horizon(
            replay, REFERENCE, impact_s=IMPACT_S, earliest_s=0.0, settings=SETTINGS
        )
