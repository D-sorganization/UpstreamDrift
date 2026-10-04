"""Tests for the drift-anchored match data-quality report.

Software behaviour only: verdicts here are not capture or engine
qualification.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.estimation.drift_anchored_matcher import (
    DriftAnchoredMatchResult,
    SampleLabel,
    WindowRecord,
    match_kinematics,
)
from src.shared.python.estimation.drift_prediction import ControlBand
from src.shared.python.estimation.local_torque_window import WindowOptions
from src.shared.python.estimation.matching_quality import (
    MatchQualityReport,
    QualityThresholds,
    SuspectInterval,
    report_to_dict,
    summarize_match_quality,
)
from src.shared.python.estimation.drift_anchored_matcher import MatchOptions
from src.shared.python.estimation.synthetic_swing import (
    corrupt_observations,
    simulate_swing,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend

pytestmark = pytest.mark.unit

A = SampleLabel.ACCEPTED
O = SampleLabel.OUTLIER  # noqa: E741
U = SampleLabel.UNEXPLAINED
G = SampleLabel.GAP_FILLED


def _window(
    success: bool = True, saturated: bool = False, rate_limited: bool = False
) -> WindowRecord:
    return WindowRecord(
        start=0,
        steps=4,
        success=success,
        status="ok",
        chi2_per_dof=1.0,
        raw_rms_residual=0.0,
        explained_fraction=1.0,
        saturated=np.array([saturated, False]),
        rate_limited=np.array([rate_limited, False]),
        torque_condition_number=1.0,
    )


def _result(
    labels: list[SampleLabel],
    *,
    tau: np.ndarray | None = None,
    tau_std: np.ndarray | None = None,
    disagreement: np.ndarray | None = None,
    replay: np.ndarray | None = None,
    resid: np.ndarray | None = None,
    divergence: np.ndarray | None = None,
    dominance: np.ndarray | None = None,
    windows: tuple[WindowRecord, ...] | None = None,
) -> DriftAnchoredMatchResult:
    n = len(labels)
    z2 = np.zeros((n, 2))
    z1 = np.zeros(n)
    lab = np.empty(n, dtype=object)
    lab[:] = labels
    return DriftAnchoredMatchResult(
        t=0.01 * np.arange(n),
        labels=lab,
        q_estimate=z2,
        v_estimate=z2,
        tau=np.ones((n, 2)) if tau is None else tau,
        tau_std=z2 if tau_std is None else tau_std,
        tau_disagreement=z2 if disagreement is None else disagreement,
        q_replay=z2,
        v_replay=z2,
        normalized_residual=z1 if resid is None else resid,
        replay_residual=z1 if replay is None else replay,
        ztcf_divergence=z1 if divergence is None else divergence,
        explained_fraction=np.ones(n),
        rejection_rate=z1,
        drift_dominance=np.full(n, 0.5) if dominance is None else dominance,
        windows=(_window(),) if windows is None else windows,
    )


class TestIntervals:
    def test_runs_and_boundaries(self) -> None:
        r = summarize_match_quality(_result([G, G, A, A, O, A, U, U]))
        iv = r.suspect_intervals
        assert [(i.start, i.stop, i.label) for i in iv] == [
            (0, 2, "gap_filled"),
            (4, 5, "outlier"),
            (6, 8, "unexplained"),
        ]
        assert iv[0].t_start == pytest.approx(0.0)
        assert iv[0].t_stop == pytest.approx(0.01)
        assert iv[2].t_stop == pytest.approx(0.07)

    def test_adjacent_different_labels_split(self) -> None:
        r = summarize_match_quality(_result([A, O, O, U, G, G]))
        assert [(i.start, i.stop, i.label) for i in r.suspect_intervals] == [
            (1, 3, "outlier"),
            (3, 4, "unexplained"),
            (4, 6, "gap_filled"),
        ]

    def test_no_suspects(self) -> None:
        r = summarize_match_quality(_result([A, A, A]))
        assert r.suspect_intervals == ()
        assert r.longest_gap_run == 0 and r.longest_unexplained_run == 0

    def test_longest_runs(self) -> None:
        r = summarize_match_quality(_result([U, A, U, U, U, G, G, A, G]))
        assert r.longest_unexplained_run == 3
        assert r.longest_gap_run == 2

    def test_interval_stats_and_nan(self) -> None:
        div = np.array([0.0, np.nan, np.nan, 2.0, 5.0, 0.0])
        dis = np.zeros((6, 2))
        dis[1:3] = [[1.0, 3.0], [2.0, 2.0]]
        r = summarize_match_quality(
            _result([A, G, G, A, A, A], divergence=div, disagreement=dis)
        )
        (iv,) = r.suspect_intervals
        assert np.isnan(iv.peak_ztcf_divergence)
        assert iv.mean_tau_disagreement == pytest.approx(2.0)
        r2 = summarize_match_quality(
            _result([A, A, A, O, O, A], divergence=div, disagreement=dis)
        )
        assert r2.suspect_intervals[0].peak_ztcf_divergence == pytest.approx(5.0)


class TestFractionsAndRms:
    def test_fractions(self) -> None:
        r = summarize_match_quality(_result([A, A, O, U, G]))
        assert r.n_samples == 5
        assert r.accepted_fraction == pytest.approx(0.4)
        assert r.outlier_fraction == pytest.approx(0.2)
        assert r.unexplained_fraction == pytest.approx(0.2)
        assert r.gap_fraction == pytest.approx(0.2)

    def test_rms_over_accepted_only_ignores_nan(self) -> None:
        replay = np.array([3.0, 4.0, 100.0, np.nan])
        resid = np.array([1.0, 1.0, 100.0, np.nan])
        r = summarize_match_quality(_result([A, A, O, G], replay=replay, resid=resid))
        assert r.replay_rms == pytest.approx(np.sqrt((9 + 16) / 2))
        assert r.estimate_rms == pytest.approx(1.0)

    def test_rms_nan_when_nothing_accepted(self) -> None:
        r = summarize_match_quality(_result([G, G]))
        assert np.isnan(r.replay_rms) and np.isnan(r.estimate_rms)

    def test_per_channel_percentiles(self) -> None:
        n = 5
        dis = np.tile(np.array([[1.0, 2.0]]), (n, 1))
        tau = np.tile(np.array([[10.0, -20.0]]), (n, 1))
        std = np.tile(np.array([[1.0, 4.0]]), (n, 1))
        r = summarize_match_quality(
            _result([A] * n, tau=tau, tau_std=std, disagreement=dis)
        )
        np.testing.assert_allclose(r.tau_disagreement_p95, [1.0, 2.0])
        np.testing.assert_allclose(r.relative_torque_uncertainty_p95, [0.1, 0.2])

    def test_window_fractions_and_dominance(self) -> None:
        ws = (
            _window(),
            _window(success=False),
            _window(saturated=True),
            _window(rate_limited=True),
        )
        r = summarize_match_quality(
            _result([A, A], windows=ws, dominance=np.array([0.2, 0.6]))
        )
        assert r.window_failure_fraction == pytest.approx(0.25)
        assert r.saturated_window_fraction == pytest.approx(0.25)
        assert r.rate_limited_window_fraction == pytest.approx(0.25)
        assert r.mean_drift_dominance == pytest.approx(0.4)

    def test_no_windows(self) -> None:
        r = summarize_match_quality(_result([A], windows=()))
        assert r.window_failure_fraction == 0.0


class TestVerdict:
    def test_qualified(self) -> None:
        r = summarize_match_quality(_result([A] * 10))
        assert r.verdict == "qualified-software-match"
        assert r.reasons == ()

    def test_unusable_by_unexplained(self) -> None:
        r = summarize_match_quality(_result([U] * 3 + [A] * 7))
        assert r.verdict == "unusable"
        assert any("unexplained_fraction" in s for s in r.reasons)

    def test_unusable_by_window_failures(self) -> None:
        ws = (_window(success=False), _window(success=False), _window())
        r = summarize_match_quality(_result([A] * 4, windows=ws))
        assert r.verdict == "unusable"
        assert any("window_failure_fraction" in s for s in r.reasons)

    def test_review_unexplained(self) -> None:
        r = summarize_match_quality(_result([U] + [A] * 9))  # 0.1
        assert r.verdict == "review"
        assert any("unexplained_fraction" in s for s in r.reasons)

    def test_review_gap(self) -> None:
        r = summarize_match_quality(_result([G] * 3 + [A] * 7))
        assert r.verdict == "review"
        assert any("gap_fraction" in s for s in r.reasons)

    def test_review_replay(self) -> None:
        r = summarize_match_quality(_result([A] * 4, replay=np.full(4, 6.0)))
        assert r.verdict == "review"
        assert any("replay_rms" in s for s in r.reasons)

    def test_review_window_failure(self) -> None:
        ws = (_window(success=False),) + (_window(),) * 4  # 0.2
        r = summarize_match_quality(_result([A] * 4, windows=ws))
        assert r.verdict == "review"
        assert any("window_failure_fraction" in s for s in r.reasons)

    def test_review_saturation(self) -> None:
        ws = (_window(saturated=True),) + (_window(),) * 2
        r = summarize_match_quality(_result([A] * 4, windows=ws))
        assert r.verdict == "review"
        assert any("saturated_window_fraction" in s for s in r.reasons)

    def test_multiple_reasons_and_custom_thresholds(self) -> None:
        res = _result([G] * 3 + [A] * 7, replay=np.full(10, 6.0))
        assert len(summarize_match_quality(res).reasons) == 2
        loose = QualityThresholds(max_gap_fraction=0.5, max_replay_rms=10.0)
        assert summarize_match_quality(res, loose).verdict == "qualified-software-match"

    def test_threshold_at_limit_is_not_exceeded(self) -> None:
        r = summarize_match_quality(_result([G] * 2 + [A] * 8))  # exactly 0.2
        assert r.verdict == "qualified-software-match"


class TestContracts:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_unexplained_fraction": 1.5},
            {"max_gap_fraction": -0.1},
            {"max_replay_rms": 0.0},
            {"max_window_failure_fraction": 2.0},
            {"max_saturated_window_fraction": -1.0},
            {"unusable_unexplained_fraction": 1.1},
        ],
    )
    def test_bad_thresholds_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            summarize_match_quality(_result([A, A]), QualityThresholds(**kwargs))

    def test_inconsistent_lengths_rejected(self) -> None:
        res = _result([A, A, A])
        bad = DriftAnchoredMatchResult(
            **{**res.__dict__, "replay_residual": np.zeros(2)}
        )
        with pytest.raises(ValueError):
            summarize_match_quality(bad)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            summarize_match_quality(_result([]))


class TestSerialisation:
    def test_json_round_trip_with_nan(self) -> None:
        res = _result(
            [A, G, G, A],
            replay=np.array([1.0, np.nan, np.nan, 1.0]),
            divergence=np.full(4, np.nan),
        )
        rep = summarize_match_quality(res)
        text = json.dumps(report_to_dict(rep), allow_nan=False)
        d = json.loads(text)
        assert d["verdict"] == rep.verdict
        assert d["suspect_intervals"][0]["peak_ztcf_divergence"] is None
        assert d["tau_disagreement_p95"] == [0.0, 0.0]
        assert d["n_samples"] == 4
        assert isinstance(d["reasons"], list)

    def test_nan_rms_serialises_none(self) -> None:
        d = report_to_dict(summarize_match_quality(_result([G, G])))
        assert d["replay_rms"] is None
        json.dumps(d, allow_nan=False)

    def test_report_type(self) -> None:
        rep = summarize_match_quality(_result([A]))
        assert isinstance(rep, MatchQualityReport)
        assert SuspectInterval.__dataclass_params__.frozen


BAND = ControlBand(
    lower=np.array([-300.0, -100.0]),
    upper=np.array([300.0, 100.0]),
    rate_limit=np.array([4000.0, 2000.0]),
)


def test_synthetic_swing_occlusion_is_reported_as_gap() -> None:
    noise = 0.002
    truth = simulate_swing()
    provider = make_backend("ode", GolfModelParams.default())
    obs = corrupt_observations(truth, noise_std=noise, seed=11)
    opts = MatchOptions(
        window=WindowOptions(sigma_obs=noise, sigma_q0=noise, sigma_v0=3.0, n_knots=2),
        window_steps=12,
        stride=4,
        carry_sigma_q=noise,
        carry_sigma_v=0.2,
    )
    result = match_kinematics(provider, truth.t, obs.q_observed, obs.mask, BAND, opts)
    report = summarize_match_quality(result)
    occ = obs.occluded_indices
    assert occ.size > 0
    gaps = [i for i in report.suspect_intervals if i.label == "gap_filled"]
    assert any(i.start <= occ[0] and i.stop > occ[-1] for i in gaps)
    covered = np.zeros(truth.t.size, dtype=bool)
    for i in gaps:
        covered[i.start : i.stop] = True
    assert covered[occ].all()
    assert report.verdict != "unusable"
