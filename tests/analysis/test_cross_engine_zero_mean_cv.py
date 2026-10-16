"""Near-zero-mean CV is undefined, never "perfectly robust" (#12807).

Contract under test (NSX epic #12593 §5A, mirrors NSX-82 / #12711):

- ``_safe_cv`` returns NaN when ``|mean| <= CV_NEAR_ZERO_MEAN``; it used to
  return 0.0, which ``robustness_score`` turned into a perfect 1.0.
- ``robustness_score`` fails closed: any undefined CV makes the score NaN.
- ``run_cross_engine_study`` stays JSON-safe: an undefined ``cv`` or
  ``robustness_score`` is ``None`` with a ``cv_reason`` /
  ``robustness_reason`` string, and is never reported as robust.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from src.shared.python.analysis import cross_engine as ce
from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
    CrossEngineRunResult,
    CrossEngineSimConfig,
)

pytestmark = pytest.mark.unit

_FAST_CONFIG = CrossEngineSimConfig(t_end=0.1, dt=0.01, n_trials=2, seed=1)


def _result(name: str, energy_mean: float) -> CrossEngineRunResult:
    """Engine result whose energy metric has the given mean (std 0.3)."""
    return CrossEngineRunResult(
        engine_name=name,
        mean_total_energy_final=energy_mean,
        std_total_energy_final=0.3,
        mean_end_effector_speed_final=1.0,
        std_end_effector_speed_final=0.1,
        mean_peak_end_effector_speed=2.0,
        std_peak_end_effector_speed=0.2,
    )


class TestSafeCv:
    @pytest.mark.parametrize("mean", [0.0, 1e-13, -1e-13, 1e-12, -1e-12])
    def test_near_zero_mean_is_undefined(self, mean: float) -> None:
        assert math.isnan(ce._safe_cv(mean, 0.3))

    def test_well_defined_mean_unchanged(self) -> None:
        assert ce._safe_cv(-2.0, 0.5) == pytest.approx(0.25)

    def test_reason_is_documented(self) -> None:
        assert "undefined" in ce.CV_UNDEFINED_REASON
        assert ce.CV_NEAR_ZERO_MEAN == 1e-12


class TestRobustnessScorePolicy:
    def test_undefined_cv_is_not_scored_robust(self) -> None:
        assert math.isnan(ce.robustness_score([float("nan")]))

    def test_any_undefined_cv_fails_closed(self) -> None:
        assert math.isnan(ce.robustness_score([0.1, float("nan"), 0.2]))

    def test_defined_values_still_clamped(self) -> None:
        assert ce.robustness_score([0.1, 0.3]) == pytest.approx(0.8)
        assert ce.robustness_score([3.0]) == 0.0

    def test_per_engine_robustness_undefined_for_zero_mean_metric(self) -> None:
        scores = ce.per_engine_robustness(
            {"zero": _result("zero", 0.0), "ok": _result("ok", 1.0)}
        )
        assert math.isnan(scores["zero"])
        assert 0.0 <= scores["ok"] <= 1.0


class TestStudyPayload:
    @pytest.fixture()
    def payload(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        results = {"zero": _result("zero", 0.0), "ok": _result("ok", 1.0)}
        monkeypatch.setattr(ce, "ENGINE_NAMES", ("zero", "ok"))
        monkeypatch.setattr(
            ce,
            "run_comparison_with_provenance",
            lambda _names, _cfg: (
                results,
                {"cv_total_energy_final": 0.1},
                {"zero": ce.BACKEND_STUB, "ok": ce.BACKEND_STUB},
            ),
        )
        return ce.run_cross_engine_study(["zero", "ok"], _FAST_CONFIG)

    def test_payload_is_strict_json(self, payload: dict[str, Any]) -> None:
        json.dumps(payload, allow_nan=False)

    def test_zero_mean_metric_declared_undefined(self, payload: dict[str, Any]) -> None:
        metric = payload["engines"]["zero"]["metrics"]["total_energy_final"]
        assert metric["cv"] is None
        assert metric["robustness_score"] is None
        assert metric["cv_reason"] == ce.CV_UNDEFINED_REASON

    def test_engine_score_undefined_not_robust(self, payload: dict[str, Any]) -> None:
        engine = payload["engines"]["zero"]
        assert engine["robustness_score"] is None
        assert "total_energy_final" in engine["robustness_reason"]

    def test_defined_metrics_carry_no_reason(self, payload: dict[str, Any]) -> None:
        engine = payload["engines"]["ok"]
        assert engine["robustness_reason"] is None
        for metric in engine["metrics"].values():
            assert metric["cv_reason"] is None
            assert 0.0 <= metric["robustness_score"] <= 1.0
