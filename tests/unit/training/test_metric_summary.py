"""Tests for :mod:`training.metric_summary`."""

from __future__ import annotations

import pytest

from training import MetricKind, TrainingMetric
from training.metric_summary import (
    RollingMean,
    best_per_metric,
    filter_by_tags,
    summarize_by_kind,
)

pytestmark = pytest.mark.unit


def _m(
    name: str,
    value: float,
    *,
    step: int = 0,
    kind: MetricKind = MetricKind.SCALAR,
    tags: dict[str, str] | None = None,
    timestamp: float = 0.0,
) -> TrainingMetric:
    return TrainingMetric(
        name=name,
        value=value,
        step=step,
        timestamp=timestamp,
        kind=kind,
        tags=tags or {},
    )


class TestBestPerMetric:
    def test_empty(self) -> None:
        assert best_per_metric([]) == {}

    def test_loss_picks_minimum(self) -> None:
        metrics = [
            _m("loss", 1.0, step=0, kind=MetricKind.LOSS),
            _m("loss", 0.1, step=1, kind=MetricKind.LOSS),
            _m("loss", 0.5, step=2, kind=MetricKind.LOSS),
        ]
        best = best_per_metric(metrics)
        assert best["loss"].value == 0.1
        assert best["loss"].step == 1

    def test_reward_picks_maximum(self) -> None:
        metrics = [
            _m("reward", 10.0, step=0, kind=MetricKind.REWARD),
            _m("reward", 50.0, step=1, kind=MetricKind.REWARD),
            _m("reward", 25.0, step=2, kind=MetricKind.REWARD),
        ]
        best = best_per_metric(metrics)
        assert best["reward"].value == 50.0

    def test_accuracy_picks_maximum(self) -> None:
        metrics = [
            _m("acc", 0.5, step=0, kind=MetricKind.ACCURACY),
            _m("acc", 0.9, step=1, kind=MetricKind.ACCURACY),
            _m("acc", 0.8, step=2, kind=MetricKind.ACCURACY),
        ]
        assert best_per_metric(metrics)["acc"].value == 0.9

    def test_grad_norm_picks_minimum(self) -> None:
        metrics = [
            _m("gn", 5.0, step=0, kind=MetricKind.GRAD_NORM),
            _m("gn", 1.0, step=1, kind=MetricKind.GRAD_NORM),
        ]
        assert best_per_metric(metrics)["gn"].value == 1.0

    def test_scalar_picks_latest(self) -> None:
        metrics = [
            _m("temperature", 0.1, step=0, kind=MetricKind.SCALAR),
            _m("temperature", 0.5, step=10, kind=MetricKind.SCALAR),
            _m("temperature", 0.2, step=5, kind=MetricKind.SCALAR),
        ]
        # Latest by step.
        assert best_per_metric(metrics)["temperature"].step == 10

    def test_multiple_metric_names(self) -> None:
        metrics = [
            _m("loss", 0.5, kind=MetricKind.LOSS),
            _m("reward", 10.0, kind=MetricKind.REWARD),
            _m("loss", 0.2, kind=MetricKind.LOSS),
        ]
        best = best_per_metric(metrics)
        assert best["loss"].value == 0.2
        assert best["reward"].value == 10.0


class TestSummarizeByKind:
    def test_groups_metrics(self) -> None:
        metrics = [
            _m("loss", 0.1, kind=MetricKind.LOSS),
            _m("reward", 5.0, kind=MetricKind.REWARD),
            _m("lr", 1e-3, kind=MetricKind.LEARNING_RATE),
            _m("val_loss", 0.2, kind=MetricKind.LOSS),
        ]
        grouped = summarize_by_kind(metrics)
        assert len(grouped[MetricKind.LOSS]) == 2
        assert len(grouped[MetricKind.REWARD]) == 1
        assert MetricKind.SCALAR not in grouped  # empty groups dropped

    def test_empty_iterable(self) -> None:
        assert summarize_by_kind([]) == {}


class TestFilterByTags:
    def test_required(self) -> None:
        a = _m("loss", 0.1, tags={"split": "train"})
        b = _m("loss", 0.2, tags={"split": "val"})
        result = filter_by_tags([a, b], required={"split": "train"})
        assert result == (a,)

    def test_forbidden(self) -> None:
        a = _m("loss", 0.1, tags={"split": "train"})
        b = _m("loss", 0.2, tags={"split": "val"})
        result = filter_by_tags([a, b], forbidden={"split": "val"})
        assert result == (a,)

    def test_required_and_forbidden_combined(self) -> None:
        a = _m("loss", 0.1, tags={"split": "train", "env": "cartpole"})
        b = _m("loss", 0.2, tags={"split": "train", "env": "lunar"})
        result = filter_by_tags(
            [a, b],
            required={"split": "train"},
            forbidden={"env": "lunar"},
        )
        assert result == (a,)

    def test_missing_required_key_drops_metric(self) -> None:
        a = _m("loss", 0.1, tags={"other": "x"})
        result = filter_by_tags([a], required={"split": "train"})
        assert result == ()


class TestRollingMean:
    def test_grows_to_window(self) -> None:
        rm = RollingMean(window=3)
        assert rm.value is None
        rm.push(10.0)
        assert rm.value == 10.0
        rm.push(20.0)
        assert rm.value == 15.0
        rm.push(30.0)
        assert rm.value == 20.0

    def test_evicts_oldest(self) -> None:
        rm = RollingMean(window=3)
        for v in (10.0, 20.0, 30.0, 40.0):
            rm.push(v)
        # Window now holds (20, 30, 40) -> mean 30.
        assert rm.value == 30.0
        assert len(rm) == 3

    def test_reset(self) -> None:
        rm = RollingMean(window=2)
        rm.push(1.0)
        rm.reset()
        assert rm.value is None
        assert len(rm) == 0

    def test_rejects_bad_window(self) -> None:
        with pytest.raises(ValueError):
            RollingMean(window=0)
        with pytest.raises(ValueError):
            RollingMean(window=-3)

    def test_rejects_non_numeric(self) -> None:
        rm = RollingMean(window=2)
        with pytest.raises(TypeError):
            rm.push("oops")  # type: ignore[arg-type]

    def test_rejects_bool(self) -> None:
        rm = RollingMean(window=2)
        with pytest.raises(TypeError):
            rm.push(True)  # type: ignore[arg-type]

    def test_coerces_int_to_float(self) -> None:
        rm = RollingMean(window=2)
        rm.push(5)
        rm.push(15)
        assert rm.value == 10.0
