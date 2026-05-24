"""Tests for :mod:`training.metrics`."""

from __future__ import annotations

import math
from types import MappingProxyType

import pytest

from training import MetricKind, TrainingConfigError, TrainingMetric

pytestmark = pytest.mark.unit


class TestMetricKind:
    def test_loss_lower_is_better(self) -> None:
        assert MetricKind.LOSS.lower_is_better is True
        assert MetricKind.LOSS.higher_is_better is False

    def test_reward_higher_is_better(self) -> None:
        assert MetricKind.REWARD.higher_is_better is True
        assert MetricKind.REWARD.lower_is_better is False

    def test_accuracy_higher_is_better(self) -> None:
        assert MetricKind.ACCURACY.higher_is_better is True

    def test_grad_norm_lower_is_better(self) -> None:
        assert MetricKind.GRAD_NORM.lower_is_better is True

    def test_scalar_is_neutral(self) -> None:
        assert MetricKind.SCALAR.lower_is_better is False
        assert MetricKind.SCALAR.higher_is_better is False

    def test_learning_rate_is_neutral(self) -> None:
        assert MetricKind.LEARNING_RATE.lower_is_better is False
        assert MetricKind.LEARNING_RATE.higher_is_better is False


class TestTrainingMetricConstruction:
    def test_minimal(self) -> None:
        m = TrainingMetric(name="val_loss", value=0.5, step=0, timestamp=1700_000_000.0)
        assert m.name == "val_loss"
        assert m.value == 0.5
        assert m.kind is MetricKind.SCALAR  # default

    def test_with_kind_and_tags(self) -> None:
        m = TrainingMetric(
            name="reward",
            value=12.5,
            step=100,
            timestamp=1700_000_000.0,
            kind=MetricKind.REWARD,
            tags={"env": "cartpole"},
        )
        assert m.kind is MetricKind.REWARD
        assert m.tags == {"env": "cartpole"}

    def test_tags_are_frozen(self) -> None:
        m = TrainingMetric(
            name="loss",
            value=0.1,
            step=0,
            timestamp=1.0,
            tags={"split": "val"},
        )
        assert isinstance(m.tags, MappingProxyType)
        with pytest.raises(TypeError):
            m.tags["split"] = "train"  # type: ignore[index]

    def test_value_coerced_to_float(self) -> None:
        m = TrainingMetric(name="lr", value=1, step=0, timestamp=0.0)
        assert isinstance(m.value, float)

    def test_external_tags_dict_mutation_does_not_leak(self) -> None:
        source = {"split": "train"}
        m = TrainingMetric(name="loss", value=0.1, step=0, timestamp=0.0, tags=source)
        source["split"] = "val"
        assert m.tags["split"] == "train"


class TestTrainingMetricValidation:
    @pytest.mark.parametrize("bad_name", ["", "   ", "\t"])
    def test_rejects_empty_name(self, bad_name: str) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingMetric(name=bad_name, value=0.1, step=0, timestamp=0.0)

    def test_rejects_non_string_name(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingMetric(name=42, value=0.1, step=0, timestamp=0.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_value(self, bad_value: float) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingMetric(name="loss", value=bad_value, step=0, timestamp=0.0)

    def test_rejects_bool_as_value(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingMetric(name="loss", value=True, step=0, timestamp=0.0)  # type: ignore[arg-type]

    def test_rejects_negative_step(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingMetric(name="loss", value=0.1, step=-1, timestamp=0.0)

    def test_rejects_bool_step(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingMetric(name="loss", value=0.1, step=True, timestamp=0.0)  # type: ignore[arg-type]

    def test_rejects_negative_timestamp(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingMetric(name="loss", value=0.1, step=0, timestamp=-0.1)

    def test_rejects_non_metrickind(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingMetric(
                name="loss",
                value=0.1,
                step=0,
                timestamp=0.0,
                kind="loss",  # type: ignore[arg-type]
            )

    def test_rejects_empty_tag_key(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingMetric(
                name="loss",
                value=0.1,
                step=0,
                timestamp=0.0,
                tags={"": "val"},
            )

    def test_rejects_empty_tag_value(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingMetric(
                name="loss",
                value=0.1,
                step=0,
                timestamp=0.0,
                tags={"split": ""},
            )

    def test_rejects_non_string_tag_value(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingMetric(
                name="loss",
                value=0.1,
                step=0,
                timestamp=0.0,
                tags={"split": 1},  # type: ignore[dict-item]
            )
