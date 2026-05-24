"""Tests for :mod:`training.config`."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import MappingProxyType

import pytest

from training import (
    CURRENT_SCHEMA_VERSION,
    ResourceRequest,
    TrainingConfig,
    TrainingConfigError,
    TrainingFramework,
)

pytestmark = pytest.mark.unit


def _minimal_config(**overrides: object) -> TrainingConfig:
    defaults: dict[str, object] = {
        "framework": TrainingFramework.PYTORCH,
        "entry_point": "my_module:train",
        "output_dir": Path("/tmp/out"),
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


class TestTrainingFramework:
    def test_pytorch_value(self) -> None:
        assert TrainingFramework.PYTORCH.value == "pytorch"

    def test_gymnasium_value(self) -> None:
        assert TrainingFramework.GYMNASIUM.value == "gymnasium"


class TestTrainingConfigConstruction:
    def test_minimal_construction(self) -> None:
        cfg = _minimal_config()
        assert cfg.framework is TrainingFramework.PYTORCH
        assert cfg.entry_point == "my_module:train"
        assert cfg.output_dir == Path("/tmp/out")
        assert cfg.schema_version == CURRENT_SCHEMA_VERSION
        assert cfg.dataset_id is None
        assert cfg.max_epochs is None
        assert cfg.max_steps is None
        assert cfg.seed is None

    def test_default_resources_is_resourcerequest(self) -> None:
        cfg = _minimal_config()
        assert isinstance(cfg.resources, ResourceRequest)

    def test_is_frozen(self) -> None:
        cfg = _minimal_config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.entry_point = "other"  # type: ignore[misc]

    def test_hyperparameters_default_empty(self) -> None:
        cfg = _minimal_config()
        assert dict(cfg.hyperparameters) == {}

    def test_hyperparameters_frozen_via_proxy(self) -> None:
        cfg = _minimal_config(hyperparameters={"lr": 1e-3})
        assert isinstance(cfg.hyperparameters, MappingProxyType)
        with pytest.raises(TypeError):
            cfg.hyperparameters["lr"] = 1e-2  # type: ignore[index]

    def test_external_hyperparameters_mutation_does_not_leak(self) -> None:
        params: dict[str, object] = {"lr": 1e-3}
        cfg = _minimal_config(hyperparameters=params)
        params["lr"] = 999.0
        assert cfg.hyperparameters["lr"] == 1e-3

    def test_tags_frozen(self) -> None:
        cfg = _minimal_config(tags={"owner": "claude"})
        assert isinstance(cfg.tags, MappingProxyType)


class TestTrainingConfigValidation:
    def test_rejects_non_framework_enum(self) -> None:
        with pytest.raises(TrainingConfigError):
            _minimal_config(framework="pytorch")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_entry", ["", "   "])
    def test_rejects_empty_entry_point(self, bad_entry: str) -> None:
        with pytest.raises(TrainingConfigError):
            _minimal_config(entry_point=bad_entry)

    def test_rejects_non_path_output_dir(self) -> None:
        with pytest.raises(TrainingConfigError):
            _minimal_config(output_dir="/tmp/out")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_version", [0, -1])
    def test_rejects_non_positive_schema_version(self, bad_version: int) -> None:
        with pytest.raises(TrainingConfigError):
            _minimal_config(schema_version=bad_version)

    @pytest.mark.parametrize("bad_value", [0, -5])
    def test_rejects_non_positive_max_epochs(self, bad_value: int) -> None:
        with pytest.raises(TrainingConfigError):
            _minimal_config(max_epochs=bad_value)

    @pytest.mark.parametrize("bad_value", [0, -5])
    def test_rejects_non_positive_max_steps(self, bad_value: int) -> None:
        with pytest.raises(TrainingConfigError):
            _minimal_config(max_steps=bad_value)

    def test_accepts_both_caps_set(self) -> None:
        """Both ``max_epochs`` and ``max_steps`` may be set together."""
        cfg = _minimal_config(max_epochs=10, max_steps=10_000)
        assert cfg.max_epochs == 10
        assert cfg.max_steps == 10_000

    def test_rejects_negative_seed(self) -> None:
        with pytest.raises(TrainingConfigError):
            _minimal_config(seed=-1)

    def test_accepts_zero_seed(self) -> None:
        cfg = _minimal_config(seed=0)
        assert cfg.seed == 0

    def test_rejects_empty_dataset_id(self) -> None:
        with pytest.raises(TrainingConfigError):
            _minimal_config(dataset_id="")
        with pytest.raises(TrainingConfigError):
            _minimal_config(dataset_id="   ")

    def test_rejects_non_resourcerequest(self) -> None:
        with pytest.raises(TrainingConfigError):
            _minimal_config(resources={"cpu_cores": 1})  # type: ignore[arg-type]

    def test_rejects_invalid_tag_key(self) -> None:
        with pytest.raises(TrainingConfigError):
            _minimal_config(tags={"": "val"})

    def test_rejects_non_string_tag_value(self) -> None:
        with pytest.raises(TrainingConfigError):
            _minimal_config(tags={"k": 1})  # type: ignore[dict-item]
