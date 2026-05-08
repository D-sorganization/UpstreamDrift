"""Unit tests for PipelineConfig validation (orchestrator.py)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.shared.python.motion_pipeline.orchestrator import (
    AdapterOverride,
    PipelineConfig,
    PreprocessingStep,
    Stage,
)


def test_pipeline_config_minimal_valid() -> None:
    cfg = PipelineConfig(adapter=AdapterOverride(format="c3d"))
    assert cfg.adapter.format == "c3d"
    assert cfg.ik_backend == "mujoco"
    assert cfg.matching_backend == "mujoco"
    assert cfg.output_format == "json"
    assert cfg.preprocessing == []


def test_pipeline_config_missing_adapter_raises() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig()  # type: ignore[call-arg]


def test_adapter_override_requires_format() -> None:
    with pytest.raises(ValidationError):
        AdapterOverride()  # type: ignore[call-arg]


def test_preprocessing_step_defaults_enabled_true() -> None:
    step = PreprocessingStep(name="filter")
    assert step.enabled is True
    assert step.params == {}


def test_preprocessing_step_can_be_disabled() -> None:
    step = PreprocessingStep(name="filter", enabled=False)
    assert step.enabled is False


def test_pipeline_config_with_preprocessing_steps() -> None:
    cfg = PipelineConfig(
        adapter=AdapterOverride(format="c3d"),
        preprocessing=[
            PreprocessingStep(name="filter"),
            PreprocessingStep(name="resample", params={"target_fps": 100}),
        ],
    )
    assert len(cfg.preprocessing) == 2
    assert cfg.preprocessing[1].params == {"target_fps": 100}


def test_pipeline_config_custom_backends() -> None:
    cfg = PipelineConfig(
        adapter=AdapterOverride(format="c3d"),
        ik_backend="drake",
        matching_backend="pinocchio",
    )
    assert cfg.ik_backend == "drake"
    assert cfg.matching_backend == "pinocchio"


def test_stage_enum_has_five_stages() -> None:
    expected = {
        "adapter",
        "preprocessing",
        "scaling",
        "inverse_kinematics",
        "motion_matching",
    }
    assert {s.value for s in Stage} == expected


def test_pipeline_config_serializes_to_dict() -> None:
    cfg = PipelineConfig(adapter=AdapterOverride(format="c3d"))
    d = cfg.model_dump()
    assert d["adapter"]["format"] == "c3d"
    assert d["ik_backend"] == "mujoco"
