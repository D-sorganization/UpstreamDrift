"""Self-contained fixtures for orchestrator unit tests."""

from __future__ import annotations

from src.shared.python.motion_pipeline.orchestrator import (
    AdapterOverride,
    PipelineConfig,
)


def make_minimal_config(
    source_format: str = "json",
    ik_backend: str = "mujoco",
    matching_backend: str = "mujoco",
) -> PipelineConfig:
    return PipelineConfig(
        adapter=AdapterOverride(format=source_format),
        ik_backend=ik_backend,
        matching_backend=matching_backend,
    )
