"""Function to apply preprocessing steps from configuration."""

from __future__ import annotations

import contextlib
from typing import Any
from src.shared.python.motion_pipeline.contracts import (
    KeypointSequence,
    MarkerTrajectory,
)
from src.shared.python.motion_pipeline.preprocessing.pipeline import (
    PreprocessingPipeline,
    GapFillStep,
    FilterStep,
    ResampleStep,
    NormalizeStep,
)
from src.shared.python.motion_pipeline.preprocessing.gap_fill import GapFillStrategy
from src.shared.python.motion_pipeline.preprocessing.filter import FilterType


def apply_preprocessing(
    data: KeypointSequence | MarkerTrajectory,
    steps: list[Any],
) -> KeypointSequence | MarkerTrajectory:
    """Apply a sequence of preprocessing step configurations to data.

    Each step in `steps` is a config object with `name` and `params`.
    """
    pipeline = PreprocessingPipeline()
    for step_config in steps:
        name = step_config.name.lower().replace("_", "").replace("-", "")
        params = step_config.params or {}

        if name in ("gapfill", "gapfillstep"):
            strategy_str = params.get("strategy")
            strategy = None
            if strategy_str:
                with contextlib.suppress(ValueError):
                    strategy = GapFillStrategy(strategy_str)
            pipeline.add_step(
                GapFillStep(strategy=strategy, max_gap=params.get("max_gap", 10))  # type: ignore[arg-type]
            )
        elif name in ("filter", "filterstep"):
            filter_type_str = params.get("filter_type")
            filter_type = None
            if filter_type_str:
                with contextlib.suppress(ValueError):
                    filter_type = FilterType(filter_type_str)
            pipeline.add_step(
                FilterStep(
                    filter_type=filter_type,  # type: ignore[arg-type]
                    cutoff=params.get("cutoff", 6.0),
                    order=params.get("order", 2),
                    fps=params.get("fps"),
                )
            )
        elif name in ("resample", "resamplestep"):
            pipeline.add_step(
                ResampleStep(
                    target_fps=params.get("target_fps", 100.0),
                    source_fps=params.get("source_fps"),
                )
            )
        elif name in ("normalize", "normalizestep"):
            pipeline.add_step(
                NormalizeStep(
                    target_up=params.get("target_up"),  # type: ignore[arg-type]
                    source_up=params.get("source_up"),
                    center_origin=params.get("center_origin", True),
                    target_unit=params.get("target_unit"),
                )
            )
        else:
            raise ValueError(f"Unknown preprocessing step name: {step_config.name}")

    return pipeline.apply(data)
