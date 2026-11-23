"""Versioned numerical-stage evidence, separate from physical impact certification.

This archive is not yet embedded in the v4 solution bundle. Older bundle records
cannot be upgraded without capturing their actual accepted-stage evidence.
"""

from dataclasses import asdict
from typing import Any

import numpy as np

from .stage_trace import (
    RunOutcome,
    RunPlan,
    SegmentPlan,
    Stage,
    StageRecorder,
    TracedStep,
)

SCHEMA = "research-stage-evidence/v1"
SEMANTICS = "ideal_direct_generalized_no_saturation"


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def encode_stage_archive(recorder: StageRecorder) -> dict[str, Any]:
    """Serialize only complete ideal-drive observations; never invent old evidence."""
    if (
        not recorder.capture_ideal_drive
        or recorder.plan is None
        or recorder.plan.requested_steps < 1
        or recorder.outcome is None
        or recorder.outcome.status != "complete"
        or recorder._integrity_failure() is not None
    ):
        raise ValueError("complete ideal-drive stage recording required")
    return _plain(
        {
            "schema": SCHEMA,
            "semantics": SEMANTICS,
            "plan": asdict(recorder.plan),
            "steps": [asdict(step) for step in recorder.steps],
        }
    )


def _keys(value: Any, expected: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("stage archive has missing or unknown fields")


def decode_stage_archive(data: dict[str, Any]) -> StageRecorder:
    """Reject unknown versions and malformed, incomplete or inconsistent traces."""
    _keys(data, {"schema", "semantics", "plan", "steps"})
    if data["schema"] != SCHEMA or data["semantics"] != SEMANTICS:
        raise ValueError("unsupported stage archive; re-recording is required")
    p = data["plan"]
    _keys(
        p,
        {
            "requested_steps",
            "dt_s",
            "substeps",
            "effort_holds",
            "project",
            "coordinate_order",
        },
    )
    if (
        any(
            type(p[k]) is not int
            for k in ("requested_steps", "substeps", "effort_holds")
        )
        or p["requested_steps"] < 1
        or type(p["project"]) is not bool
    ):
        raise ValueError("invalid plan types")
    try:
        plan = RunPlan(**{**p, "coordinate_order": tuple(p["coordinate_order"])})
        recorder = StageRecorder(capture_ideal_drive=True)
        recorder.start_run(plan)
        for item in data["steps"]:
            _keys(
                item,
                {
                    "stages",
                    "segments",
                    "pre_projection_q",
                    "pre_projection_v",
                    "post_projection_q",
                    "post_projection_v",
                },
            )
            segments = []
            for s in item["segments"]:
                _keys(
                    s,
                    {
                        "step",
                        "hold",
                        "segment",
                        "start_s",
                        "duration_s",
                        "substeps",
                        "force_active",
                        "step_sizes_s",
                    },
                )
                if (
                    any(
                        type(s[k]) is not int
                        for k in ("step", "hold", "segment", "substeps")
                    )
                    or type(s["force_active"]) is not bool
                ):
                    raise ValueError("invalid segment types")
                segments.append(
                    SegmentPlan(**{**s, "step_sizes_s": tuple(s["step_sizes_s"])})
                )
            stages = []
            for s in item["stages"]:
                _keys(
                    s,
                    {
                        "step",
                        "hold",
                        "segment",
                        "substep",
                        "stage",
                        "time_s",
                        "q",
                        "v",
                        "tau",
                        "applied_drive",
                        "applied_external",
                        "effort_semantics",
                    },
                )
                if any(
                    type(s[k]) is not int
                    for k in ("step", "hold", "segment", "substep", "stage")
                ):
                    raise ValueError("invalid stage identity")
                stages.append(
                    Stage(
                        **{
                            **s,
                            **{
                                k: np.asarray(s[k], dtype=float)
                                for k in (
                                    "q",
                                    "v",
                                    "tau",
                                    "applied_drive",
                                    "applied_external",
                                )
                            },
                        }
                    )
                )
            boundary = [
                np.asarray(item[k], dtype=float)
                for k in (
                    "pre_projection_q",
                    "pre_projection_v",
                    "post_projection_q",
                    "post_projection_v",
                )
            ]
            recorder.steps.append(TracedStep(tuple(stages), tuple(segments), *boundary))
        recorder.finish()
        if recorder.outcome is None or recorder.outcome.status != "complete":
            raise ValueError("invalid or incomplete stage recording")
        return recorder
    except (TypeError, KeyError, OverflowError, IndexError) as exc:
        raise ValueError("invalid stage archive") from exc


def compare_stage_archives(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Compare full accepted evidence, including force components and clocks."""
    return encode_stage_archive(decode_stage_archive(expected)) == encode_stage_archive(
        decode_stage_archive(actual)
    )
