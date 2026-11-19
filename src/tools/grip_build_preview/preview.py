"""Orchestrate GB-2 build results into headless RGB frame sequences."""

from __future__ import annotations

import hashlib

import numpy as np

from src.shared.python.grip_build.receipt import GripBuildResult
from src.shared.python.grip_spec.types import GripSpec
from src.tools.grip_build_preview.frame_plan import sample_preview_frames
from src.tools.grip_build_preview.render import render_preview_frame
from src.tools.grip_build_preview.settings import (
    DEFAULT_PREVIEW_SETTINGS,
    PreviewSettings,
    PreviewSettingsProvider,
    get_preview_settings,
)
from src.tools.grip_build_preview.stage_timeline import build_stage_keyframes


def frame_content_hash(frame: np.ndarray) -> str:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise TypeError("frame must be a uint8 ndarray")
    return hashlib.sha256(frame.tobytes()).hexdigest()


def generate_preview_rgb_frames(
    spec: GripSpec,
    result: GripBuildResult,
    settings: PreviewSettings | PreviewSettingsProvider | None = None,
) -> tuple[np.ndarray, ...]:
    if not isinstance(spec, GripSpec):
        raise TypeError("spec must be GripSpec")
    if not isinstance(result, GripBuildResult):
        raise TypeError("result must be GripBuildResult")
    resolved = (
        settings
        if isinstance(settings, PreviewSettings)
        else get_preview_settings(settings)
        if settings is not None
        else DEFAULT_PREVIEW_SETTINGS
    )
    keyframes = build_stage_keyframes(spec, result)
    plan = sample_preview_frames(keyframes, resolved)
    return tuple(render_preview_frame(spec, frame, resolved) for frame in plan)
