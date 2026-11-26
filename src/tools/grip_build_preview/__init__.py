"""Headless grip build preview (HRG-GB4, #15126). Visual-only; not core model."""

from src.tools.grip_build_preview.preview import generate_preview_rgb_frames
from src.tools.grip_build_preview.settings import (
    DEFAULT_PREVIEW_SETTINGS,
    PreviewSettings,
)

__all__ = ["DEFAULT_PREVIEW_SETTINGS", "PreviewSettings", "generate_preview_rgb_frames"]
