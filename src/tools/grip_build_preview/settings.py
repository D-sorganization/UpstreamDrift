"""Preview render settings (provider-friendly, no hard-coded physics)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from src.shared.python.grip_build.receipt import BuildStatus


@runtime_checkable
class PreviewSettingsProvider(Protocol):
    """Supplies colours and timing; swap for themes or CI fixtures."""

    def settings(self) -> PreviewSettings: ...


@dataclass(frozen=True)
class StatusColors:
    """RGB tuples for stage HUD (not contact forces)."""

    ok: tuple[int, int, int] = (72, 187, 120)
    gap: tuple[int, int, int] = (251, 191, 36)
    joint_limit: tuple[int, int, int] = (248, 113, 113)
    infeasible: tuple[int, int, int] = (220, 38, 38)


@dataclass(frozen=True)
class PreviewSettings:
    """Headless 2D preview layout."""

    width_px: int = 480
    height_px: int = 360
    frames_per_segment: int = 4
    dpi: int = 100
    margin_m: float = 0.02
    club_line_rgb: tuple[int, int, int] = (200, 200, 200)
    bone_capsule_rgb: tuple[int, int, int] = (130, 180, 255)
    contact_marker_rgb: tuple[int, int, int] = (255, 255, 90)
    background_rgb: tuple[int, int, int] = (24, 24, 32)
    status_colors: StatusColors = StatusColors()

    def __post_init__(self) -> None:
        if self.width_px < 64 or self.height_px < 64:
            raise ValueError("width_px and height_px must be >= 64")
        if self.frames_per_segment < 1:
            raise ValueError("frames_per_segment must be >= 1")
        if self.dpi < 50:
            raise ValueError("dpi must be >= 50")

    def color_for_status(self, status: BuildStatus) -> tuple[int, int, int]:
        mapping = {
            BuildStatus.OK: self.status_colors.ok,
            BuildStatus.GAP: self.status_colors.gap,
            BuildStatus.JOINT_LIMIT: self.status_colors.joint_limit,
            BuildStatus.INFEASIBLE: self.status_colors.infeasible,
        }
        return mapping[status]


DEFAULT_PREVIEW_SETTINGS: Final[PreviewSettings] = PreviewSettings()


@dataclass(frozen=True)
class _DefaultPreviewProvider:
    def settings(self) -> PreviewSettings:
        return DEFAULT_PREVIEW_SETTINGS


def get_preview_settings(
    provider: PreviewSettingsProvider | None = None,
) -> PreviewSettings:
    if provider is None:
        return DEFAULT_PREVIEW_SETTINGS
    if not isinstance(provider, PreviewSettingsProvider):
        raise TypeError("provider must implement PreviewSettingsProvider")
    return provider.settings()
