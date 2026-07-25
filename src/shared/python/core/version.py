"""Version information for UpstreamDrift.

``__version__`` is *resolved*, never hardcoded: it comes from the same chain
the About dialog and ``GET /api/v1/about`` use
(:func:`src.shared.python.version_info.resolve_app_version` — the repo-root
``VERSION`` file, then installed package metadata, then a fallback).

Before issue #8064 this module hardcoded ``1.0.0`` while ``VERSION`` and
``pyproject.toml`` said ``2.1.1``, so the launcher title bar and the About
dialog reported different builds inside the same process.
"""

from src.shared.python.version_info import resolve_app_version


def _parse_version_info(version: str) -> tuple[int, int, int]:
    """Return ``(major, minor, patch)`` for a resolved version string.

    Non-numeric suffixes (``2.1.1-rc1``, ``1.0.0-beta``) and missing
    components degrade to zeros rather than raising, because a version string
    is never worth crashing the launcher over.

    Args:
        version: Resolved version string, e.g. ``"2.1.1"``.

    Returns:
        Exactly three integers.
    """
    if not isinstance(version, str):
        raise TypeError(f"version must be a string, got {type(version).__name__}")

    parts: list[int] = []
    for raw in version.split(".")[:3]:
        digits = ""
        for char in raw.strip():
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


__version__ = resolve_app_version()
__version_info__ = _parse_version_info(__version__)

__title__ = "UpstreamDrift"
__description__ = (
    "Professional biomechanical analysis platform for physics-based modeling"
)
__author__ = "UpstreamDrift Team"
__author_email__ = "support@upstreamdrift.dev"
__license__ = "MIT"
__url__ = "https://github.com/D-sorganization/UpstreamDrift"

# Build information
__build_date__ = "2026-01-12"
__python_requires__ = ">=3.11"

# Feature flags for professional version
FEATURES = {
    "video_pose_estimation": True,
    "ball_flight_physics": True,
    "api_server": True,
    "multi_engine_support": True,
    "professional_visualization": True,
    "cloud_integration": True,
    "enterprise_features": False,  # Reserved for enterprise license
}

# Supported physics engines
SUPPORTED_ENGINES = ["mujoco", "drake", "pinocchio", "myosuite", "opensim"]

# Professional edition features
PROFESSIONAL_FEATURES = [
    "Cross-engine validation and comparison",
    "Video-based pose estimation with MediaPipe",
    "Ball flight physics with Magnus effect",
    "REST API for cloud integration",
    "Standardized model library",
    "Professional visualization suite",
    "Batch processing capabilities",
    "Export to multiple formats",
]
