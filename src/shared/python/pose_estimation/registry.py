"""Pose estimator registry — the single seam for runtime estimators.

Part of epic #8390 (C2/#8402). Previously, adding an estimator required
coordinated edits in ~5 places (the ``VideoPosePipeline._load_estimator``
if/elif, the API's ``VALID_ESTIMATOR_TYPES``, the motion-capture route's
skeleton/availability tables, the UI option list, and the availability
probes) — which is how ``movenet``/``blazepose`` drifted into existence
without implementations (#8392). This registry mirrors the
``motion_pipeline.sources`` adapter-registry pattern for runtime
estimators: one entry per estimator carrying its lazy factory,
availability probe, install hint, and skeleton template. Consumers
derive their tables from here.

Deliberately dependency-light at import time: factories import their
estimator modules lazily, and the estimator interface is imported only
for type checking.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.shared.python.pose_estimation.interface import PoseEstimator

__all__ = [
    "EstimatorInfo",
    "create_estimator",
    "estimator_availability",
    "get_estimator_info",
    "implemented_estimator_types",
    "list_estimators",
    "register_estimator",
    "unregister_estimator",
]


@dataclass(frozen=True)
class EstimatorInfo:
    """Registered runtime pose estimator.

    Attributes:
        name: Stable identifier (API value, pipeline config value).
        display_name: Human-readable name for UIs.
        description: One-line description for source listings.
        probe_module: Module whose importability gates availability.
        install_hint: Human-readable remedy when unavailable.
        skeleton: Joint template as ``(name, parent)`` mappings, in order.
        factory: Lazy constructor; receives keyword options (e.g.
            ``min_confidence``) and returns a ``PoseEstimator``. Must not
            import heavy dependencies until called.
    """

    name: str
    display_name: str
    description: str
    probe_module: str
    install_hint: str
    skeleton: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    factory: Callable[..., PoseEstimator] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("estimator name must be non-empty")
        if not self.probe_module:
            raise ValueError("probe_module must be non-empty")


_REGISTRY: dict[str, EstimatorInfo] = {}


def register_estimator(info: EstimatorInfo) -> EstimatorInfo:
    """Register an estimator; rejects duplicate names."""
    if info.name in _REGISTRY:
        raise ValueError(f"estimator {info.name!r} is already registered")
    _REGISTRY[info.name] = info
    return info


def unregister_estimator(name: str) -> None:
    """Remove an estimator (test hygiene; missing names are a no-op)."""
    _REGISTRY.pop(name, None)


def list_estimators() -> tuple[EstimatorInfo, ...]:
    """All registered estimators in registration order."""
    return tuple(_REGISTRY.values())


def implemented_estimator_types() -> frozenset[str]:
    """Names of every registered estimator."""
    return frozenset(_REGISTRY)


def get_estimator_info(name: str) -> EstimatorInfo:
    """Look up a registered estimator.

    Raises:
        ValueError: For unknown names (message lists valid ones).
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        valid = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(
            f"Unknown estimator type: {name}. Registered: {valid}"
        ) from None


def estimator_availability(name: str) -> tuple[bool, str | None]:
    """Probe availability of a registered estimator.

    Returns ``(available, reason)``; ``reason`` is ``None`` when
    available. Spec-less mock modules count as unavailable.
    """
    info = get_estimator_info(name)
    try:
        if find_spec(info.probe_module) is not None:
            return True, None
    except (ImportError, ValueError, ModuleNotFoundError):
        pass
    return False, info.install_hint


def create_estimator(name: str, **options: Any) -> PoseEstimator:
    """Construct a registered estimator via its lazy factory.

    Args:
        name: Registered estimator name.
        **options: Forwarded to the factory (unknown keys are the
            factory's concern).

    Raises:
        ValueError: Unknown name, or entry without a factory.
    """
    info = get_estimator_info(name)
    if info.factory is None:
        raise ValueError(f"estimator {name!r} has no runtime factory")
    return info.factory(**options)


# ---------------------------------------------------------------------------
# Built-in estimators
# ---------------------------------------------------------------------------

_MEDIAPIPE_SKELETON: tuple[dict[str, Any], ...] = (
    {"name": "nose", "parent": None},
    {"name": "left_eye", "parent": "nose"},
    {"name": "right_eye", "parent": "nose"},
    {"name": "left_ear", "parent": "left_eye"},
    {"name": "right_ear", "parent": "right_eye"},
    {"name": "left_shoulder", "parent": "nose"},
    {"name": "right_shoulder", "parent": "nose"},
    {"name": "left_elbow", "parent": "left_shoulder"},
    {"name": "right_elbow", "parent": "right_shoulder"},
    {"name": "left_wrist", "parent": "left_elbow"},
    {"name": "right_wrist", "parent": "right_elbow"},
    {"name": "left_hip", "parent": "left_shoulder"},
    {"name": "right_hip", "parent": "right_shoulder"},
    {"name": "left_knee", "parent": "left_hip"},
    {"name": "right_knee", "parent": "right_hip"},
    {"name": "left_ankle", "parent": "left_knee"},
    {"name": "right_ankle", "parent": "right_knee"},
)

_OPENPOSE_SKELETON: tuple[dict[str, Any], ...] = (
    {"name": "head", "parent": None},
    {"name": "neck", "parent": "head"},
    {"name": "right_shoulder", "parent": "neck"},
    {"name": "right_elbow", "parent": "right_shoulder"},
    {"name": "right_wrist", "parent": "right_elbow"},
    {"name": "left_shoulder", "parent": "neck"},
    {"name": "left_elbow", "parent": "left_shoulder"},
    {"name": "left_wrist", "parent": "left_elbow"},
    {"name": "mid_hip", "parent": "neck"},
    {"name": "right_hip", "parent": "mid_hip"},
    {"name": "right_knee", "parent": "right_hip"},
    {"name": "right_ankle", "parent": "right_knee"},
    {"name": "left_hip", "parent": "mid_hip"},
    {"name": "left_knee", "parent": "left_hip"},
    {"name": "left_ankle", "parent": "left_knee"},
)


def _make_mediapipe(**options: Any) -> PoseEstimator:
    from src.shared.python.pose_estimation.mediapipe_estimator import (
        MediaPipeEstimator,
    )

    min_confidence = float(options.get("min_confidence", 0.5))
    return MediaPipeEstimator(
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
        enable_temporal_smoothing=bool(options.get("enable_temporal_smoothing", True)),
    )


def _make_openpose(**options: Any) -> PoseEstimator:
    from src.shared.python.pose_estimation.openpose_estimator import (
        OpenPoseEstimator,
    )

    return OpenPoseEstimator()


register_estimator(
    EstimatorInfo(
        name="mediapipe",
        display_name="MediaPipe Pose",
        description="Real-time pose estimation using Google MediaPipe",
        probe_module="mediapipe",
        install_hint=(
            "MediaPipe (>=0.10, Tasks API) is not installed on the server "
            "(pip install mediapipe); fetch the pose model with "
            "python3 -m src.shared.python.pose_estimation.mediapipe_models"
        ),
        skeleton=_MEDIAPIPE_SKELETON,
        factory=_make_mediapipe,
    )
)
register_estimator(
    EstimatorInfo(
        name="openpose",
        display_name="OpenPose",
        description="Multi-person pose estimation using OpenPose",
        probe_module="pyopenpose",
        install_hint="OpenPose Python bindings are not installed on the server",
        skeleton=_OPENPOSE_SKELETON,
        factory=_make_openpose,
    )
)
