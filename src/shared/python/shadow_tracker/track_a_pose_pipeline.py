"""Track A historical-capture pose pipeline with 2h OpenPose cap and fallback."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
from importlib import import_module
from importlib.util import find_spec
import inspect
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from src.shared.python.core.contracts import StateError
from src.shared.python.core.error_utils import ModelError
from src.shared.python.logging_pkg.logging_config import get_logger

from ._validation import check_bool, check_int, check_nonneg_float, check_str
from .historical_capture import observation_record

if TYPE_CHECKING:
    from src.shared.python.pose_estimation.interface import PoseEstimationResult

logger = get_logger(__name__)
HISTORICAL_CAPTURE_SCHEMA_VERSION = "historical-capture/1.0.0"
OPENPOSE_SETUP_CAP_S = 7200.0
BackendState = Literal["ok", "blocked"]
FrameStatus = Literal["detected", "partial", "missing"]
_SETUP_ERRS = (ImportError, OSError, RuntimeError, StateError, ValueError, ModelError)


class TrackAPoseEstimator(Protocol):
    """Narrow pose-estimator seam for Track A frame observation."""

    def estimate_from_image(self, image: np.ndarray) -> PoseEstimationResult: ...


ProbeFn = Callable[[], TrackAPoseEstimator | None]


def _nn_int(v: object, name: str) -> int:
    i = check_int(v, name)
    if i < 0:
        raise ValueError(f"{name} must be non-negative")
    return i


def _num(v: object, name: str, *, unit: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be numeric")
    f = float(v)
    if not math.isfinite(f) or (unit and not 0.0 <= f <= 1.0):
        raise ValueError(f"{name} invalid: {f}")
    return f


def _impl_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class _JsonRecord:
    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass into JSON-compatible primitives."""
        return asdict(self)  # type: ignore[call-overload]


@dataclass(frozen=True, slots=True)
class BackendStatus(_JsonRecord):
    """Per-backend setup and detection summary for Track A observation."""

    status: BackendState
    elapsed_setup_s: float
    reason: str
    detected_frames: int = 0
    mean_coverage: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in ("ok", "blocked"):
            raise ValueError(f"Invalid status: {self.status!r}")
        check_nonneg_float(self.elapsed_setup_s, "elapsed_setup_s")
        check_str(self.reason, "reason")
        det = _nn_int(self.detected_frames, "detected_frames")
        cov = _num(self.mean_coverage, "mean_coverage", unit=True)
        if self.status == "blocked" and (det != 0 or cov != 0.0):
            raise ValueError("blocked backend cannot report nonzero metrics")


@dataclass(frozen=True, slots=True)
class TrackAFrameObservation(_JsonRecord):
    """Single-frame Track A image observation with explicit missingness."""

    frame_index: int
    timestamp_s: float
    status: FrameStatus
    coordinate_system: str
    confidence: float
    keypoint_coverage: float
    landmarks: dict[str, dict[str, Any]]
    missing_keypoints: tuple[str, ...]
    low_visibility_keypoints: tuple[str, ...]
    physical_time_s: None = None
    physical_time_reason: str = "Historical playback scale is unverified"
    timebase: Literal["unverified"] = "unverified"
    qualification: Literal["image_observations_only"] = "image_observations_only"

    def __post_init__(self) -> None:
        _nn_int(self.frame_index, "frame_index")
        _num(self.timestamp_s, "timestamp_s")
        if self.status not in ("detected", "partial", "missing"):
            raise ValueError(f"Invalid frame status: {self.status!r}")
        _num(self.confidence, "confidence", unit=True)
        _num(self.keypoint_coverage, "keypoint_coverage", unit=True)
        if self.physical_time_s is not None or self.timebase != "unverified":
            raise ValueError("Track A physical time must remain unverified")
        if self.qualification != "image_observations_only":
            raise ValueError("qualification must be image_observations_only")


@dataclass(frozen=True, slots=True)
class TrackAPipelineReceipt(_JsonRecord):
    """Deterministic ``historical-capture/1.0.0`` receipt for Track A."""

    schema_version: str
    subject_id: str
    source_path: str | None
    frame_count: int
    openpose_native: BackendStatus
    openpose_body25_onnx: BackendStatus
    rtmpose: BackendStatus
    fallback_triggered: bool
    active_backends: tuple[str, ...]
    observations: dict[str, tuple[TrackAFrameObservation, ...]]
    setup_cap_s: float = OPENPOSE_SETUP_CAP_S
    qualification: Literal["image_observations_only"] = "image_observations_only"
    physical_time_verified: bool = False
    timebase: Literal["unverified"] = "unverified"
    implementation_sha256: str = field(default_factory=_impl_sha256)

    def __post_init__(self) -> None:
        if self.schema_version != HISTORICAL_CAPTURE_SCHEMA_VERSION:
            raise ValueError(f"Unexpected schema_version: {self.schema_version!r}")
        check_str(self.subject_id, "subject_id")
        _nn_int(self.frame_count, "frame_count")
        if check_nonneg_float(self.setup_cap_s, "setup_cap_s") <= 0.0:
            raise ValueError("setup_cap_s must be positive")
        if (
            self.qualification != "image_observations_only"
            or self.timebase != "unverified"
        ):
            raise ValueError("Track A receipt must remain unverified")
        if check_bool(self.physical_time_verified, "physical_time_verified"):
            raise ValueError("physical_time_verified must remain False")


def sanitize_track_a_frame(
    result: PoseEstimationResult,
    *,
    frame_index: int = 0,
    expected_keypoints: Sequence[str] | None = None,
    min_visibility: float = 0.05,
) -> TrackAFrameObservation:
    """Validate and sanitize one detector frame without dropping or fabricating."""
    idx = _nn_int(frame_index, "frame_index")
    vis_min = _num(min_visibility, "min_vis", unit=True)
    raw_c = result.raw_confidences or {}
    confs = {
        check_str(k, "k"): _num(v, f"vis[{k}]", unit=True) for k, v in raw_c.items()
    }
    base = observation_record(result)
    base_lm: dict[str, dict[str, Any]] = base["landmarks"]
    src = expected_keypoints if expected_keypoints is not None else [*base_lm, *confs]
    exp = tuple(dict.fromkeys(check_str(k, "kp") for k in src))
    names = tuple(dict.fromkeys([*exp, *base_lm, *confs]))
    exp_set = frozenset(exp)
    landmarks: dict[str, dict[str, Any]] = {}
    missing, low_vis, vis_cnt = [], [], 0
    for name in names:
        if name in base_lm:
            entry = dict(base_lm[name])
            vis = entry["visibility"]
            if vis is not None and vis < vis_min:
                entry["status"] = "low_visibility"
                low_vis.append(name)
            else:
                entry["status"] = "visible"
                vis_cnt += int(name in exp_set)
            landmarks[name] = entry
        else:
            entry = {"x": None, "y": None, "visibility": confs.get(name)}
            entry["status"] = "missing"
            landmarks[name] = entry
            missing.append(name)
    cov = (vis_cnt / len(exp)) if exp else 0.0
    st: FrameStatus = "missing"
    if vis_cnt > 0:
        st = (
            "detected"
            if vis_cnt == len(exp) and not missing and not low_vis
            else "partial"
        )
    return TrackAFrameObservation(
        idx,
        float(result.timestamp),
        st,
        str(base["coordinate_system"]),
        float(base["confidence"]),
        cov,
        landmarks,
        tuple(missing),
        tuple(low_vis),
    )


def _builtin_setup(backend: str) -> TrackAPoseEstimator:
    b25 = (None, "openpose_dnn_estimator", "OpenPoseDnnEstimator")
    specs = {
        "openpose_native": ("pyopenpose", "openpose_estimator", "OpenPoseEstimator"),
        "openpose_body25_onnx": b25,
        "rtmpose": ("onnxruntime", "rtmpose_onnx_estimator", "RtmposeOnnxEstimator"),
    }
    probe, sub, cls_name = specs[backend]
    if probe is not None and find_spec(probe) is None:
        raise ImportError(f"{probe} is not installed")
    est = getattr(import_module(f"src.shared.python.pose_estimation.{sub}"), cls_name)()
    est.load_model()
    return est  # type: ignore[no-any-return]


def _decode_with_av(source_path: Path) -> list[np.ndarray]:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    try:
        import av
    except ImportError as err:
        raise ImportError("av (PyAV) is required when frames are absent") from err
    with av.open(str(source_path)) as c:
        return [f.to_ndarray(format="bgr24") for f in c.decode(c.streams.video[0])]


def run_track_a_pipeline(
    *,
    subject_id: str,
    frames: Sequence[np.ndarray] | None = None,
    source_path: Path | None = None,
    setup_cap_s: float = OPENPOSE_SETUP_CAP_S,
    min_visibility: float = 0.05,
    expected_keypoints: Sequence[str] | None = None,
    openpose_native_probe: ProbeFn | None = None,
    openpose_body25_onnx_factory: ProbeFn | None = None,
    rtmpose_factory: ProbeFn | None = None,
    frame_decoder: Callable[[Path], Sequence[np.ndarray]] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> TrackAPipelineReceipt:
    """Run Track A pipeline with 2h native OpenPose cap and separate fallback."""
    clean_subject = check_str(subject_id, "subject_id")
    cap = check_nonneg_float(setup_cap_s, "setup_cap_s")
    if cap <= 0.0:
        raise ValueError("setup_cap_s must be positive")
    _num(min_visibility, "min_visibility", unit=True)
    if frames is None and source_path is None:
        raise ValueError("Either frames or source_path must be provided")
    dec = frame_decoder or _decode_with_av
    resolved = list(frames) if frames is not None else list(dec(source_path))  # type: ignore[arg-type]
    if not resolved:
        raise ValueError("At least one frame is required")

    obs_map: dict[str, tuple[TrackAFrameObservation, ...]] = {}
    sts: dict[str, BackendStatus] = {}
    fb = False

    def _exec(
        name: str, setup_fn: ProbeFn, cap_s: float | None = None
    ) -> BackendStatus:
        start = float(monotonic_fn())
        try:
            est = setup_fn()
        except _SETUP_ERRS as exc:
            el = check_nonneg_float(float(monotonic_fn()) - start, "el")
            rsn = str(exc).strip() or type(exc).__name__
            return BackendStatus("blocked", el, rsn)
        el = check_nonneg_float(float(monotonic_fn()) - start, "el")
        if est is None:
            return BackendStatus("blocked", el, f"{name} returned no estimator")
        if cap_s is not None and el > cap_s:
            rsn = f"{name} setup exceeded {cap_s:.1f}s cap ({el:.1f}s)"
            return BackendStatus("blocked", el, rsn)
        fn = est.estimate_from_image
        two_arg = len(inspect.signature(fn).parameters) >= 2
        obs = tuple(
            sanitize_track_a_frame(
                fn(fr, i * 33) if two_arg else fn(fr),  # type: ignore[call-arg]
                frame_index=i,
                expected_keypoints=expected_keypoints,
                min_visibility=min_visibility,
            )
            for i, fr in enumerate(resolved)
        )
        obs_map[name] = obs
        det = sum(1 for item in obs if item.status in ("detected", "partial"))
        cov = sum(item.keypoint_coverage for item in obs) / len(obs)
        return BackendStatus("ok", el, f"{name} ready", det, cov)

    for name, custom, c_s in (
        ("openpose_native", openpose_native_probe, cap),
        ("openpose_body25_onnx", openpose_body25_onnx_factory, None),
        ("rtmpose", rtmpose_factory, None),
    ):
        if name != "openpose_native" and not fb:
            sts[name] = BackendStatus("blocked", 0.0, "Skipped: openpose_native ok")
            continue
        fac: ProbeFn = custom or (lambda n=name: _builtin_setup(n))  # type: ignore[misc]
        sts[name] = _exec(name, fac, c_s)
        if name == "openpose_native":
            fb = sts[name].status == "blocked"

    active = tuple(n for n, st in sts.items() if st.status == "ok")
    src_s = str(source_path) if source_path else None
    n_st = sts["openpose_native"]
    b_st = sts["openpose_body25_onnx"]
    r_st = sts["rtmpose"]
    return TrackAPipelineReceipt(
        HISTORICAL_CAPTURE_SCHEMA_VERSION,
        clean_subject,
        src_s,
        len(resolved),
        n_st,
        b_st,
        r_st,
        fb,
        active,
        obs_map,
        setup_cap_s=cap,
    )
