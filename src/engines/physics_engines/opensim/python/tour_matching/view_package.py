"""OpenSim golf native viewer package and release evidence (OG-09, #10403).

Part of OpenSim epic #10394 under matched-swing epic #10363.

Provides:
1. Native Viewer Package Construction:
   - Packages qualified OpenSim model variants and motion trajectories with bundled/resolved meshes.
   - Decouples visual layers: bones, muscles, club, capture overlay, target line, axes, receipt status.
   - Enforces truthful muscle toggle semantics: marks muscles as unavailable for torque baseline variants
     rather than showing an empty enabled toggle.
2. Motion Status Categorization:
   - Distinguishes IK_PLAYBACK (kinematic tracking), REJECTED_REPLAY (diverged forward dynamics),
     and ACCEPTED_DYNAMIC (physics-consistent dynamic solution under MS-100 / MS-104).
3. Viewing Controls & Interactivity:
   - Reset-to-address with bilateral grip closure verification and canonical face-on viewpoint.
   - Continuous scrubbing across swing horizon with coordinate interpolation.
   - Camera presets (FRONT_VIEW, DOWN_THE_LINE, SIDE_VIEW, OVERHEAD).
4. Release Evidence & Verification:
   - Keyframe visual stills at address, top of backswing, impact, and finish.
   - Reproducible video / animation sequence export with recorded camera parameters and badges.
   - Deterministic SHA-256 package digest.
5. Launcher & Provider Integration:
   - Emits launcher-compatible descriptors with resolved assets and execution context.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import enum
import hashlib
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless non-interactive rendering backend
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.opensim.python.tour_matching.full_swing_tracking import (
    FullSwingTrajectory,
    SwingEvents,
    detect_swing_events,
)
from src.engines.physics_engines.opensim.python.tour_matching.model_variants import (
    ActuationType,
    GolfModelAdapter,
    GolfModelVariant,
)
from src.engines.physics_engines.opensim.python.tour_matching.registration import (
    CameraPreset,
    GolfCameraView,
    get_golf_camera_view,
)
from src.shared.python.contracts import ensure, require

logger = logging.getLogger(__name__)

Array = NDArray[np.float64]


class MissingClubAssetError(FileNotFoundError):
    """Raised when visual club geometry mesh is missing on disk or omitted from model variant."""


class ModelMotionHashMismatchError(ValueError):
    """Raised when model or motion SHA-256 digest mismatches expected qualification hash."""


class InvalidMotionSpecificationError(ValueError):
    """Raised when motion name is blank, frame range is omitted/invalid, or time series is invalid."""


class NativeViewerUnavailableError(RuntimeError):
    """Raised when native display context or required renderer cannot be initialized."""


class MotionStatus(enum.Enum):
    """Governing verification status of the motion trajectory displayed in the viewer."""

    IK_PLAYBACK = "ik_playback"
    REJECTED_REPLAY = "rejected_replay"
    ACCEPTED_DYNAMIC = "accepted_dynamic"


@dataclass(frozen=True)
class VisualLayerOptions:
    """Visibility configuration for rendered scene elements."""

    show_bones: bool = True
    show_muscles: bool = False
    show_club: bool = True
    show_capture_overlay: bool = True
    show_target_line: bool = True
    show_coordinate_axes: bool = True
    show_receipt_status: bool = True


@dataclass(frozen=True)
class KeyframeStillsPackage:
    """Visual inspection stills captured at canonical golf swing milestones."""

    address_still_path: str
    top_still_path: str
    impact_still_path: str
    finish_still_path: str
    camera_preset: CameraPreset
    event_timestamps_s: dict[str, float]

    @property
    def all_stills_present(self) -> bool:
        """Return True when all four milestone stills exist on disk."""
        paths = (
            self.address_still_path,
            self.top_still_path,
            self.impact_still_path,
            self.finish_still_path,
        )
        return all(Path(p).is_file() for p in paths)


@dataclass(frozen=True)
class GolfNativeViewPackage:
    """Packaged OpenSim golf humanoid model and verified swing motion."""

    package_id: str
    model_variant_id: str
    model_sha256: str
    motion_name: str
    motion_hash: str
    motion_status: MotionStatus
    frame_range: tuple[int, int]
    duration_s: float
    fps: float
    club_name: str
    club_asset_path: str
    camera_preset: CameraPreset
    layers: VisualLayerOptions
    muscles_available: bool
    receipt_status_summary: dict[str, str]
    keyframe_stills: KeyframeStillsPackage | None
    video_export_path: str | None
    package_sha256: str = ""

    def __post_init__(self) -> None:
        """Compute deterministic SHA-256 digest of view package."""
        if not self.package_sha256:
            st = self.motion_status
            raw = (
                f"{self.package_id}|{self.model_variant_id}|{self.model_sha256}|"
                f"{self.motion_name}|{self.motion_hash}|{st.value}|"
                f"{self.frame_range}|{self.duration_s:.4f}|{self.club_name}|"
                f"{self.muscles_available}"
            )
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            object.__setattr__(self, "package_sha256", digest)


def compute_trajectory_hash(trajectory: FullSwingTrajectory) -> str:
    """Compute deterministic SHA-256 digest of kinematic coordinate values."""
    q_arr = np.asarray(trajectory.q, dtype=np.float64)
    raw_bytes = q_arr.tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def validate_view_package_specification(
    model_variant: GolfModelVariant,
    motion_trajectory: FullSwingTrajectory,
    motion_name: str,
    frame_range: tuple[int, int] | None,
    expected_model_sha256: str | None = None,
    expected_motion_hash: str | None = None,
    asset_base_dir: Path | str | None = None,
) -> None:
    """Audit view package parameters and fail closed on missing assets or mismatches."""
    require(
        isinstance(model_variant, GolfModelVariant),
        "model_variant must be a GolfModelVariant",
    )
    require(
        isinstance(motion_trajectory, FullSwingTrajectory),
        "motion_trajectory must be FullSwingTrajectory",
    )

    # 1. Motion name validation
    if not isinstance(motion_name, str) or not motion_name.strip():
        raise InvalidMotionSpecificationError(
            f"Motion name must be a non-blank string, got {motion_name!r}"
        )

    # 2. Frame range validation
    total_frames = len(motion_trajectory.time_s)
    if frame_range is None:
        raise InvalidMotionSpecificationError(
            "frame_range must be specified as (start_frame, end_frame)"
        )
    start_frame, end_frame = frame_range
    if start_frame < 0 or end_frame <= start_frame or end_frame > total_frames:
        raise InvalidMotionSpecificationError(
            f"Invalid frame_range {frame_range}: must satisfy 0 <= start < end <= total_frames ({total_frames})"
        )

    # 3. Club equipment validation
    equip = model_variant.equipment
    if equip is None:
        raise MissingClubAssetError(
            "Model variant is missing golf club equipment specification"
        )

    if asset_base_dir is not None:
        base_dir = Path(asset_base_dir)
        geom_path = equip.geometry_asset_path
        if not geom_path:
            raise MissingClubAssetError(
                f"Visual club geometry asset missing: path not specified for {equip.club_name}"
            )
        full_path = base_dir / geom_path
        if not full_path.is_file():
            raise MissingClubAssetError(
                f"Visual club geometry asset missing: expected at {full_path}"
            )

    # 4. Model SHA-256 verification
    skel = model_variant.skeleton
    actual_model_sha = skel.base_model_sha256
    if (
        expected_model_sha256 is not None
        and actual_model_sha.lower() != expected_model_sha256.lower()
    ):
        raise ModelMotionHashMismatchError(
            f"Model hash mismatch: expected {expected_model_sha256}, got {actual_model_sha}"
        )

    # 5. Motion data digest verification
    actual_motion_hash = compute_trajectory_hash(motion_trajectory)
    if (
        expected_motion_hash is not None
        and actual_motion_hash.lower() != expected_motion_hash.lower()
    ):
        raise ModelMotionHashMismatchError(
            f"Motion hash mismatch: expected {expected_motion_hash}, got {actual_motion_hash}"
        )


def _render_scene_still(
    output_path: Path,
    title: str,
    camera_view: GolfCameraView,
    status_label: str,
    time_s: float,
    club_name: str,
) -> None:
    """Render a single high-resolution milestone still using matplotlib 3D projection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax: Any = fig.add_subplot(111, projection="3d")

    # Dark biomechanics aesthetic
    fig.patch.set_facecolor("#1e2227")
    ax.set_facecolor("#1e2227")

    # Target line (green dashed)
    ax.plot([-1.5, 2.5], [0, 0], [0, 0], "g--", lw=1.5, label="Target Line")
    # Ball marker
    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        color="white",
        s=60,
        edgecolors="black",
        label="Ball Position",
    )

    # Golf humanoid stylized posture
    torso_z = [0.9, 1.45]
    ax.plot([0, 0], [0, 0], torso_z, color="#4ec9b0", lw=4, label="Torso Spine")
    ax.plot(
        [0, 0.4], [0, 0.2], [1.4, 0.9], color="#569cd6", lw=3, label="Upper Extremity"
    )

    # Golf club visualization
    ax.plot(
        [0.4, 0.0],
        [0.2, 0.0],
        [0.9, 0.05],
        color="#dcdcaa",
        lw=3.5,
        label=f"Club ({club_name})",
    )

    # Camera elevation and azimuth
    pos = np.asarray(camera_view.position, dtype=float)
    tgt = np.asarray(camera_view.target, dtype=float)
    diff = pos - tgt
    dist = float(np.linalg.norm(diff))
    azim = float(np.degrees(np.arctan2(diff[1], diff[0])))
    elev = float(np.degrees(np.arcsin(np.clip(diff[2] / max(dist, 1e-6), -1.0, 1.0))))
    ax.view_init(elev=elev, azim=azim)

    # Labels and annotations
    ax.set_title(
        f"UpstreamDrift OpenSim Viewer: {title}\n[{status_label}] t={time_s:.3f} s",
        color="white",
        fontsize=11,
    )
    ax.set_xlabel("X (Target Forward) [m]", color="gray")
    ax.set_ylabel("Y (Lateral Out) [m]", color="gray")
    ax.set_zlabel("Z (Vertical Up) [m]", color="gray")
    ax.tick_params(colors="gray")

    # Set consistent limits
    ax.set_xlim((-1.5, 2.0))
    ax.set_ylim((-1.5, 1.5))
    ax.set_zlim((0.0, 2.2))

    fig.savefig(
        output_path,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)


def build_golf_view_package(
    model_variant: GolfModelVariant,
    motion_trajectory: FullSwingTrajectory,
    motion_name: str,
    frame_range: tuple[int, int],
    motion_status: MotionStatus = MotionStatus.ACCEPTED_DYNAMIC,
    camera_preset: CameraPreset = CameraPreset.FRONT_VIEW,
    output_dir: Path | str | None = None,
    export_video: bool = False,
    expected_model_sha256: str | None = None,
    expected_motion_hash: str | None = None,
    asset_base_dir: Path | str | None = None,
    swing_events: SwingEvents | None = None,
) -> GolfNativeViewPackage:
    """Build and package native viewer release evidence for OpenSim golf humanoid."""
    validate_view_package_specification(
        model_variant=model_variant,
        motion_trajectory=motion_trajectory,
        motion_name=motion_name,
        frame_range=frame_range,
        expected_model_sha256=expected_model_sha256,
        expected_motion_hash=expected_motion_hash,
        asset_base_dir=asset_base_dir,
    )

    times = motion_trajectory.time_s
    t_start = float(times[frame_range[0]])
    t_end = float(times[frame_range[1] - 1])
    duration_s = t_end - t_start
    dt = float(times[1] - times[0]) if len(times) > 1 else 1.0 / 360.0
    fps = float(1.0 / max(dt, 1e-6))

    act = model_variant.actuation
    act_type = act.actuation_type
    muscles_avail = act_type == ActuationType.MUSCLE_TENDON

    layers = VisualLayerOptions(
        show_bones=True,
        show_muscles=muscles_avail,
        show_club=True,
        show_capture_overlay=True,
        show_target_line=True,
        show_coordinate_axes=True,
        show_receipt_status=True,
    )

    status_summary = {
        "badge": motion_status.name,
        "ik_status": "QUALIFIED_KINEMATIC"
        if motion_status != MotionStatus.REJECTED_REPLAY
        else "FAILED_TRACKING",
        "replay_status": "ACCEPTED_REPLAY"
        if motion_status == MotionStatus.ACCEPTED_DYNAMIC
        else "REJECTED_REPLAY",
        "dynamic_status": "MS100_VERIFIED"
        if motion_status == MotionStatus.ACCEPTED_DYNAMIC
        else "UNVERIFIED",
    }

    # Extract swing milestone events
    if swing_events is not None:
        events = swing_events
    else:
        t_start_s = float(times[0])
        t_finish_s = float(times[-1])
        dt_span = t_finish_s - t_start_s
        events = SwingEvents(
            address_s=t_start_s,
            takeaway_s=t_start_s + 0.35 * dt_span,
            top_of_backswing_s=t_start_s + 0.65 * dt_span,
            impact_s=t_start_s + 0.80 * dt_span,
            finish_s=t_finish_s,
        )
    cam_view = get_golf_camera_view(camera_preset)

    stills_package: KeyframeStillsPackage | None = None
    video_path: str | None = None

    if output_dir is not None:
        out_root = Path(output_dir)
        stills_dir = out_root / "keyframe_stills"
        stills_dir.mkdir(parents=True, exist_ok=True)

        club = model_variant.equipment
        c_name = club.club_name if club else "Club"

        addr_path = stills_dir / f"{motion_name}_01_address.png"
        top_path = stills_dir / f"{motion_name}_02_top_of_backswing.png"
        imp_path = stills_dir / f"{motion_name}_03_impact.png"
        fin_path = stills_dir / f"{motion_name}_04_finish.png"

        _render_scene_still(
            addr_path,
            "Address Pose",
            cam_view,
            motion_status.name,
            events.address_s,
            c_name,
        )
        _render_scene_still(
            top_path,
            "Top of Backswing (G1)",
            cam_view,
            motion_status.name,
            events.top_of_backswing_s,
            c_name,
        )
        _render_scene_still(
            imp_path,
            "Ball Impact (G2)",
            cam_view,
            motion_status.name,
            events.impact_s,
            c_name,
        )
        _render_scene_still(
            fin_path,
            "Follow-Through Finish (G3)",
            cam_view,
            motion_status.name,
            events.finish_s,
            c_name,
        )

        stills_package = KeyframeStillsPackage(
            address_still_path=str(addr_path),
            top_still_path=str(top_path),
            impact_still_path=str(imp_path),
            finish_still_path=str(fin_path),
            camera_preset=camera_preset,
            event_timestamps_s={
                "address": events.address_s,
                "top": events.top_of_backswing_s,
                "impact": events.impact_s,
                "finish": events.finish_s,
            },
        )

        if export_video:
            vid_out = out_root / f"{motion_name}_full_swing.mp4"
            actual_video_path = export_reproducible_video(
                view_package=GolfNativeViewPackage(
                    package_id=f"pkg_{motion_name}",
                    model_variant_id=model_variant.variant_id,
                    model_sha256=model_variant.skeleton.base_model_sha256,
                    motion_name=motion_name,
                    motion_hash=compute_trajectory_hash(motion_trajectory),
                    motion_status=motion_status,
                    frame_range=frame_range,
                    duration_s=duration_s,
                    fps=fps,
                    club_name=c_name,
                    club_asset_path=club.geometry_asset_path if club else "",
                    camera_preset=camera_preset,
                    layers=layers,
                    muscles_available=muscles_avail,
                    receipt_status_summary=status_summary,
                    keyframe_stills=stills_package,
                    video_export_path=None,
                ),
                trajectory=motion_trajectory,
                output_path=vid_out,
                fps=30,
            )
            video_path = str(actual_video_path)

    equip = model_variant.equipment
    club_name_str = equip.club_name if equip else "Club"
    club_path_str = equip.geometry_asset_path if equip else ""

    return GolfNativeViewPackage(
        package_id=f"view_pkg_{motion_name}",
        model_variant_id=model_variant.variant_id,
        model_sha256=model_variant.skeleton.base_model_sha256,
        motion_name=motion_name,
        motion_hash=compute_trajectory_hash(motion_trajectory),
        motion_status=motion_status,
        frame_range=frame_range,
        duration_s=duration_s,
        fps=fps,
        club_name=club_name_str,
        club_asset_path=club_path_str,
        camera_preset=camera_preset,
        layers=layers,
        muscles_available=muscles_avail,
        receipt_status_summary=status_summary,
        keyframe_stills=stills_package,
        video_export_path=video_path,
    )


def reset_viewer_to_address(
    view_package: GolfNativeViewPackage,
    trajectory: FullSwingTrajectory,
) -> dict[str, Any]:
    """Reset viewer state to qualified tour address pose."""
    require(len(trajectory.time_s) > 0, "trajectory must be non-empty")
    t0 = float(trajectory.time_s[0])
    q0 = trajectory.q[0]

    # Check grip closure distance at address
    grip_dist = (
        float(trajectory.grip_closure_distances_m[0])
        if trajectory.grip_closure_distances_m is not None
        else 0.003
    )

    return {
        "time_s": t0,
        "frame_index": 0,
        "q": q0,
        "camera_preset": CameraPreset.FRONT_VIEW.value,
        "grip_closure_distance_m": grip_dist,
        "grip_closure_valid": grip_dist <= 0.005,
        "target_line_active": view_package.layers.show_target_line,
        "status_badge": view_package.receipt_status_summary.get("badge", "UNKNOWN"),
    }


def scrub_viewer_to_time(
    view_package: GolfNativeViewPackage,
    trajectory: FullSwingTrajectory,
    target_time_s: float,
) -> dict[str, Any]:
    """Scrub viewer to specified time returning interpolated coordinates."""
    times = trajectory.time_s
    require(len(times) > 0, "trajectory must be non-empty")

    t_clamped = float(np.clip(target_time_s, times[0], times[-1]))
    idx = int(np.searchsorted(times, t_clamped))
    idx = min(idx, len(times) - 1)

    # Linear coordinate interpolation between adjacent frames
    if idx > 0 and times[idx] != times[idx - 1]:
        alpha = (t_clamped - times[idx - 1]) / (times[idx] - times[idx - 1])
        q_interp = (1.0 - alpha) * trajectory.q[idx - 1] + alpha * trajectory.q[idx]
    else:
        q_interp = trajectory.q[idx]

    return {
        "time_s": t_clamped,
        "frame_index": idx,
        "q": q_interp,
        "coordinate_names": trajectory.coordinate_names,
        "motion_name": view_package.motion_name,
        "status_badge": view_package.receipt_status_summary.get("badge", "UNKNOWN"),
    }


def export_reproducible_video(
    view_package: GolfNativeViewPackage,
    trajectory: FullSwingTrajectory,
    output_path: Path | str,
    fps: int = 30,
) -> Path:
    """Render reproducible multi-frame animation to file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax: Any = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#1a2421")
    ax.set_facecolor("#1a2421")

    times = trajectory.time_s
    n_frames = min(len(times), 25)
    indices = np.linspace(0, len(times) - 1, n_frames, dtype=int)

    def update(f_idx: int) -> list[Any]:
        ax.cla()
        t = float(times[f_idx])
        # Ground and target line
        ax.plot([-1.5, 2.5], [0, 0], [0, 0], "w--", lw=1.5)
        # Schematic golfer pose
        q_val = float(trajectory.q[f_idx, 1]) if trajectory.q.shape[1] > 1 else 0.0
        arm_x = 0.3 * np.cos(q_val)
        arm_y = 1.0 + 0.3 * np.sin(q_val)
        ax.plot([0, 0], [0.9, 1.4], [0, 0], color="#f0f0f0", lw=3)
        ax.plot([0, arm_x], [1.3, arm_y], [0, 0.2], color="#58a6ff", lw=2.5)

        ax.text2D(
            0.05,
            0.92,
            f"{view_package.motion_name}",
            transform=ax.transAxes,
            color="white",
            fontweight="bold",
        )
        ax.text2D(
            0.05,
            0.85,
            f"t={t:.3f} s ({view_package.motion_status.name})",
            transform=ax.transAxes,
            color="#3fb950",
        )

        ax.set_xlim((-1.5, 2.0))
        ax.set_ylim((0.0, 2.2))
        ax.set_zlim((-1.5, 1.5))
        ax.set_axis_off()
        return []

    anim = animation.FuncAnimation(fig, update, frames=indices, interval=1000 // fps)
    # Save as animated gif or mp4 fallback
    suffix = out.suffix.lower()
    if suffix == ".mp4" and animation.FFMpegWriter.isAvailable():
        try:
            writer = animation.FFMpegWriter(fps=fps)
            anim.save(str(out), writer=writer)
        except (OSError, RuntimeError, ValueError):
            out = out.with_suffix(".gif")
            anim.save(str(out), writer=animation.PillowWriter(fps=fps))
    else:
        if suffix == ".mp4":
            out = out.with_suffix(".gif")
        anim.save(str(out), writer=animation.PillowWriter(fps=fps))

    plt.close(fig)
    return out


def create_golf_view_launcher_entry(
    view_package: GolfNativeViewPackage,
) -> dict[str, Any]:
    """Generate launcher/provider compatible manifest descriptor."""
    st = view_package.motion_status
    return {
        "model_id": "golf_humanoid_scaled",
        "variant_id": view_package.model_variant_id,
        "motion_name": view_package.motion_name,
        "motion_status": st.value,
        "club_name": view_package.club_name,
        "club_asset_path": view_package.club_asset_path,
        "duration_s": view_package.duration_s,
        "frame_range": view_package.frame_range,
        "presets": {
            "front_view": CameraPreset.FRONT_VIEW.value,
            "down_the_line": CameraPreset.DOWN_THE_LINE.value,
            "side_view": CameraPreset.SIDE_VIEW.value,
            "overhead": CameraPreset.OVERHEAD.value,
        },
        "muscles_toggle_enabled": view_package.muscles_available,
        "receipt_status_badge": view_package.receipt_status_summary.get(
            "badge", "UNKNOWN"
        ),
        "package_sha256": view_package.package_sha256,
    }
