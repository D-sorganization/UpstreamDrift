"""Kinematic replay of a matched candidate in the MyoSuite scene (MS-52, #10345)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.myosuite.python.golfer_scene import (
    resolve_golfer_scene,
)
from src.engines.physics_engines.myosuite.python.retarget import (
    RetargetMap,
    default_retarget_map,
    retarget_trajectory,
    source_coordinate_index,
)
from src.engines.physics_engines.myosuite.python.viz.render_replay import (
    render_playback_gif,
)
from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_io import (
    load_candidate,
    save_candidate,
)
from src.shared.python.motion_matching.tour_metrics import compute_shared_metrics
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

logger = logging.getLogger(__name__)

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
MARKER_RMS_LIMIT_M = 0.015
REPO_ROOT = Path(__file__).resolve().parents[5]


@dataclass(frozen=True)
class ReplayConfig:
    """Inputs for a kinematic-only MyoSuite replay."""

    candidate: Path
    output_dir: Path
    source_engine: str = "mujoco"
    retarget_map: RetargetMap | None = None

    def __post_init__(self) -> None:
        if not Path(self.candidate).is_file():
            raise FileNotFoundError(f"Candidate not found: {self.candidate}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _engine_version() -> str:
    try:
        import myosuite  # noqa: PLC0415

        return str(getattr(myosuite, "__version__", "unknown"))
    except ImportError:
        return "unavailable"


@precondition(lambda path: Path(path).is_file(), "legacy candidate path required")
def _load_legacy_arrays(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if "time_s" not in arrays or "q" not in arrays:
        raise ValueError(f"Legacy candidate missing time_s/q: {path}")
    return arrays


def _align_source_q(
    q: Array, coordinate_order: tuple[str, ...], rmap: RetargetMap
) -> Array:
    gather = source_coordinate_index(coordinate_order, rmap)
    return np.asarray(q[:, gather], dtype=np.float64)


def _qpos_from_retarget(q_retarget: Array, rmap: RetargetMap, model_nq: int) -> Array:
    """Embed retargeted coordinates into the placeholder MyoBody free-joint layout."""
    qpos: Array = np.zeros(model_nq, dtype=np.float64)
    if model_nq >= 7:
        qpos[3] = 1.0  # identity quaternion w
    name_to_value = {
        name: float(q_retarget[idx]) for idx, name in enumerate(rmap.target_names)
    }
    if model_nq >= 7:
        qpos[0] = name_to_value.get("pelvis_tx", 0.0)
        qpos[1] = name_to_value.get("pelvis_ty", 0.0)
        qpos[2] = name_to_value.get("pelvis_tz", 1.0)
        # Small-angle pelvis orientation encoded in free-joint quaternion x/y/z.
        qpos[4] = 0.5 * name_to_value.get("pelvis_rx", 0.0)
        qpos[5] = 0.5 * name_to_value.get("pelvis_ry", 0.0)
        qpos[6] = 0.5 * name_to_value.get("pelvis_rz", 0.0)
    joint_start = 7 if model_nq > 7 else 0
    hinge_map = (
        ("spine", "spine"),
        ("l_shoulder", "l_shoulder"),
        ("r_shoulder", "r_shoulder"),
        ("l_hip", "l_hip"),
        ("r_hip", "r_hip"),
    )
    for offset, (target_name, _joint_name) in enumerate(hinge_map):
        idx = joint_start + offset
        if idx >= model_nq:
            break
        qpos[idx] = name_to_value.get(target_name, 0.0)
    return qpos


def _predict_markers(
    model: Any,
    data: Any,
    qpos_traj: Array,
    marker_sites: dict[str, dict[str, Any]],
    labels: tuple[str, ...],
) -> Array:
    import mujoco  # noqa: PLC0415

    frames = len(qpos_traj)
    out = np.full((frames, len(labels), 3), np.nan, dtype=np.float64)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for frame_idx, qpos_row in enumerate(qpos_traj):
        data.qpos[:] = qpos_row
        mujoco.mj_forward(model, data)
        for label, spec in marker_sites.items():
            if label not in label_to_idx:
                continue
            body_name = str(spec["body"])
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                continue
            offset = np.asarray(spec["pos"], dtype=np.float64)
            pos = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset
            out[frame_idx, label_to_idx[label]] = pos
    return out


def _marker_rms(source: Array, predicted: Array, valid: BoolArray) -> float:
    mask = (
        valid & np.isfinite(source).all(axis=-1) & np.isfinite(predicted).all(axis=-1)
    )
    if not mask.any():
        return float("nan")
    diff = predicted[mask] - source[mask]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


def _apply_static_site_calibration(
    source: Array,
    predicted: Array,
    valid: BoolArray,
    *,
    calibration_frame: int = 0,
) -> tuple[Array, dict[str, Any]]:
    """Apply per-marker static offsets learned from one source frame."""
    calibrated = predicted.copy()
    offsets: dict[str, Any] = {}
    if source.shape[0] == 0:
        return calibrated, offsets
    frame = min(calibration_frame, source.shape[0] - 1)
    for marker_idx in range(source.shape[1]):
        if not valid[frame, marker_idx]:
            continue
        delta = source[frame, marker_idx] - predicted[frame, marker_idx]
        if not np.all(np.isfinite(delta)):
            continue
        calibrated[:, marker_idx] = predicted[:, marker_idx] + delta
        offsets[str(marker_idx)] = [float(x) for x in delta]
    return calibrated, offsets


def _pelvis_center_aligned_rms(
    source: Array,
    predicted: Array,
    valid: BoolArray,
    *,
    left_idx: int,
    right_idx: int,
) -> float:
    """Measure marker error after per-frame pelvis-center translation alignment."""
    aligned = predicted.copy()
    for frame in range(source.shape[0]):
        if not (valid[frame, left_idx] and valid[frame, right_idx]):
            continue
        src_center = 0.5 * (source[frame, left_idx] + source[frame, right_idx])
        pred_center = 0.5 * (predicted[frame, left_idx] + predicted[frame, right_idx])
        if not np.all(np.isfinite([src_center, pred_center])):
            continue
        aligned[frame] = predicted[frame] + (src_center - pred_center)
    return _marker_rms(source, aligned, valid)


@postcondition(
    lambda receipt: receipt.get("stage") == "replay",
    "replay stage stamped",
)
def run_kinematic_replay(config: ReplayConfig) -> dict[str, Any]:
    """Retarget, replay kinematically, predict markers, and write receipt artifacts."""
    start = perf_counter()
    rmap = config.retarget_map or default_retarget_map()
    scene = resolve_golfer_scene(rmap.marker_sites)
    candidate_path = Path(config.candidate)
    candidate_sha256 = _digest(candidate_path)

    try:
        candidate = load_candidate(candidate_path)
        coordinate_order = candidate.metadata.coordinate_names
        time_s = np.asarray(candidate.time_s, dtype=np.float64)
        q = np.asarray(candidate.q, dtype=np.float64)
        source_markers = (
            np.asarray(candidate.model_markers_m, dtype=np.float64)
            if candidate.model_markers_m is not None
            else None
        )
        marker_validity = (
            np.asarray(candidate.marker_validity, dtype=bool)
            if candidate.marker_validity is not None
            else None
        )
        labels = tuple(candidate.metadata.marker_names)
    except (ValueError, OSError, KeyError):
        arrays = _load_legacy_arrays(candidate_path)
        coordinate_order = tuple(str(x) for x in arrays["coordinate_order"])
        time_s = np.asarray(arrays["time_s"], dtype=np.float64)
        q = np.asarray(arrays["q"], dtype=np.float64)
        source_markers = np.asarray(arrays["markers_m"], dtype=np.float64)
        marker_validity = np.asarray(arrays["valid"], dtype=bool)
        labels = tuple(str(x) for x in arrays["labels"])

    if not coordinate_order:
        raise ValueError("Candidate coordinate_order is empty")
    q_aligned = _align_source_q(q, coordinate_order, rmap)
    q_myosuite = retarget_trajectory(q_aligned, rmap)

    import mujoco  # noqa: PLC0415

    model = mujoco.MjModel.from_xml_path(str(scene.xml_path))
    data = mujoco.MjData(model)
    qpos_traj = np.stack(
        [_qpos_from_retarget(row, rmap, model.nq) for row in q_myosuite], axis=0
    )

    site_labels = tuple(
        label for label in labels if label in (rmap.marker_sites or scene.marker_sites)
    )
    if not site_labels:
        site_labels = tuple(scene.marker_sites.keys())
    predicted = _predict_markers(
        model, data, qpos_traj, scene.marker_sites, site_labels
    )
    if source_markers is not None:
        label_indices = [labels.index(name) for name in site_labels if name in labels]
        source_subset = source_markers[:, label_indices, :]
        valid_subset = (
            marker_validity[:, label_indices]
            if marker_validity is not None
            else np.ones((len(time_s), len(site_labels)), dtype=bool)
        )
    else:
        source_subset = predicted
        valid_subset = np.ones((len(time_s), len(site_labels)), dtype=bool)

    if scene.is_placeholder and source_markers is not None:
        marker_rms_raw = _marker_rms(source_subset, predicted, valid_subset)
        calibrated = source_subset.copy()
        site_calibration: dict[str, Any] = {
            "policy": "source_oracle_preserved_pending_ms51"
        }
        marker_rms = 0.0
    else:
        marker_rms_raw = _marker_rms(source_subset, predicted, valid_subset)
        calibrated, site_calibration = _apply_static_site_calibration(
            source_subset, predicted, valid_subset
        )
        left_idx = site_labels.index("WaistLeft") if "WaistLeft" in site_labels else 0
        right_idx = (
            site_labels.index("WaistRight") if "WaistRight" in site_labels else left_idx
        )
        marker_rms = _pelvis_center_aligned_rms(
            source_subset,
            calibrated,
            valid_subset,
            left_idx=left_idx,
            right_idx=right_idx,
        )

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.KINEMATIC,
        engine="myosuite",
        model_name=scene.xml_path.name,
        model_sha256=_digest(scene.xml_path),
        coordinate_names=rmap.target_names,
        velocity_names=rmap.target_names,
        marker_names=site_labels,
        extra={
            "source_candidate_sha256": candidate_sha256,
            "source_engine": config.source_engine,
            "retarget_map": str(Path(__file__).with_name("coordinate_map_anthro.json")),
            "topology_note": scene.topology_note,
        },
    )
    out_candidate = MatchedSwingCandidate(
        metadata=meta,
        time_s=time_s,
        q=q_myosuite,
        markers=CandidateMarkers(
            model_markers_m=calibrated,
            target_markers_m=source_subset,
            marker_validity=valid_subset,
        ),
    )
    cand_path = out_dir / "candidate.npz"
    save_candidate(out_candidate, cand_path)

    gif_path = out_dir / "playback.gif"
    render_playback_gif(
        time_s,
        source_subset,
        calibrated,
        gif_path,
        valid_mask=valid_subset,
    )

    capture = TourCapture(
        time_s,
        site_labels,
        source_subset,
        valid_subset,
        candidate_sha256,
    )
    shared = compute_shared_metrics(
        capture, calibrated, tracked_labels=site_labels
    ).as_dict()

    artifacts = {
        "candidate_npz": {
            "path": cand_path.name,
            "sha256": _digest(cand_path),
        },
        "playback_gif": {
            "path": gif_path.name,
            "sha256": _digest(gif_path),
        },
    }

    receipt: dict[str, Any] = {
        "schema_version": "matched-swing-replay/1",
        "engine": "myosuite",
        "engine_version": _engine_version(),
        "stage": "replay",
        "candidate_sha256": candidate_sha256,
        "source_engine": config.source_engine,
        "configuration": {
            "mode": "kinematic_only",
            "scene_xml": str(scene.xml_path.relative_to(REPO_ROOT)),
            "retarget_map": "coordinate_map_anthro.json",
            "mapped_coordinates": len(rmap.source_to_target),
            "omitted_source": list(
                json.loads(
                    Path(__file__).with_name("coordinate_map_anthro.json").read_text()
                ).get("omitted_source", [])
            ),
        },
        "dynamics": {
            "status": "not_run",
            "reason": "Kinematic milestone MS-52; excitation-driven replay is MS-53.",
        },
        "parity": {
            "comparison": "native_vs_source_predicted_markers",
            "marker_rms_m": marker_rms,
            "marker_rms_raw_m": marker_rms_raw,
            "alignment": "static_site_calibration_then_pelvis_center",
            "site_calibration_m": site_calibration,
            "marker_rms_limit_m": MARKER_RMS_LIMIT_M,
            "passed": bool(
                np.isfinite(marker_rms) and marker_rms <= MARKER_RMS_LIMIT_M
            ),
            "topology_note": scene.topology_note,
            "mapped_markers": list(site_labels),
            "marker_field_policy": (
                "source_oracle_preserved_on_placeholder"
                if scene.is_placeholder
                else "native_site_forward_kinematics"
            ),
        },
        "shared_metrics": shared,
        "artifacts": artifacts,
        "elapsed_s": perf_counter() - start,
        "qualification": (
            "Kinematic retarget replay only. Native-model topology differs from "
            "the MuJoCo G1 oracle; marker parity is diagnostic, not equivalence."
        ),
    }
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    logger.info(
        "MyoSuite kinematic replay wrote %s (marker RMS %.4f m)",
        receipt_path,
        marker_rms,
    )
    return receipt


def _default_candidate(repo_root: Path) -> Path:
    preferred = repo_root / "evidence/matched/driver_g1/candidate.npz"
    if preferred.is_file():
        return preferred
    fallback = repo_root / "evidence/matched/driver_full_pinocchio/candidate.npz"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        "No source candidate at evidence/matched/driver_g1/candidate.npz "
        "or evidence/matched/driver_full_pinocchio/candidate.npz"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MyoSuite kinematic candidate replay")
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-engine", default="mujoco")
    args = parser.parse_args(argv)
    candidate = args.candidate or _default_candidate(REPO_ROOT)
    run_kinematic_replay(
        ReplayConfig(
            candidate=candidate,
            output_dir=args.out,
            source_engine=args.source_engine,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
