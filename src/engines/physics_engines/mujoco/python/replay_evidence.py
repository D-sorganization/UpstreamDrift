"""Generate native MuJoCo replay, parity diagnostics and honest playback evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from src.engines.physics_engines.mujoco.python.candidate_replay import (
    ReplayPlant,
    replay_controls,
)
from src.engines.physics_engines.mujoco.python.full_body_ik import (
    FullBodyMarkerKinematics,
)
from src.engines.physics_engines.mujoco.python.replay_contract import (
    ReplaySettings,
    configuration_failures,
    g1_acceptance,
    load_candidate,
    metric_coverage,
    window_indices,
)
from src.shared.python.contracts import precondition
from src.shared.python.motion_matching.cross_engine_replay import (
    render_marker_overlay_animation,
)
from src.shared.python.motion_matching.ground_support import capture_to_native_world
from src.shared.python.motion_matching.loaders._marker_clusters import (
    y_up_to_z_up_rotation,
)
from src.shared.python.motion_matching.pipeline.receipt_schema import (
    AcceptanceReceipt,
    CandidateReplayReceipt,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    TourCapture,
    load_tour_capture,
)
from src.shared.python.motion_matching.tour_metrics import compute_shared_metrics


@dataclass(frozen=True)
class ReplayFiles:
    candidate: Path
    source_receipt: Path
    document: Path
    capture: Path
    attachments: Path
    output: Path
    candidate_sha256: str


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare(files: ReplayFiles, settings: ReplaySettings, diagnostic: bool) -> tuple:
    source = json.loads(files.source_receipt.read_text())
    document = json.loads(files.document.read_text())
    for field, path in (
        ("document_sha256", files.document),
        ("capture_sha256", files.capture),
    ):
        if source.get(field) != _digest(path):
            raise ValueError(f"Source {field} mismatch")
    arrays = load_candidate(files.candidate, files.candidate_sha256)
    if source.get("candidate_sha256", files.candidate_sha256) != files.candidate_sha256:
        raise ValueError("Source candidate SHA256 mismatch")
    failures = configuration_failures(source, document, settings)
    if failures and not diagnostic:
        raise ValueError("Unverified source configuration: " + "; ".join(failures))
    if tuple(arrays["coordinate_order"]) != tuple(document["coordinate_order"]):
        raise ValueError("Candidate coordinate order differs from model")
    capture = load_tour_capture(files.capture).subset(tuple(arrays["labels"]))
    count = len(arrays["time_s"])
    if not np.allclose(capture.time_s[:count], arrays["time_s"], atol=1e-10, rtol=0):
        raise ValueError("Candidate time does not match capture frames")
    if not np.array_equal(capture.valid[:count], arrays["valid"]):
        raise ValueError("Candidate validity differs from canonical capture")
    native_target = capture_to_native_world(capture.points_m[:count])
    if not np.allclose(
        native_target[arrays["valid"]],
        arrays["target_m"][arrays["valid"]],
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("Candidate target differs from canonical capture")
    attachment_data = json.loads(files.attachments.read_text())
    offsets = attachment_data["ik"]["attachments_m"]
    attachments = {
        name: (offsets[name]["body"], offsets[name]["offset_m"])
        for name in arrays["labels"]
    }
    return source, document, arrays, capture, attachments, failures


def _metrics(capture: TourCapture, markers: np.ndarray) -> dict[str, float]:
    count = len(markers)
    prefix = TourCapture(
        capture.time_s[:count],
        capture.labels,
        capture.points_m[:count],
        capture.valid[:count],
        capture.source_sha256,
    )
    y_up = markers @ y_up_to_z_up_rotation()
    return compute_shared_metrics(prefix, y_up, tracked_labels=prefix.labels).as_dict()


def _audit(
    plant: ReplayPlant, kin: FullBodyMarkerKinematics, q: np.ndarray, v: np.ndarray
) -> dict[str, float]:
    forces: list[float] = []
    penetrations: list[float] = []
    translations: list[float] = []
    rotations: list[float] = []
    for qi, vi in zip(q, v, strict=True):
        samples = plant.adapter.evaluate_contact_samples(
            dict(zip(plant.names, qi, strict=True)),
            dict(zip(plant.names, vi, strict=True)),
        )
        forces.append(
            sum(float(np.linalg.norm(s.normal_force_n)) for s in samples.values())
        )
        penetrations.extend(s.penetration_m for s in samples.values())
        translation, rotation = kin.closure_error(qi)
        translations.append(translation)
        rotations.append(rotation)
    return {
        "max_normal_force_n": max(forces),
        "max_penetration_m": max(penetrations),
        "max_closure_residual_m": max(translations),
        "max_closure_residual_rad": max(rotations),
    }


def _save_playback(
    files: ReplayFiles,
    arrays: dict,
    q: np.ndarray,
    v: np.ndarray,
    markers: np.ndarray,
    reference: np.ndarray,
) -> dict:
    count = len(q)
    files.output.mkdir(parents=True, exist_ok=True)
    replay = files.output / "returned-replay.npz"
    np.savez_compressed(
        replay,
        time_s=arrays["time_s"][:count],
        native_state=np.c_[q, v],
        coordinate_order=arrays["coordinate_order"],
        markers_m=markers,
        target_m=arrays["target_m"][:count],
        valid=arrays["valid"][:count],
        labels=arrays["labels"],
        candidate_sha256=files.candidate_sha256,
    )
    rotation = y_up_to_z_up_rotation()
    paths = {"replay": replay}
    for name, points, label in (
        ("playback", markers, "MuJoCo Rejected Replay"),
        ("kinematic-playback", reference, "MuJoCo IK Playback Only"),
    ):
        n = len(points)
        path = files.output / f"{name}.gif"
        render_marker_overlay_animation(
            arrays["time_s"][:n],
            arrays["target_m"][:n] @ rotation,
            points @ rotation,
            path,
            engine_name=label,
            stride=max(1, n // 80),
            valid_mask=arrays["valid"][:n],
        )
        paths[name] = path
    return {
        name: {"path": path.name, "sha256": _digest(path)}
        for name, path in paths.items()
    }


def _convergence(
    plant: ReplayPlant,
    kin: FullBodyMarkerKinematics,
    arrays: dict,
    efforts: np.ndarray,
    markers: np.ndarray,
    settings: ReplaySettings,
) -> float | None:
    count = len(window_indices(arrays["time_s"], 0.85))
    if len(markers) < count:
        return None
    refined = replace(
        settings,
        rtol=settings.rtol / 10,
        atol=settings.atol / 10,
        max_step_s=settings.max_step_s / 2,
    )
    qr, _, failure = replay_controls(
        plant,
        arrays["time_s"][:count],
        arrays["q"][0],
        arrays["v"][0],
        efforts[:count],
        refined,
    )
    if failure is not None:
        return None
    refined_markers = np.stack([kin.marker_positions(qi) for qi in qr])
    diff = markers[:count] - refined_markers
    return float(
        np.sqrt(np.max(np.einsum("...i,...i->...", diff, diff)))
    )  # ⚡ Bolt: np.sqrt(np.max(np.einsum(...))) is ~2.7x faster than np.max(np.linalg.norm(..., axis=-1))


@precondition(lambda files: files.candidate.is_file(), "Source candidate required")
def generate_replay(
    files: ReplayFiles, settings: ReplaySettings, *, diagnostic: bool = False
) -> dict:
    """Replay all saved intervals; certify G1 only with complete, converged evidence."""
    import mujoco

    start = perf_counter()
    source, document, arrays, capture, attachments, failures = _prepare(
        files, settings, diagnostic
    )
    plant = ReplayPlant(document, source["ground_height_m"], settings)
    kin = FullBodyMarkerKinematics(plant.adapter, attachments)
    reference = np.stack([kin.marker_positions(q) for q in arrays["q"]])
    diff = reference - arrays["markers_m"]
    difference = np.sqrt(
        np.einsum("...i,...i->...", diff, diff)
    )  # ⚡ Bolt: np.sqrt(np.einsum(...)) avoids temporary allocations and is ~2.7x faster than np.linalg.norm(..., axis=-1)
    marker_parity = float(np.max(difference[arrays["valid"]]))
    if marker_parity > 1e-8:
        failures.append("Same-state marker parity exceeds 1e-8 m")
    if failures and not diagnostic:
        raise ValueError("Parity failed: " + "; ".join(failures))
    efforts = np.zeros_like(arrays["q"])
    if len(arrays["u"]) == len(arrays["q"]):
        efforts[:, plant.actuated] = arrays["u"]
    else:
        efforts[:-1, plant.actuated] = arrays["u"]
        efforts[-1, plant.actuated] = arrays["u"][-1]
    q, v, failure = replay_controls(
        plant, arrays["time_s"], arrays["q"][0], arrays["v"][0], efforts, settings
    )
    markers = np.stack([kin.marker_positions(qi) for qi in q])
    full_metrics = _metrics(capture, markers)
    g1_count = len(window_indices(arrays["time_s"], 0.85))
    complete = len(q) >= g1_count
    count = min(len(q), g1_count)
    g1_metrics = _metrics(capture, markers[:count]) | _audit(
        plant, kin, q[:count], v[:count]
    )
    convergence = _convergence(plant, kin, arrays, efforts, markers, settings)
    evidence = {
        "parity": not failures,
        "complete": complete,
        "converged": convergence is not None and convergence <= 1e-5,
        "root_history": "delta_tau_root" in arrays
        and np.isfinite(arrays["delta_tau_root"]).all().item()
        and float(np.max(np.abs(arrays["delta_tau_root"]))) <= 0.1,
        "coverage": complete and metric_coverage(arrays, g1_count),
        # This archive stores smoothed IK, not an independent forward replay.
        # Source dynamics qualification remains blocked under #10336.
        "source_dynamics": False,
    }
    receipt = CandidateReplayReceipt(
        engine="mujoco",
        engine_version=mujoco.__version__,
        candidate_sha256=files.candidate_sha256,
        document_sha256=_digest(files.document),
        capture_sha256=_digest(files.capture),
        source_receipt_sha256=_digest(files.source_receipt),
        attachments_sha256=_digest(files.attachments),
        configuration={
            "settings": asdict(settings),
            "contact": document["contact"],
            "ground_height_m": source["ground_height_m"],
            "control_interpolation": "zero_order_hold",
            "root_efforts": "zero",
            "controller": "none",
            "grip": "rigid KKT",
            "integrator": "DOP853",
        },
        parity={
            "status": "UNVERIFIED" if failures else "PASSED",
            "failures": failures,
            "same_state_marker_max_m": marker_parity,
            "dynamics_parity": "UNVERIFIED: no independent source forward replay",
        },
        integration={
            "acceptance_audit_sha256": _digest(
                Path(__file__).with_name("replay_contract.py")
            ),
            "requested_frames": len(arrays["time_s"]),
            "completed_frames": len(q),
            "requested_end_s": float(arrays["time_s"][-1]),
            "completed_end_s": float(arrays["time_s"][len(q) - 1]),
            "failure": failure,
            "pose_resets": 0,
            "g1_convergence_max_marker_m": convergence,
            "g1_convergence_tolerance_m": 1e-5,
        },
        shared_metrics=full_metrics,
        g1_metrics=g1_metrics,
        acceptance=AcceptanceReceipt.model_validate(
            g1_acceptance(g1_metrics, evidence)
        ),
        artifacts=_save_playback(files, arrays, q, v, markers, reference),
        elapsed_s=perf_counter() - start,
        qualification="Diagnostic replay only. Legacy source lacks plant/control provenance and root history; IK playback is not dynamics acceptance.",
    ).model_dump(mode="json")
    (files.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    return receipt
