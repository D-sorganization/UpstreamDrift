"""Lane configuration, capture stance detection, and kinematic bounds."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.engine_core.engine_availability import is_engine_available
from src.shared.python.motion_matching import posture_metrics as post
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK
from src.shared.python.motion_matching.pipeline.plant import (
    MatchingPlant,
    get_plant,
)
from src.shared.python.motion_matching.ground_support import (
    calibrate_ground_height,
    capture_to_native_world,
)
from src.shared.python.motion_matching.pipeline.constants import (
    BOUND_WIDENING,
    CALIBRATION_STRIDE,
    CONTACT_STIFFNESS_N_M,
    ELBOW_PIT_MARKERS,
    ELBOW_PIT_WEIGHT,
    HEAD_MARKER_WEIGHT,
    IK_UNBOUNDED,
    LEG_LABELS,
    PRIOR,
    SPIN_COORDINATES,
    SPIN_PRIOR,
    STANCE_TOLERANCE_M,
    TOE_SPHERES,
    TOE_STANDOFF_M,
    TRAJECTORY_RESTART_MARGIN_M,
    TRAJECTORY_RESTART_THRESHOLD_M,
    TRAJECTORY_RESTARTS,
    WRIST_COORDINATES,
)
from src.shared.python.motion_matching.range_of_motion import (
    HUMAN_RANGES_DEG,
    LOWER_LIMB_RANGES_DEG,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    TourCapture,
    load_tour_capture,
)


def stance_spheres(
    points: np.ndarray, valid: np.ndarray, labels: tuple[str, ...]
) -> list[tuple[str, ...]]:
    """Per frame, the contact spheres judged to be on the ground from marker heights.

    Preconditions:
    - ``points`` must be a 3D float array of shape (frames, markers, 3).
    - ``valid`` must be a 2D boolean array of shape (frames, markers).
    - ``labels`` must have length matching the second dimension of ``points``.

    Postcondition: returns a list of sphere name tuples of length ``frames``.
    """
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must be a 3D array of shape (frames, markers, 3)")
    if valid.shape != points.shape[:2]:
        raise ValueError("valid shape must match points (frames, markers)")
    if len(labels) != points.shape[1]:
        raise ValueError("labels length must match points markers count")

    heights = np.where(valid, points[:, :, 2], np.nan)
    low = np.where(
        np.isfinite(heights), heights - heights[0] < STANCE_TOLERANCE_M, False
    )

    def column(label: str) -> np.ndarray:
        return (
            low[:, labels.index(label)]
            if label in labels
            else np.zeros(points.shape[0], dtype=bool)
        )

    out: list[tuple[str, ...]] = []
    for k in range(points.shape[0]):
        pinned: list[str] = []
        for side in ("r", "l"):
            prefix = side.upper()
            if column(f"{prefix}AnkleOut")[k]:
                pinned.append(f"heel_{side}")
            if column(f"{prefix}ToeIn")[k] and column(f"{prefix}ToeOut")[k]:
                pinned.append(f"forefoot_{side}")
                pinned.append(f"toe_{side}")
        out.append(tuple(pinned))
    return out


def add_toe_spheres(document: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``document`` with toe spheres."""
    if "contact" not in document:
        raise ValueError("document must contain a 'contact' block")
    contact = dict(document["contact"])
    stiffness = (
        contact["parameters"]["stiffness_n_m"]
        if "subject" in document
        else CONTACT_STIFFNESS_N_M
    )
    contact["parameters"] = {**contact["parameters"], "stiffness_n_m": stiffness}
    names = {sphere["name"] for sphere in contact["spheres"]}
    extra = [
        {"name": n, "body": b, "position_m": list(p), "radius_m": r}
        for n, (b, p, r) in TOE_SPHERES.items()
        if n not in names
    ]
    contact["spheres"] = list(contact["spheres"]) + extra
    out = dict(document)
    out["contact"] = contact
    prov = str(document.get("provenance", ""))
    out["provenance"] = (
        f"{prov} | toe contact spheres added on calcanei; stiffness {stiffness:.0e} N/m"
    )
    return out


def fitted_grip(document: dict[str, Any]) -> bool:
    """True when the document's hands carry a fitted (nonzero) rotation."""
    rotation = document.get("subject", {}).get("grip_rotation_deg") or {}
    return any(abs(v) > 0 for angles in rotation.values() for v in angles)


def document_bounds(document: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Radian IK bounds declared by an anthropometric document."""
    ranges = document.get("coordinate_ranges_deg", {})
    for name, (lo, hi) in ranges.items():
        if lo >= hi:
            raise ValueError(
                f"ranges low < high violated for coordinate {name}: [{lo}, {hi}]"
            )
    return {
        name: (float(np.radians(lo)), float(np.radians(hi)))
        for name, (lo, hi) in ranges.items()
        if name not in IK_UNBOUNDED and not name.endswith(("_r", "_l"))
    }


def wrist_bounds() -> dict[str, tuple[float, float]]:
    """Human wrist and forearm ranges as radian IK bounds."""
    return {
        name: (float(np.radians(lo)), float(np.radians(hi)))
        for name, (lo, hi) in HUMAN_RANGES_DEG.items()
        if name in WRIST_COORDINATES
    }


def document_seed(
    document: dict[str, Any],
    kin: Any,
    candidate: dict[str, Any] | None = None,
) -> np.ndarray:
    """Start pose: the document's address seed (anthropometric geometry) or
    the qualified candidate's coordinates (native geometry)."""
    q = np.zeros(len(kin.coordinate_order))
    seed = document.get("address_seed_deg")
    if candidate is not None:
        for name, value in zip(
            candidate["coordinate_names"], candidate["q0"], strict=True
        ):
            if seed is None or name.startswith(("Translation", "HipInput")):
                q[kin.coordinate_order.index(name)] = value
    for name, value in (seed or {}).items():
        q[kin.coordinate_order.index(name)] = float(np.radians(value))
    return q


def configure_lane(lane: Lane, document: dict[str, Any]) -> None:
    """Document-dependent solver settings: declared ranges, the spin priors and
    address constraints of anthropometric documents, full head weight when a
    neck carries the head."""
    lane.bounds |= document_bounds(document)
    if "address_seed_deg" in document:
        lane.anthropometric = True
        lane.prior_weights = dict.fromkeys(SPIN_COORDINATES, SPIN_PRIOR)
    if "NeckInputZ" in document.get("coordinate_order", ()):
        lane.marker_weights = {}


class Lane:
    """Capture, ground, stance and bounds shared by every stage."""

    def __init__(
        self,
        labels: tuple[str, ...] | None = None,
        c3d: Path | str | None = None,
        *,
        capture: TourCapture | None = None,
        plant: MatchingPlant | None = None,
    ) -> None:
        self.plant = plant
        if capture is not None:
            self.c3d = Path(c3d) if c3d is not None else Path("synthetic.c3d")
            c = capture.subset(labels) if labels is not None else capture
            self.labels = tuple(c.labels)
        elif c3d is not None:
            self.c3d = Path(c3d)
            if labels is None:
                raise ValueError("labels must be provided with c3d")
            self.labels = tuple(labels)
            c = load_tour_capture(self.c3d).subset(self.labels)
        else:
            raise ValueError("Either c3d or capture must be provided")

        self.points = capture_to_native_world(c.points_m)
        self.valid = c.valid.copy()
        self.times = np.asarray(c.time_s, dtype=float)
        self.frames = int(c.frames)
        self.ground_cal = calibrate_ground_height(
            self.points,
            self.valid,
            self.labels,
            LEG_LABELS[4:],
            standoff_m=TOE_STANDOFF_M,
        )
        self.ground = GroundPlane(
            normal=(0.0, 0.0, 1.0), height_m=self.ground_cal.height_m
        )
        self.stance = stance_spheres(self.points, self.valid, self.labels)
        self.bounds = {
            f"{joint}_{side}": (
                float(np.radians(lo * BOUND_WIDENING)),
                float(np.radians(hi * BOUND_WIDENING)),
            )
            for joint, (lo, hi) in LOWER_LIMB_RANGES_DEG.items()
            for side in ("r", "l")
        }
        self.calibration_frames = list(range(0, self.frames, CALIBRATION_STRIDE))
        self.marker_weights = {
            label: HEAD_MARKER_WEIGHT
            for label in MARKER_SEGMENTS["head"]
            if label in self.labels
        }
        self.prior_weights: dict[str, float] = {}
        self.anthropometric = False

    def kinematics(
        self,
        spec_bytes: bytes | Mapping[str, Any],
        attachments: Mapping[str, tuple[str, Sequence[float]]],
    ) -> tuple[Any, Any]:
        """Instantiate the model and marker kinematics via MatchingPlant.

        Precondition: spec ground plane matches the toe calibration height.
        """
        if self.plant is None:
            self.plant = get_plant("mujoco", spec_bytes)
        ground_plane = self.plant.ground_plane
        ground = self.ground
        if abs(ground_plane.height_m - ground.height_m) > 1e-3:
            raise ValueError("Spec ground height disagrees with the toe calibration")
        ordered = {
            label: (
                attachments[label][0],
                (
                    float(attachments[label][1][0]),
                    float(attachments[label][1][1]),
                    float(attachments[label][1][2]),
                ),
            )
            for label in self.labels
            if label in attachments
        }
        kin = self.plant.create_ik(ordered)
        adapter = getattr(
            self.plant, "adapter", getattr(self.plant, "model", self.plant)
        )
        return adapter, kin

    def marker_pit(self, side: str, frames: Sequence[int]) -> np.ndarray | None:
        """Mean marker-derived elbow pit direction of ``side`` over ``frames``
        (None when the markers are missing or the elbow is straight)."""
        cols = [self.labels.index(m) for m in ELBOW_PIT_MARKERS[side]]
        pits: list[np.ndarray] = []
        for f in frames:
            if not self.valid[f, cols].all():
                continue
            pit = post.elbow_pit_direction(*self.points[f, cols])
            if pit is not None:
                pits.append(pit)
        if not pits:
            return None
        mean = np.mean(pits, axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else None

    def pit_targets_for(
        self, frames: Sequence[int], weight: float
    ) -> dict[str, tuple[Sequence[float], Sequence[float], float]]:
        out: dict[str, tuple[Sequence[float], Sequence[float], float]] = {}
        for side in ("L", "R"):
            pit = self.marker_pit(side, frames)
            if pit is not None:
                out[f"{side}S"] = (
                    (1.0, 0.0, 0.0),
                    tuple(float(v) for v in pit),
                    weight,
                )
        return out

    def pit_targets_per_frame(self, weight: float) -> list[dict | None]:
        """One pit target set per capture frame (None where unobservable)."""
        return [self.pit_targets_for([f], weight) or None for f in range(self.frames)]

    def trajectory(
        self,
        kin: BaseFullBodyIK,
        q_start: np.ndarray,
        frames: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, list[Any]]:
        return kin.solve_trajectory(
            self.points,
            self.valid,
            q_start,
            ground=self.ground,
            frames=frames,
            prior_weight=PRIOR,
            flat_feet_per_frame=self.stance,
            plant_stance=True,
            bounds=self.bounds,
            marker_weights=self.marker_weights,
            prior_weights=self.prior_weights,
            axis_targets_per_frame=(
                self.pit_targets_per_frame(0.01) if self.anthropometric else None
            ),
            restarts=TRAJECTORY_RESTARTS,
            restart_threshold_m=TRAJECTORY_RESTART_THRESHOLD_M,
            restart_margin_m=TRAJECTORY_RESTART_MARGIN_M,
        )

    def pinned_rms(
        self, spec_bytes: bytes, attachments: dict, q_start: np.ndarray
    ) -> float:
        from src.shared.python.motion_matching.pipeline.reference import marker_errors

        _, kin = self.kinematics(spec_bytes, attachments)
        frames = self.calibration_frames
        q, _ = self.trajectory(kin, q_start, frames=frames)
        errors = marker_errors(kin, q, self.points[frames])
        return float(np.sqrt(np.mean(errors[self.valid[frames]] ** 2)))

    def best_address(
        self,
        kin: Any,
        base: np.ndarray,
        *,
        neutral: bool = False,
        pit_weight: float = ELBOW_PIT_WEIGHT,
    ) -> Any:
        """Best address fit over the leg seeds."""
        from src.shared.python.motion_matching.pipeline.address import best_address

        return best_address(self, kin, base, neutral=neutral, pit_weight=pit_weight)

    def static_trial(
        self,
        spec_bytes: bytes,
        seeds: dict[str, tuple[str, Sequence[float]]],
        q_seed: np.ndarray,
    ) -> tuple[dict[str, tuple[str, tuple[float, float, float]]], Any, Any]:
        """Alternating static trial: neutral address fit and marker placement."""
        from src.shared.python.motion_matching.pipeline.address import static_trial

        return static_trial(self, spec_bytes, seeds, q_seed)

    def static_offsets(
        self,
        kin: Any,
        q: np.ndarray,
        seeds: dict[str, tuple[str, Sequence[float]]],
    ) -> dict[str, tuple[str, tuple[float, float, float]]]:
        """Static-trial placement of every seed marker at pose q."""
        from src.shared.python.motion_matching.pipeline.address import static_offsets

        return static_offsets(self, kin, q, seeds)

    def elbow_pit_targets(
        self, kin: Any, q: np.ndarray, weight: float
    ) -> dict[str, tuple[Sequence[float], Sequence[float], float]]:
        """Axis targets pulling upper arms toward address marker pit directions."""
        from src.shared.python.motion_matching.pipeline.address import elbow_pit_targets

        return elbow_pit_targets(self, kin, q, weight)

    def calibrate_legs(
        self,
        spec_bytes: bytes,
        upper: Mapping[str, tuple[str, Sequence[float]]],
        seeds: Mapping[str, tuple[str, Sequence[float]]],
        q_start: np.ndarray,
    ) -> tuple[dict[str, tuple[str, tuple[float, float, float]]], Any]:
        """Alternating marker calibration of seeds with stance pins."""
        from src.shared.python.motion_matching.pipeline.address import calibrate_legs

        return calibrate_legs(self, spec_bytes, upper, seeds, q_start)


def probe_pink_capability() -> tuple[bool, dict[str, Any]]:
    """Probe if Pink constrained IK solver and dependencies are available."""
    missing = [eng for eng in ("pink", "pinocchio") if not is_engine_available(eng)]
    if missing:
        msg = f"Required packages unavailable: {', '.join(missing)}"
        return False, {"available": False, "missing": missing, "reason": msg}
    return True, {"available": True, "missing": [], "reason": "ok"}
