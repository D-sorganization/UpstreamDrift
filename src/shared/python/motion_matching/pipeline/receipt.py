"""Receipt creation and assembly for full-body pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.motion_matching.full_body_spec import canonical_sha256
from src.shared.python.motion_matching.pipeline.address import posture_summary
from src.shared.python.motion_matching.pipeline.address_feet import MODES
from src.shared.python.motion_matching.pipeline.constants import (
    IK_UNBOUNDED,
    RATE_HZ,
    STANCE_TOLERANCE_M,
    TOE_SPHERES,
)
from src.shared.python.motion_matching.acceptance import Horizon, evaluate
from src.shared.python.motion_matching.pipeline.receipt_schema import validate_receipt

if TYPE_CHECKING:
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.shared.python.motion_matching.pipeline.lane import Lane


@dataclass(frozen=True)
class BaseSpecProvenance:
    """Receipt provenance of the committed base spec (#12820).

    ``sha256`` is the canonical digest of the document as read from disk,
    before foot-progression seeding rewrites its ``address_seed_deg``, so
    the receipt chains to the committed file whatever the feet mode.
    ``foot_progression_mode`` and ``foot_seed_deg`` record the seeding, so
    the seeded document is the committed one with ``foot_seed_deg`` merged
    into ``address_seed_deg``.
    """

    sha256: str
    foot_progression_mode: str = "off"
    foot_seed_deg: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.foot_progression_mode not in MODES:
            raise ValueError(
                f"foot_progression_mode must be one of {MODES}, "
                f"got {self.foot_progression_mode!r}"
            )
        if self.foot_progression_mode == "off":
            if self.foot_seed_deg is not None:
                raise ValueError("foot_seed_deg must be None when feet are off")
            return
        if not self.foot_seed_deg:
            raise ValueError("a seeded foot-progression mode needs foot_seed_deg")
        for name, value in self.foot_seed_deg.items():
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"foot_seed_deg[{name!r}] must be finite")

    @classmethod
    def of(
        cls,
        document: Mapping[str, Any],
        foot_progression_mode: str = "off",
        foot_seed_deg: Mapping[str, float] | None = None,
    ) -> BaseSpecProvenance:
        """Provenance of the unseeded ``document`` (the one hashing helper)."""
        if not isinstance(document, Mapping):
            raise TypeError("document must be a mapping")
        seed = None if foot_seed_deg is None else dict(foot_seed_deg)
        return cls(canonical_sha256(document), foot_progression_mode, seed)

    def receipt_fields(self) -> dict[str, Any]:
        """Top-level receipt fields carrying this provenance."""
        seed = self.foot_seed_deg
        return {
            "base_spec_sha256": self.sha256,
            "foot_progression_mode": self.foot_progression_mode,
            "foot_seed_deg": None if seed is None else dict(seed),
        }


@dataclass(frozen=True)
class GroundSupportReceiptInputs:
    """Inputs required to assemble a ground-support execution receipt."""

    base_spec: dict[str, Any]
    spec_path: Path
    scaled_path: Path
    hipcal_path: Path
    recalibrate_upper: bool
    anthropometric: tuple[float, float] | None
    qualification_note: str
    spec_bytes: bytes
    hip_report: dict[str, Any]
    candidate_bytes: bytes
    c3d_path: Path
    capture_name: str
    lane: Lane
    address_report: dict[str, Any]
    ik_report: dict[str, Any]
    dynamics_report: dict[str, Any]
    kin: FullBodyMarkerKinematics
    q_ref: np.ndarray
    elapsed_s: float
    backend: str = "mujoco"
    ik_backend: str = "lm"
    tracking_backend: str = "kkt"
    validate: bool = True
    provenance: BaseSpecProvenance | None = None


def _spec_canonical_sha256(spec_bytes: bytes) -> str:
    """Canonical digest of UTF-8 JSON specification bytes."""
    try:
        document = json.loads(spec_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "spec_bytes must be UTF-8 JSON to compute spec_canonical_sha256"
        ) from exc
    if not isinstance(document, dict):
        raise ValueError("spec_bytes JSON root must be an object")
    return canonical_sha256(document)


def build_ground_support_receipt(
    inputs: GroundSupportReceiptInputs,
) -> dict[str, Any]:
    """Assemble the standardized receipt for ground-supported full-body pipeline.

    Args:
        inputs: Context bundle containing spec paths, reports, lane, and kinematics.

    Returns:
        Structured receipt dictionary matching GS milestone schema.
    """
    lane = inputs.lane
    kin = inputs.kin
    q_ref = inputs.q_ref
    provenance = inputs.provenance

    stance_fraction = {
        name: float(np.mean([name in s for s in lane.stance]))
        for name in ("heel_r", "forefoot_r", "toe_r", "heel_l", "forefoot_l", "toe_l")
    }

    # Top of backswing posture at ~0.83 s
    tob_frame = min(int(round(0.83 * lane.rate_hz)), len(q_ref) - 1)
    tob_posture = posture_summary(kin, q_ref[tob_frame])

    receipt_dict = {
        "backend": inputs.backend,
        "ik_backend": inputs.ik_backend,
        "tracking_backend": inputs.tracking_backend,
        **(provenance or BaseSpecProvenance.of(inputs.base_spec)).receipt_fields(),
        "base_spec_file": inputs.spec_path.name,
        "spec_file": inputs.scaled_path.name,
        "hipcal_spec_file": inputs.hipcal_path.name,
        "recalibrate_upper": bool(inputs.recalibrate_upper),
        "anthropometric": (
            list(inputs.anthropometric) if inputs.anthropometric else None
        ),
        "de_leva_table_sha256": (
            inputs.base_spec.get("de_leva_table_sha256")
            or inputs.base_spec.get("subject", {}).get("de_leva_table_sha256")
        ),
        "posture_top_of_backswing": tob_posture,
        "spec_sha256": hashlib.sha256(inputs.spec_bytes).hexdigest(),
        "spec_canonical_sha256": _spec_canonical_sha256(inputs.spec_bytes),
        "hip_calibration": inputs.hip_report,
        "candidate_sha256": hashlib.sha256(inputs.candidate_bytes).hexdigest(),
        "capture": inputs.capture_name,
        "capture_sha256": hashlib.sha256(inputs.c3d_path.read_bytes()).hexdigest(),
        "club": inputs.base_spec.get("club"),
        "grip_rotation_deg": inputs.base_spec.get("subject", {}).get(
            "grip_rotation_deg"
        ),
        "wrists_bounded": any(name in lane.bounds for name in IK_UNBOUNDED),
        "labels": list(lane.labels),
        "ground": {
            "height_m": lane.ground_cal.height_m,
            "lowest_toe_marker_m": lane.ground_cal.lowest_marker_height_m,
            "standoff_m": lane.ground_cal.standoff_m,
            "policy": lane.ground_cal.policy,
            "stance_tolerance_m": STANCE_TOLERANCE_M,
            "stance_rule": "heights relative to address; both feet flat at frame 0",
            "toe_spheres": {
                k: {"body": b, "position_m": list(p), "radius_m": r}
                for k, (b, p, r) in TOE_SPHERES.items()
            },
            "stance_fraction": stance_fraction,
        },
        "address": inputs.address_report,
        "ik": inputs.ik_report,
        "dynamics": inputs.dynamics_report,
        "elapsed_s": inputs.elapsed_s,
        "qualification": (
            "kinematic IK and computed-torque tracking milestone on the MuJoCo "
            f"full-body model ({inputs.qualification_note}); not a fit, not acceptance"
        ),
    }

    # Evaluate physical & kinematic acceptance under Matched Swing Program contract (MS-01)
    verdict = evaluate(receipt_dict, horizon=Horizon.G3)
    receipt_dict["acceptance"] = verdict.as_dict()

    if inputs.validate:
        validate_receipt(receipt_dict)

    return receipt_dict


def log_pipeline_summary(
    log: logging.Logger,
    receipt: Mapping[str, Any],
    ik_report: Mapping[str, Any],
    calibration: Any,
    calibration2: Any,
) -> None:
    """Log formatted summaries of address, dynamics, IK, and calibration.

    Args:
        log: Logger instance.
        receipt: Complete ground support receipt dictionary.
        ik_report: Full IK stage report dictionary.
        calibration: Initial leg calibration result object.
        calibration2: Recalibration result object after segment scaling.
    """
    log.info(
        json.dumps(
            {k: receipt[k] for k in ("address", "dynamics")}, indent=1, default=float
        )
    )
    log.info(
        "ik %s",
        json.dumps(
            {
                k: ik_report[k]
                for k in (
                    "marker_rms_m",
                    "segment_rms_m",
                    "reference",
                    "segment_scaling",
                    "leg_angle_ranges_deg",
                )
            },
            indent=1,
            default=float,
        ),
    )
    log.info(
        "calibration rms %s -> scaled %s",
        calibration.rms_per_iteration_m,
        calibration2.rms_per_iteration_m,
    )
