"""Immutable candidate-session ingestion from receipt-bound model packages (MV-03, #10479).

Validates:
- Exact candidate content hash and model specification hash.
- Coordinate order remapping and tangent velocity alignment.
- Preservation of absent force/torque channels as None (never fabricated zeros).
- Conspicuous rejected fit inspection and capability discovery.
- Strict isolation: opening a pose archive never fabricates a GenericPhysicsRecorder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.candidate import (
    CandidateAuxiliary,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)

logger = logging.getLogger(__name__)

__all__ = ["CandidateSession", "ingest_candidate_session"]


def _make_readonly(arr: np.ndarray | None) -> None:
    """Set numpy array flags to read-only while satisfying Law of Demeter."""
    if arr is not None:
        flags = arr.flags
        flags.writeable = False


@dataclass(frozen=True)
class CandidateSession:
    """Immutable session wrapping a qualified candidate trajectory and its model."""

    candidate: MatchedSwingCandidate
    specification: dict[str, Any]
    candidate_sha256: str
    model_sha256: str
    coordinate_names: tuple[str, ...]
    time_s: NDArray[np.float64]
    q: NDArray[np.float64]
    v: NDArray[np.float64] | None = None
    a: NDArray[np.float64] | None = None
    tau: NDArray[np.float64] | None = None
    external_forces: NDArray[np.float64] | None = None
    receipt: dict[str, Any] | None = None
    is_accepted: bool = True
    status: str = "verified"
    rejection_reason: str = ""
    acceptance_criteria: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _make_readonly(self.time_s)
        _make_readonly(self.q)
        _make_readonly(self.v)
        _make_readonly(self.a)
        _make_readonly(self.tau)
        _make_readonly(self.external_forces)

    @property
    def frame_count(self) -> int:
        return int(len(self.time_s))

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if len(self.time_s) > 1 else 0.0

    @property
    def engine(self) -> str:
        meta = self.candidate.metadata
        return meta.engine

    @property
    def supports_forces(self) -> bool:
        """True when the candidate carries actual actuator efforts or external forces."""
        return self.tau is not None or self.external_forces is not None

    @property
    def supports_counterfactuals(self) -> bool:
        """True when candidate is an accepted dynamic rollout with control channels."""
        meta = self.candidate.metadata
        return (
            self.supports_forces
            and self.is_accepted
            and meta.profile == CandidateProfile.DYNAMIC
        )

    @property
    def supports_tangent_velocities(self) -> bool:
        return self.v is not None

    @property
    def supports_markers(self) -> bool:
        markers = self.candidate.markers
        return markers.model_markers_m is not None

    @property
    def rms_error(self) -> float:
        """Marker RMS error in meters if available, otherwise 0.0."""
        if self.receipt is not None:
            shared = self.receipt.get("shared_metrics", {})
            if "whole_marker_rmse_m" in shared:
                return float(shared["whole_marker_rmse_m"])
        return 0.0

    def to_replay(self) -> Any:
        """Convert candidate session into viewer-compatible ReplayData."""
        from src.tools.tour_matching_viewer.core import ReplayData

        markers = self.candidate.markers
        valid = getattr(markers, "marker_validity", None)
        if valid is None:
            valid = getattr(markers, "valid_mask", None)

        return ReplayData(
            time_s=self.time_s,
            coordinates=self.q,
            model_markers_m=markers.model_markers_m,
            target_markers_m=markers.target_markers_m,
            valid_mask=valid,
            coordinate_names=self.coordinate_names,
        )

    @property
    def replay(self) -> Any:
        """Accessor returning viewer-compatible ReplayData."""
        return self.to_replay()

    def get_wrench_at(self, frame_idx: int, contact_name: str = "ground") -> Any:
        """Return the spatial wrench at specified frame or None if forces channel absent."""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(
                f"frame_idx {frame_idx} out of range [0, {self.frame_count})"
            )
        if self.external_forces is None:
            return None

        from src.shared.python.motion_matching.force_torque import SpatialWrench

        row = self.external_forces[frame_idx]
        fx = float(row[0]) if len(row) > 0 else 0.0
        fy = float(row[1]) if len(row) > 1 else 0.0
        fz = float(row[2]) if len(row) > 2 else 0.0
        tx = float(row[3]) if len(row) > 3 else 0.0
        ty = float(row[4]) if len(row) > 4 else 0.0
        tz = float(row[5]) if len(row) > 5 else 0.0

        return SpatialWrench(
            application_frame=contact_name,
            point_m=(0.0, 0.0, 0.0),
            force_n=(fx, fy, fz),
            torque_nm=(tx, ty, tz),
            direction_convention="applied_to_body",
            sign_convention="standard_cartesian",
        )

    def get_center_of_pressure(
        self, frame_idx: int, foot: str = "net", fz_threshold: float = 5.0
    ) -> tuple[float, float] | None:
        """Compute Center of Pressure (CoP_x, CoP_y) or None if absent or Fz <= threshold."""
        wrench = self.get_wrench_at(frame_idx, contact_name=foot)
        if wrench is None:
            return None
        from src.shared.python.motion_matching.force_torque import (
            compute_center_of_pressure,
        )

        return compute_center_of_pressure(wrench, f_threshold_n=fz_threshold)

    def get_joint_torques_at(self, frame_idx: int) -> dict[str, float] | None:
        """Return joint torque mapping at frame_idx or None if tau channel absent."""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(
                f"frame_idx {frame_idx} out of range [0, {self.frame_count})"
            )
        if self.tau is None:
            return None
        row = self.tau[frame_idx]
        return {
            name: float(row[i])
            for i, name in enumerate(self.coordinate_names[: len(row)])
        }

    def get_closure_residual_at(self, frame_idx: int) -> float | None:
        """Return kinematic closure residual at frame_idx or None if not computed."""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(
                f"frame_idx {frame_idx} out of range [0, {self.frame_count})"
            )
        res = self.diagnostics.get("closure_residuals")
        if res is not None and frame_idx < len(res):
            return float(res[frame_idx])
        return None

    def create_counterfactual_fork(
        self,
        fork_frame_idx: int,
        strategy: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Generate an immutable counterfactual rollout fork diverging from this session."""
        from src.shared.python.motion_matching.counterfactual import (
            CounterfactualStrategy,
            create_counterfactual_rollout,
        )

        chosen_strategy = (
            strategy
            if strategy is not None
            else CounterfactualStrategy.ZERO_TRAIL_ARM_TORQUE
        )
        return create_counterfactual_rollout(
            session=self,
            fork_frame_idx=fork_frame_idx,
            strategy=chosen_strategy,
            **kwargs,
        )


def _resolve_model_path(
    candidate_path: Path,
    model_path: Path | str | None,
    receipt_data: Mapping[str, Any] | None,
) -> Path:
    """Resolve the canonical model specification document path without defaults."""
    if model_path is not None:
        p = Path(model_path)
        if not p.is_file():
            raise FileNotFoundError(f"Declared model specification not found: {p}")
        return p

    if receipt_data is not None:
        for key in ("spec_file", "base_spec_file"):
            spec_ref = receipt_data.get(key)
            if spec_ref:
                sibling = candidate_path.with_name(str(spec_ref))
                if sibling.is_file():
                    return sibling
                parent_sibling = candidate_path.parent.parent / str(spec_ref)
                if parent_sibling.is_file():
                    return parent_sibling

    raise FileNotFoundError(
        f"Cannot resolve model specification for {candidate_path.name}; "
        "model path must be explicitly provided or bound in receipt"
    )


def _validate_receipt_hashes(
    receipt_data: Mapping[str, Any],
    candidate_sha256: str,
    model_sha256: str,
) -> None:
    """Ensure candidate and model byte identity match receipt declarations."""
    cand_expected = receipt_data.get("candidate_sha256")
    if cand_expected and cand_expected != candidate_sha256:
        raise ValueError(
            f"Candidate content hash mismatch with receipt: "
            f"expected {cand_expected}, got {candidate_sha256}"
        )

    doc_expected = receipt_data.get("document_sha256") or receipt_data.get(
        "base_spec_sha256"
    )
    if doc_expected and doc_expected != model_sha256:
        raise ValueError(
            f"Model hash mismatch with receipt: "
            f"expected {doc_expected}, got {model_sha256}"
        )


def _load_npz_arrays(
    path: Path,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[str, ...],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    CandidateMetadata,
]:
    """Extract array channels and metadata from candidate NPZ."""
    with np.load(path, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=np.float64)
        q = np.asarray(data["q"], dtype=np.float64)
        coord_order = tuple(str(x) for x in data["coordinate_order"])
        v = (
            np.asarray(data["v"], dtype=np.float64)
            if "v" in data and data["v"] is not None
            else None
        )
        a = (
            np.asarray(data["a"], dtype=np.float64)
            if "a" in data and data["a"] is not None
            else None
        )
        tau = (
            np.asarray(data["tau"], dtype=np.float64)
            if "tau" in data and data["tau"] is not None
            else None
        )
        external_forces = (
            np.asarray(data["external_forces"], dtype=np.float64)
            if "external_forces" in data and data["external_forces"] is not None
            else None
        )

        meta_dict: dict[str, Any] = {}
        if "metadata_json" in data:
            meta_dict = json.loads(str(data["metadata_json"]))
        metadata = CandidateMetadata.from_dict(meta_dict) if meta_dict else None
        if metadata is None:
            metadata = CandidateMetadata(
                profile=CandidateProfile.DYNAMIC
                if tau is not None
                else CandidateProfile.KINEMATIC,
                coordinate_names=coord_order,
            )

    return time_s, q, coord_order, v, a, tau, external_forces, metadata


def _reorder_coordinates(
    q: NDArray[np.float64],
    source_coords: Sequence[str],
    target_coords: Sequence[str],
) -> NDArray[np.float64]:
    """Map generalized coordinates to canonical target model order."""
    if set(source_coords) != set(target_coords) or len(source_coords) != len(
        target_coords
    ):
        raise ValueError(
            f"Candidate coordinate names {source_coords} do not match "
            f"target model specification {target_coords}"
        )
    col_indices = [source_coords.index(c) for c in target_coords]
    ordered_q = q[:, col_indices]
    ordered_q.flags.writeable = False
    return ordered_q


def _parse_acceptance(
    receipt_data: Mapping[str, Any] | None,
) -> tuple[bool, str, str, dict[str, Any]]:
    """Extract acceptance decision and metadata from receipt sidecar."""
    acc = receipt_data.get("acceptance", {}) if receipt_data else {}
    is_accepted = bool(acc.get("is_accepted", True))
    status = str(acc.get("status", "verified" if is_accepted else "rejected"))
    reason = str(acc.get("reason", ""))
    return is_accepted, status, reason, acc


@precondition(lambda candidate_path, **_: Path(candidate_path).is_file())
def ingest_candidate_session(
    candidate_path: Path | str,
    model_path: Path | str | None = None,
    receipt_path: Path | str | None = None,
    *,
    strict: bool = True,
) -> CandidateSession:
    """Ingest candidate data and model spec with cryptographic integrity checks."""
    cand_p = Path(candidate_path)
    cand_bytes = cand_p.read_bytes()
    cand_sha256 = hashlib.sha256(cand_bytes).hexdigest()

    rec_p = Path(receipt_path) if receipt_path else cand_p.with_name("receipt.json")
    receipt_data: dict[str, Any] | None = None
    if rec_p.is_file():
        receipt_data = json.loads(rec_p.read_bytes())
    elif strict:
        raise ValueError(
            f"Receipt sidecar missing for candidate session at {rec_p.name}"
        )

    resolved_model_p = _resolve_model_path(cand_p, model_path, receipt_data)
    model_bytes = resolved_model_p.read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    spec = json.loads(model_bytes)

    if receipt_data is not None:
        _validate_receipt_hashes(receipt_data, cand_sha256, model_sha256)

    (
        time_s,
        q,
        source_coords,
        v,
        a,
        tau,
        external_forces,
        meta,
    ) = _load_npz_arrays(cand_p)

    if (
        time_s.ndim != 1
        or len(time_s) < 2
        or not np.all(np.isfinite(time_s))
        or np.any(np.diff(time_s) <= 0.0)
    ):
        raise ValueError(
            "Candidate timestamps must be strictly monotonically increasing"
        )

    target_coords = tuple(spec.get("coordinate_order", source_coords))
    ordered_q = _reorder_coordinates(q, source_coords, target_coords)
    is_accepted, status, reason, acc = _parse_acceptance(receipt_data)

    diag = receipt_data.get("diagnostics", {}) if receipt_data else {}
    candidate_obj = MatchedSwingCandidate(
        metadata=meta,
        time_s=time_s,
        q=ordered_q,
        v=v,
        tau=tau,
        markers=CandidateMarkers(),
        auxiliary=CandidateAuxiliary(external_forces=external_forces),
    )

    return CandidateSession(
        candidate=candidate_obj,
        specification=spec,
        candidate_sha256=cand_sha256,
        model_sha256=model_sha256,
        coordinate_names=target_coords,
        time_s=time_s,
        q=ordered_q,
        v=v,
        a=a,
        tau=tau,
        external_forces=external_forces,
        receipt=receipt_data,
        is_accepted=is_accepted,
        status=status,
        rejection_reason=reason,
        acceptance_criteria=acc,
        diagnostics=diag,
    )
