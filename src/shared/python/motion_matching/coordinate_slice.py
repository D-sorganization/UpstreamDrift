"""Physically justified 44-to-27 coordinate slice for Simscape upper-body replay (MS-62, #10349)."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.candidate import (
    CandidateAuxiliary,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_io import save_candidate

logger = logging.getLogger(__name__)

SLICE_MAP_SCHEMA_VERSION = "coordinate-slice-v1"
INCHES_PER_M = 39.37007874015748
Array: TypeAlias = NDArray[np.float64]

_DEFAULT_SLICE_MAP = (
    Path(__file__).resolve().parents[4]
    / "evidence/matched/driver_g1_simscape_slice/slice_map.json"
)


@dataclass(frozen=True)
class SliceMap:
    """JSON-backed 44-to-27 coordinate, velocity and effort map."""

    schema_version: str
    source_coordinates: tuple[str, ...]
    target_coordinates: tuple[str, ...]
    coordinate_indices: dict[str, int]
    omitted: dict[str, dict[str, Any]]
    qualification: str = ""

    @property
    def source_coordinate_count(self) -> int:
        return len(self.source_coordinates)

    @property
    def target_coordinate_count(self) -> int:
        return len(self.target_coordinates)

    @property
    def omitted_count(self) -> int:
        return len(self.omitted)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SliceMap:
        version = str(data.get("schema_version", ""))
        if version != SLICE_MAP_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported slice map schema {version!r} "
                f"(expected {SLICE_MAP_SCHEMA_VERSION!r})"
            )
        source = tuple(str(x) for x in data["source_coordinates"])
        target = tuple(str(x) for x in data["target_coordinates"])
        indices = {str(k): int(v) for k, v in data["coordinate_indices"].items()}
        omitted = {str(k): dict(v) for k, v in data.get("omitted", {}).items()}
        cls._validate(source, target, indices, omitted)
        return cls(
            schema_version=version,
            source_coordinates=source,
            target_coordinates=target,
            coordinate_indices=indices,
            omitted=omitted,
            qualification=str(data.get("qualification", "")),
        )

    @staticmethod
    def _validate(
        source: Sequence[str],
        target: Sequence[str],
        indices: Mapping[str, int],
        omitted: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if len(set(source)) != len(source):
            raise ValueError("Duplicate source coordinate names")
        if len(set(target)) != len(target):
            raise ValueError("Duplicate target coordinate names")
        if set(indices) != set(target):
            raise ValueError("coordinate_indices must cover every target coordinate")
        for name, idx in indices.items():
            if source[idx] != name:
                raise ValueError(
                    f"coordinate_indices[{name!r}]={idx} does not match source"
                )
        unknown = set(omitted) - set(source)
        if unknown:
            raise ValueError(
                f"omitted coordinates not in source list: {sorted(unknown)}"
            )
        overlap = set(omitted) & set(target)
        if overlap:
            raise ValueError(
                f"omitted coordinates overlap target set: {sorted(overlap)}"
            )


@precondition(lambda path: Path(path).is_file(), "slice map JSON must exist")
@postcondition(lambda r: isinstance(r, SliceMap), "must return SliceMap")
def load_slice_map(path: Path | str) -> SliceMap:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SliceMap.from_dict(data)


def _check_source_names(source_names: Sequence[str], slice_map: SliceMap) -> None:
    if tuple(source_names) != slice_map.source_coordinates:
        raise ValueError(
            "candidate coordinate names disagree with slice map source list "
            f"(expected {len(slice_map.source_coordinates)}, got {len(source_names)})"
        )


@precondition(
    lambda q, v, source_names, slice_map: q.shape == v.shape and q.ndim == 2,
    "q and v must be 2-D arrays with matching shape",
)
def project_kinematic_trajectory(
    q: Array,
    v: Array,
    source_names: Sequence[str],
    slice_map: SliceMap,
) -> tuple[Array, Array, list[str]]:
    _check_source_names(source_names, slice_map)
    n_frames = q.shape[0]
    n_tgt = slice_map.target_coordinate_count
    q_out = np.zeros((n_frames, n_tgt), dtype=np.float64)
    v_out = np.zeros((n_frames, n_tgt), dtype=np.float64)
    for tgt_idx, name in enumerate(slice_map.target_coordinates):
        src_idx = slice_map.coordinate_indices[name]
        q_out[:, tgt_idx] = q[:, src_idx]
        v_out[:, tgt_idx] = v[:, src_idx]
    return q_out, v_out, list(slice_map.target_coordinates)


@precondition(
    lambda tau, v, source_names, slice_map: tau.shape == v.shape and tau.ndim == 2,
    "tau and v must be 2-D arrays with matching shape",
)
def derive_boundary_wrenches(
    tau: Array,
    v: Array,
    source_names: Sequence[str],
    slice_map: SliceMap,
) -> Array:
    _check_source_names(source_names, slice_map)
    n_frames = tau.shape[0]
    boundary = np.zeros((n_frames, 6), dtype=np.float64)
    name_to_idx = {name: idx for idx, name in enumerate(source_names)}
    for omitted_name, spec in slice_map.omitted.items():
        src_idx = name_to_idx[omitted_name]
        axis = int(spec.get("axis", 0)) % 6
        boundary[:, axis] += tau[:, src_idx]
    return boundary


def check_slice_virtual_work(
    v: Array,
    tau: Array,
    source_names: Sequence[str],
    slice_map: SliceMap,
    *,
    tolerance: float = 1e-6,
) -> tuple[bool, float]:
    v = np.asarray(v, dtype=np.float64).reshape(1, -1) if v.ndim == 1 else v
    tau = np.asarray(tau, dtype=np.float64).reshape(1, -1) if tau.ndim == 1 else tau
    _check_source_names(source_names, slice_map)
    _, v27, _ = project_kinematic_trajectory(v, v, source_names, slice_map)
    tau27 = np.zeros_like(v27)
    for tgt_idx, name in enumerate(slice_map.target_coordinates):
        src_idx = slice_map.coordinate_indices[name]
        tau27[:, tgt_idx] = tau[:, src_idx]
    name_to_idx = {name: idx for idx, name in enumerate(source_names)}
    omitted_idx = [name_to_idx[name] for name in slice_map.omitted]
    power_full = float(np.sum(tau * v))
    power_retained = float(np.sum(tau27 * v27))
    power_omitted = float(np.sum(tau[:, omitted_idx] * v[:, omitted_idx]))
    residual = abs(power_full - (power_retained + power_omitted))
    return bool(residual <= tolerance), residual


def slice_candidate(
    candidate: MatchedSwingCandidate,
    out: Path | str | None = None,
    *,
    slice_map: SliceMap,
) -> dict[str, Any] | MatchedSwingCandidate:
    _check_source_names(candidate.metadata.coordinate_names, slice_map)
    if candidate.v is None:
        raise ValueError("candidate lacks velocity array required for slice")
    source_names = candidate.metadata.coordinate_names
    q27, v27, names27 = project_kinematic_trajectory(
        candidate.q, candidate.v, source_names, slice_map
    )
    tau27 = None
    boundary = None
    max_vw_residual = 0.0
    if candidate.tau is not None and candidate.tau.shape[1] == candidate.q.shape[1]:
        tau27 = np.zeros_like(q27)
        for tgt_idx, name in enumerate(slice_map.target_coordinates):
            src_idx = slice_map.coordinate_indices[name]
            tau27[:, tgt_idx] = candidate.tau[:, src_idx]
        boundary = derive_boundary_wrenches(
            candidate.tau, candidate.v, source_names, slice_map
        )
        _, max_vw_residual = check_slice_virtual_work(
            candidate.v[0], candidate.tau[0], source_names, slice_map
        )
    profile = candidate.metadata.profile
    missing_fields = list(candidate.metadata.missing_fields)
    if tau27 is None and profile == CandidateProfile.DYNAMIC:
        profile = CandidateProfile.KINEMATIC
        if "tau" not in missing_fields:
            missing_fields.append("tau")
    meta = CandidateMetadata(
        profile=profile,
        engine="simscape_slice",
        model_name="simscape_native_27",
        coordinate_names=tuple(names27),
        velocity_names=tuple(names27),
        actuator_names=tuple(names27) if tau27 is not None else (),
        marker_names=candidate.metadata.marker_names,
        missing_fields=missing_fields,
        extra={
            **candidate.metadata.extra,
            "slice_map_schema": slice_map.schema_version,
            "omitted_coordinates": sorted(slice_map.omitted),
        },
    )
    auxiliary = CandidateAuxiliary(
        actuator_states=candidate.auxiliary.actuator_states,
        external_forces=candidate.auxiliary.external_forces,
        root_forces=boundary,
        contact_modes=candidate.auxiliary.contact_modes,
        grip_wrench=candidate.auxiliary.grip_wrench,
    )
    sliced = MatchedSwingCandidate(
        metadata=meta,
        time_s=candidate.time_s,
        q=q27,
        v=v27,
        tau=tau27,
        markers=candidate.markers,
        auxiliary=auxiliary,
    )
    receipt = {
        "schema_version": "coordinate-slice-receipt-v1",
        "issue": "#10349",
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "source_coordinate_count": slice_map.source_coordinate_count,
        "target_coordinate_count": slice_map.target_coordinate_count,
        "omitted_count": slice_map.omitted_count,
        "virtual_work_max_residual": max_vw_residual,
        "qualification": slice_map.qualification,
    }
    if out is None:
        return sliced
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_candidate(sliced, out_path)
    receipt["candidate_npz"] = out_path.name
    return receipt


@precondition(
    lambda document_path: Path(document_path).is_file(), "geometry document must exist"
)
def workspace_overrides_from_geometry_document(
    document_path: Path | str,
) -> dict[str, float]:
    doc = json.loads(Path(document_path).read_text(encoding="utf-8"))
    segments = doc.get("anthropometry", {}).get("segments", {})
    if not segments:
        raise ValueError("document lacks anthropometry.segments")
    upper_arm_in = float(segments["upper_arm"]["length_m"]) * INCHES_PER_M
    return {
        "UpperArmLength": upper_arm_in,
        "LeftUpperArmLength": upper_arm_in,
        "RightUpperArmLength": upper_arm_in,
        "LowerArmLength": float(segments["forearm"]["length_m"]) * INCHES_PER_M,
        "UpperTorsoLength": float(segments["trunk"]["length_m"]) * 0.45 * INCHES_PER_M,
    }


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project a 44-DOF candidate onto the 27-coordinate Simscape slice."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--map", type=Path, default=_DEFAULT_SLICE_MAP)
    args = parser.parse_args(argv)
    from src.shared.python.motion_matching.candidate_convert import (
        convert_analytic_matched_npz,
    )

    candidate = convert_analytic_matched_npz(args.candidate, engine="legacy_npz")
    receipt = slice_candidate(candidate, args.out, slice_map=load_slice_map(args.map))
    assert isinstance(receipt, dict)
    receipt_path = args.out.parent / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
