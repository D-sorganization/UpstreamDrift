"""Converters from legacy simulation and IK formats to MatchedSwingCandidate packages (MS-15, #10334).

Supports:
1. Historical returned81 cross-engine replay archives (*_returned81_replay.npz).
2. OpenSim motion files (.mot / .sto).
3. Ground-support IK archives (ik_trajectory.npz).
4. Ground-support dynamics archives (dynamics_record.npz).

Preserves uncertainty, validity masks, and records explicit missing-field annotations;
never invents synthetic dynamics for kinematic artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateAuxiliary,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)

logger = logging.getLogger(__name__)

TRANSLATIONAL_SUBSTRINGS = ("translation", "pos_x", "pos_y", "pos_z", "tx", "ty", "tz")


def _extract_optional_array(
    data: Any, key: str, dtype: Any = np.float64
) -> np.ndarray | None:
    return np.asarray(data[key], dtype=dtype) if key in data else None


@precondition(
    lambda npz_path, spec=None, engine="unknown": Path(npz_path).is_file(),
    "replay NPZ file must exist",
)
@postcondition(
    lambda r: isinstance(r, MatchedSwingCandidate), "must return MatchedSwingCandidate"
)
def convert_returned81_replay(
    npz_path: Path | str,
    spec: Mapping[str, Any] | None = None,
    engine: str = "unknown",
) -> MatchedSwingCandidate:
    """Convert a returned81 replay archive to MatchedSwingCandidate format."""
    p = Path(npz_path)
    with np.load(p, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=np.float64)
        native_state = np.asarray(data["native_state"], dtype=np.float64)
        markers_m = _extract_optional_array(data, "markers_m")
        target_m = _extract_optional_array(data, "target_m")
        valid = _extract_optional_array(data, "valid", dtype=bool)

    nq = native_state.shape[1] // 2
    q = native_state[:, :nq]
    v = native_state[:, nq:]

    coord_names: tuple[str, ...]
    if spec and "coordinate_order" in spec:
        coords = spec["coordinate_order"]
        coord_names = tuple(coords[:nq])
    else:
        coord_names = tuple(f"q_{i}" for i in range(nq))

    vel_names: tuple[str, ...] = tuple(f"v_{i}" for i in range(nq))

    marker_names: tuple[str, ...] = ()
    if markers_m is not None:
        if spec and "markers" in spec:
            marker_names = tuple(spec["markers"])[: markers_m.shape[1]]
        else:
            marker_names = tuple(f"marker_{i}" for i in range(markers_m.shape[1]))

    # Classified as kinematic because controls (tau) were not captured; record missing fields
    metadata = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.KINEMATIC,
        engine=engine,
        model_name=str(
            spec.get("name", "returned81_model") if spec else "returned81_model"
        ),
        model_sha256=str(spec.get("sha256", "") if spec else ""),
        coordinate_names=coord_names,
        velocity_names=vel_names,
        marker_names=marker_names,
        missing_fields=["tau", "actuator_controls"],
        extra={"source_file": p.name, "legacy_format": "returned81_npz"},
    )

    return MatchedSwingCandidate(
        metadata=metadata,
        time_s=time_s,
        q=q,
        v=v,
        tau=None,
        markers=CandidateMarkers(
            model_markers_m=markers_m,
            target_markers_m=target_m,
            marker_validity=valid,
        ),
    )


@precondition(
    lambda mot_path, spec=None, engine="opensim": Path(mot_path).is_file(),
    "OpenSim MOT file must exist",
)
@postcondition(
    lambda r: isinstance(r, MatchedSwingCandidate), "must return MatchedSwingCandidate"
)
def convert_opensim_mot(
    mot_path: Path | str,
    spec: Mapping[str, Any] | None = None,
    engine: str = "opensim",
) -> MatchedSwingCandidate:
    """Convert an OpenSim .mot or .sto motion file to a versioned MatchedSwingCandidate."""
    p = Path(mot_path)
    in_degrees = False
    in_header = True
    col_names: list[str] = []
    data_rows: list[list[float]] = []

    with open(p, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if in_header:
                lower = stripped.lower()
                if lower.startswith(("indegrees=", "in_degrees=")):
                    val = lower.split("=")[1].strip()
                    in_degrees = val in ("yes", "true", "1")
                elif lower == "endheader":
                    in_header = False
                continue

            # First row after header is column names
            if not col_names:
                col_names = stripped.split()
                continue

            parts = stripped.split()
            data_rows.append([float(x) for x in parts])

    if not col_names or not data_rows:
        raise ValueError(
            f"OpenSim motion file {p} contains no data rows or column headers"
        )

    raw_data = np.asarray(data_rows, dtype=np.float64)
    # First column is time
    time_s = raw_data[:, 0]
    coord_names = tuple(col_names[1:])
    q_data = raw_data[:, 1:].copy()

    # Convert degrees to radians for rotational coordinates
    if in_degrees:
        for idx, name in enumerate(coord_names):
            is_trans = any(sub in name.lower() for sub in TRANSLATIONAL_SUBSTRINGS)
            if not is_trans:
                q_data[:, idx] = np.deg2rad(q_data[:, idx])

    metadata = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.KINEMATIC,
        engine=engine,
        model_name=str(spec.get("name", "opensim_model") if spec else "opensim_model"),
        coordinate_names=coord_names,
        missing_fields=["v", "tau", "actuator_controls"],
        extra={"source_file": p.name, "in_degrees_original": in_degrees},
    )

    return MatchedSwingCandidate(
        metadata=metadata,
        time_s=time_s,
        q=q_data,
        v=None,
        tau=None,
    )


@precondition(
    lambda npz_path, spec=None, engine="mujoco": Path(npz_path).is_file(),
    "IK NPZ file must exist",
)
@postcondition(
    lambda r: isinstance(r, MatchedSwingCandidate), "must return MatchedSwingCandidate"
)
def convert_ground_support_ik(
    npz_path: Path | str,
    spec: Mapping[str, Any] | None = None,
    engine: str = "mujoco",
) -> MatchedSwingCandidate:
    """Convert a ground-support IK trajectory archive to a MatchedSwingCandidate."""
    p = Path(npz_path)
    with np.load(p, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=np.float64)
        q = np.asarray(data["q"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool) if "valid" in data else None

    nq = q.shape[1]
    coord_names = (
        tuple(spec["coordinate_order"][:nq])
        if spec and "coordinate_order" in spec
        else tuple(f"q_{i}" for i in range(nq))
    )

    metadata = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.KINEMATIC,
        engine=engine,
        model_name=str(spec.get("name", "gs_ik_model") if spec else "gs_ik_model"),
        coordinate_names=coord_names,
        missing_fields=["v", "tau", "actuator_controls"],
        extra={"source_file": p.name},
    )

    return MatchedSwingCandidate(
        metadata=metadata,
        time_s=time_s,
        q=q,
        v=None,
        markers=CandidateMarkers(marker_validity=valid),
    )


@precondition(
    lambda npz_path, spec=None, engine="mujoco": Path(npz_path).is_file(),
    "dynamics NPZ file must exist",
)
@postcondition(
    lambda r: isinstance(r, MatchedSwingCandidate), "must return MatchedSwingCandidate"
)
def convert_ground_support_dynamics(
    npz_path: Path | str,
    spec: Mapping[str, Any] | None = None,
    engine: str = "mujoco",
) -> MatchedSwingCandidate:
    """Convert a ground-support forward dynamics trajectory to a dynamic MatchedSwingCandidate."""
    p = Path(npz_path)
    with np.load(p, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=np.float64)
        q = np.asarray(data["q"], dtype=np.float64)
        v = np.asarray(data["v"], dtype=np.float64)
        tau = np.asarray(data["tau"], dtype=np.float64)

    nq = q.shape[1]
    nv = v.shape[1]
    nu = tau.shape[1]

    coord_names = (
        tuple(spec["coordinate_order"][:nq])
        if spec and "coordinate_order" in spec
        else tuple(f"q_{i}" for i in range(nq))
    )
    vel_names = tuple(f"v_{i}" for i in range(nv))
    act_names = tuple(f"tau_{i}" for i in range(nu))

    metadata = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.DYNAMIC,
        engine=engine,
        model_name=str(
            spec.get("name", "gs_dynamics_model") if spec else "gs_dynamics_model"
        ),
        coordinate_names=coord_names,
        velocity_names=vel_names,
        actuator_names=act_names,
        extra={"source_file": p.name},
    )

    return MatchedSwingCandidate(
        metadata=metadata,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
    )


@precondition(
    lambda npz_path, spec=None, engine="pinocchio": Path(npz_path).is_file(),
    "analytic matched NPZ file must exist",
)
@postcondition(
    lambda r: isinstance(r, MatchedSwingCandidate), "must return MatchedSwingCandidate"
)
def convert_analytic_matched_npz(
    npz_path: Path | str,
    spec: Mapping[str, Any] | None = None,
    engine: str = "pinocchio",
) -> MatchedSwingCandidate:
    """Convert an analytic matched swing NPZ archive to a versioned MatchedSwingCandidate."""
    p = Path(npz_path)
    with np.load(p, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=np.float64)
        q = np.asarray(data["q"], dtype=np.float64)
        v = np.asarray(data["v"], dtype=np.float64) if "v" in data else None
        tau: np.ndarray | None = None
        if "u" in data:
            tau = np.asarray(data["u"], dtype=np.float64)
        elif "u_optimum" in data:
            tau = np.asarray(data["u_optimum"], dtype=np.float64)

        markers_m = _extract_optional_array(data, "markers_m")
        target_m = _extract_optional_array(data, "target_m")
        valid = _extract_optional_array(data, "valid", dtype=bool)
        grip_w = _extract_optional_array(data, "grip_wrenches")
        ext_f = _extract_optional_array(data, "ground_forces")

        coord_order: tuple[str, ...] = ()
        if "coordinate_order" in data:
            coord_order = tuple(str(x) for x in data["coordinate_order"])
        elif spec and "coordinate_order" in spec:
            coord_order = tuple(spec["coordinate_order"])
        else:
            coord_order = tuple(f"q_{i}" for i in range(q.shape[1]))

        marker_names: tuple[str, ...] = ()
        if "labels" in data:
            marker_names = tuple(str(x) for x in data["labels"])

    nv = v.shape[1] if v is not None else q.shape[1]
    nu = tau.shape[1] if tau is not None else 0
    vel_names = tuple(f"v_{i}" for i in range(nv))
    act_names = tuple(f"tau_{i}" for i in range(nu))

    metadata = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=(
            CandidateProfile.DYNAMIC if tau is not None else CandidateProfile.KINEMATIC
        ),
        engine=engine,
        model_name=str(
            spec.get("name", "full_body_pinocchio") if spec else "full_body_pinocchio"
        ),
        coordinate_names=coord_order,
        velocity_names=vel_names,
        actuator_names=act_names,
        marker_names=marker_names,
        missing_fields=["root_forces", "contact_modes"],
        extra={"source_file": p.name, "legacy_format": "analytic_matched_npz"},
    )

    markers = CandidateMarkers(
        model_markers_m=markers_m,
        target_markers_m=target_m,
        marker_validity=valid,
    )
    auxiliary = CandidateAuxiliary(
        external_forces=ext_f,
        grip_wrench=grip_w,
    )

    return MatchedSwingCandidate(
        metadata=metadata,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
        markers=markers,
        auxiliary=auxiliary,
    )
