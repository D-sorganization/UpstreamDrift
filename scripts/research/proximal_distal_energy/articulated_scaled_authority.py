"""Regenerate headline authority states under structural engineering bounds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.spatial_full_body import prescribed_state
from scripts.research.proximal_distal_energy.subject_scaled_closed_contact import (
    ClosedContactConfig,
    ClosedContactSolution,
    solve_closed_contact_configuration,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
SOURCE_PATHS = (
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "scripts/research/proximal_distal_energy/articulated_scaled_authority.py",
    "scripts/research/proximal_distal_energy/subject_scaled_closed_contact.py",
    "scripts/research/proximal_distal_energy/subject_scaled_spatial_geometry.py",
)


@dataclass(frozen=True, slots=True)
class ScaledAuthorityConfig:
    """Structural corner and selected full-atlas cases to regenerate."""

    case_indices: tuple[int, ...] = (0, 8, 9, 17)
    height_scale: float = 1.0
    body_mass_scale: float = 1.0
    joint_limit_scale: float = 1.0

    def __post_init__(self) -> None:
        if (
            not self.case_indices
            or len(set(self.case_indices)) != len(self.case_indices)
            or any(
                not isinstance(value, int) or not 0 <= value < 18
                for value in self.case_indices
            )
        ):
            raise ValueError(
                "case_indices must contain unique values from 0 through 17"
            )
        for name in ("height_scale", "body_mass_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.5 <= value <= 1.5:
                raise ValueError(f"{name} must be finite and lie in [0.5, 1.5]")
        if not np.isfinite(self.joint_limit_scale) or not (
            0.5 <= self.joint_limit_scale <= 1.5
        ):
            raise ValueError("joint_limit_scale must be finite and lie in [0.5, 1.5]")


@dataclass(frozen=True, slots=True)
class ScaledAuthority:
    """Regenerated states plus explicit feasibility and provenance evidence."""

    configuration: ScaledAuthorityConfig
    time_s: FloatArray
    profile_index: IntArray
    grip_span_m: FloatArray
    solution_q: FloatArray
    feasible: BoolArray
    selected_case_indices: IntArray
    selected_failure_class: NDArray[np.str_]
    selected_maximum_closure_error_m: FloatArray
    selected_minimum_joint_limit_margin_rad: FloatArray
    selected_minimum_collision_clearance_m: FloatArray
    maximum_nominal_state_error_rad: float
    source_sha256: dict[str, str]
    authority_sha256: str


def _failure_class(solution: ClosedContactSolution) -> str:
    if not solution.solver_converged:
        return "ik_nonconvergence"
    if not solution.contact_closed:
        return "bilateral_closure_failure"
    if not solution.joint_limits_satisfied:
        return "joint_limit_failure"
    if not solution.collision_free:
        return "collision_domain_failure"
    if solution.constraint_jacobian_rank != 6:
        return "constraint_rank_failure"
    return "feasible"


def _source_hashes() -> dict[str, str]:
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _digest_payload(
    configuration: ScaledAuthorityConfig,
    time_s: FloatArray,
    profile_index: IntArray,
    grip_span_m: FloatArray,
    solution_q: FloatArray,
    feasible: BoolArray,
    selected_case_indices: IntArray,
    selected_failure_class: NDArray[np.str_],
    source_sha256: dict[str, str],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"configuration": asdict(configuration), "source_sha256": source_sha256},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for array in (
        time_s,
        profile_index,
        grip_span_m,
        solution_q,
        feasible,
        selected_case_indices,
        selected_failure_class,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _nominal_authority() -> dict[str, NDArray[Any]]:
    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
        result = {
            "time_s": np.asarray(source["time_s"], dtype=float),
            "profile_index": np.asarray(source["case_profile_index"], dtype=np.int64),
            "grip_span_m": np.asarray(source["case_grip_span_m"], dtype=float),
            "solution_q": np.asarray(source["solution_q"], dtype=float),
            "feasible": np.asarray(source["feasible"], dtype=bool),
        }
    if (
        result["time_s"].shape != (13,)
        or result["profile_index"].shape != (18,)
        or result["grip_span_m"].shape != (18,)
        or result["solution_q"].shape != (18, 13, 20)
        or result["feasible"].shape != (18, 13)
        or not np.all(result["feasible"])
    ):
        raise RuntimeError("the nominal closed-state authority is incomplete")
    return result


def build_scaled_authority(
    configuration: ScaledAuthorityConfig = ScaledAuthorityConfig(),
) -> ScaledAuthority:
    """Regenerate all phase samples for selected structural headline cases."""

    nominal = _nominal_authority()
    times = np.asarray(nominal["time_s"], dtype=float)
    profile_index = np.asarray(nominal["profile_index"], dtype=np.int64)
    grip_span = np.asarray(nominal["grip_span_m"], dtype=float)
    solution_q = np.asarray(nominal["solution_q"], dtype=float).copy()
    feasible = np.asarray(nominal["feasible"], dtype=bool).copy()
    selected = np.asarray(configuration.case_indices, dtype=np.int64)
    shape = (selected.size, times.size)
    failure_class = np.empty(shape, dtype="U32")
    closure_error = np.empty(shape)
    joint_margin = np.empty(shape)
    collision_clearance = np.empty(shape)
    profiles = default_synthetic_profiles()
    contact_config = ClosedContactConfig(
        joint_limit_scale=configuration.joint_limit_scale
    )

    for selected_slot, case_index in enumerate(selected):
        base = profiles[int(profile_index[case_index])]
        profile = replace(
            base,
            profile_id=(
                f"{base.profile_id}-h{configuration.height_scale:.3f}"
                f"-m{configuration.body_mass_scale:.3f}"
            ),
            height_m=base.height_m * configuration.height_scale,
            mass_kg=base.mass_kg * configuration.body_mass_scale,
        )
        model, metadata = build_subject_scaled_model(profile)
        hand_contact = float(metadata["hand_contact_local_x_m"])
        previous: FloatArray | None = None
        for time_slot, sample_time_s in enumerate(times):
            reference, _, _ = prescribed_state(model, float(sample_time_s))
            result = solve_closed_contact_configuration(
                model,
                q_reference=reference,
                grip_span_m=float(grip_span[case_index]),
                hand_contact_local_x_m=hand_contact,
                q_seed=previous,
                config=contact_config,
            )
            previous = result.q if result.contact_closed else None
            solution_q[case_index, time_slot] = result.q
            feasible[case_index, time_slot] = result.feasible
            failure_class[selected_slot, time_slot] = _failure_class(result)
            closure_error[selected_slot, time_slot] = float(
                np.max(result.hand_to_grip_distance_m)
            )
            joint_margin[selected_slot, time_slot] = (
                result.minimum_joint_limit_margin_rad
            )
            collision_clearance[selected_slot, time_slot] = (
                result.minimum_collision_clearance_m
            )

    nominal_corner = (
        configuration.height_scale == 1.0
        and configuration.body_mass_scale == 1.0
        and configuration.joint_limit_scale == 1.0
    )
    nominal_error = (
        float(
            np.max(
                np.abs(
                    solution_q[selected]
                    - np.asarray(nominal["solution_q"], dtype=float)[selected]
                )
            )
        )
        if nominal_corner
        else float("nan")
    )
    sources = _source_hashes()
    digest = _digest_payload(
        configuration,
        times,
        profile_index,
        grip_span,
        solution_q,
        feasible,
        selected,
        failure_class,
        sources,
    )
    return ScaledAuthority(
        configuration=configuration,
        time_s=times,
        profile_index=profile_index,
        grip_span_m=grip_span,
        solution_q=solution_q,
        feasible=feasible,
        selected_case_indices=selected,
        selected_failure_class=failure_class,
        selected_maximum_closure_error_m=closure_error,
        selected_minimum_joint_limit_margin_rad=joint_margin,
        selected_minimum_collision_clearance_m=collision_clearance,
        maximum_nominal_state_error_rad=nominal_error,
        source_sha256=sources,
        authority_sha256=digest,
    )


def validate_scaled_authority(
    authority: ScaledAuthority,
    expected: ScaledAuthorityConfig,
) -> None:
    """Fail closed on configuration, source, shape, or content drift."""

    if authority.configuration != expected:
        raise RuntimeError("scaled authority configuration does not match")
    if authority.source_sha256 != _source_hashes():
        raise RuntimeError("scaled authority source digest does not match")
    observed = _digest_payload(
        authority.configuration,
        authority.time_s,
        authority.profile_index,
        authority.grip_span_m,
        authority.solution_q,
        authority.feasible,
        authority.selected_case_indices,
        authority.selected_failure_class,
        authority.source_sha256,
    )
    if observed != authority.authority_sha256:
        raise RuntimeError("scaled authority content digest does not match")
    if authority.solution_q.shape != (18, 13, 20) or authority.feasible.shape != (
        18,
        13,
    ):
        raise RuntimeError("scaled authority has incompatible array shapes")


__all__ = [
    "ScaledAuthority",
    "ScaledAuthorityConfig",
    "build_scaled_authority",
    "validate_scaled_authority",
]
