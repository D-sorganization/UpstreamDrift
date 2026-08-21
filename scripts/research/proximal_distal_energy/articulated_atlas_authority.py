"""Corner-consistent authority/model boundary for articulated headline atlases."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    ScaledAuthority,
    validate_scaled_authority,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    SyntheticSubjectProfile,
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def _scientific_float(value: float) -> str:
    """Canonicalize a finite model scalar above last-bit runtime noise."""

    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError("scientific model parameters must be finite")
    return format(scalar, ".15g")


def scientific_model_sha256(model: SpatialModel) -> str:
    """Hash model semantics reproducibly across floating-point runtime state.

    Fifteen significant decimal digits preserve the registered engineering
    parameters while excluding last-bit summation changes observed after native
    libraries alter the Windows floating-point environment. The native engine
    cache may retain ``SpatialModel.canonical_hash``; this digest is the portable
    scientific identity used by structural evidence and checkpoints.
    """

    if not isinstance(model, SpatialModel):
        raise TypeError("model must be a SpatialModel")
    payload = {
        "schema_version": "articulated-scientific-model-identity/v1",
        "joints": [
            {
                "name": joint.name,
                "parent": joint.parent,
                "kind": joint.kind,
                "axis": [_scientific_float(value) for value in joint.axis],
                "offset_m": [_scientific_float(value) for value in joint.offset_m],
                "region": joint.region,
            }
            for joint in model.joints
        ],
        "bodies": [
            {
                "name": body.name,
                "joint": body.joint,
                "mass_kg": _scientific_float(body.mass_kg),
                "radius_m": _scientific_float(body.radius_m),
                "com_offset_m": [
                    _scientific_float(value) for value in body.com_offset_m
                ],
                "region": body.region,
            }
            for body in model.bodies
        ],
        "interfaces": {
            "club_dof_indices": [int(value) for value in model.club_dof_indices],
            "lead_hand_joint": model.lead_hand_joint,
            "trail_hand_joint": model.trail_hand_joint,
            "club_frame_joint": model.club_frame_joint,
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class ArticulatedAtlasAuthority:
    """States and the exact anthropometric model transform that produced them."""

    time_s: FloatArray
    profile_index: IntArray
    grip_span_m: FloatArray
    solution_q: FloatArray
    feasible: BoolArray
    failure_class: NDArray[np.str_]
    selected_case_indices: IntArray
    height_scale: float
    body_mass_scale: float
    joint_limit_scale: float
    authority_sha256: str

    def __post_init__(self) -> None:
        if self.time_s.shape != (13,) or not np.all(np.diff(self.time_s) > 0.0):
            raise ValueError("time_s must contain 13 increasing phase samples")
        if self.profile_index.shape != (18,) or np.any(
            (self.profile_index < 0) | (self.profile_index >= 6)
        ):
            raise ValueError("profile_index must map 18 cases into six profiles")
        if self.grip_span_m.shape != (18,) or np.any(self.grip_span_m <= 0.0):
            raise ValueError("grip_span_m must contain 18 positive spans")
        if self.solution_q.shape != (18, 13, 20):
            raise ValueError("solution_q must have shape (18, 13, 20)")
        if self.feasible.shape != (18, 13) or self.failure_class.shape != (18, 13):
            raise ValueError("feasibility fields must have shape (18, 13)")
        selected = self.selected_case_indices
        if (
            selected.ndim != 1
            or selected.size == 0
            or np.unique(selected).size != selected.size
            or np.any((selected < 0) | (selected >= 18))
        ):
            raise ValueError("selected_case_indices must be unique registered cases")
        for name in ("height_scale", "body_mass_scale", "joint_limit_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.5 <= value <= 1.5:
                raise ValueError(f"{name} must be finite and lie in [0.5, 1.5]")
        if len(self.authority_sha256) != 64:
            raise ValueError("authority_sha256 must be a SHA-256 digest")

    @classmethod
    def from_scaled(cls, authority: ScaledAuthority) -> ArticulatedAtlasAuthority:
        """Adapt a governed scaled authority without deleting failed samples."""

        validate_scaled_authority(authority, authority.configuration)
        failure_class = np.full((18, 13), "not_selected", dtype="U32")
        failure_class[authority.selected_case_indices] = (
            authority.selected_failure_class
        )
        return cls(
            time_s=np.asarray(authority.time_s, dtype=float),
            profile_index=np.asarray(authority.profile_index, dtype=np.int64),
            grip_span_m=np.asarray(authority.grip_span_m, dtype=float),
            solution_q=np.asarray(authority.solution_q, dtype=float),
            feasible=np.asarray(authority.feasible, dtype=bool),
            failure_class=failure_class,
            selected_case_indices=np.asarray(
                authority.selected_case_indices, dtype=np.int64
            ),
            height_scale=authority.configuration.height_scale,
            body_mass_scale=authority.configuration.body_mass_scale,
            joint_limit_scale=authority.configuration.joint_limit_scale,
            authority_sha256=authority.authority_sha256,
        )

    def selected_failures(self) -> tuple[dict[str, int | str], ...]:
        """Return every failed selected phase in deterministic case/time order."""

        failures: list[dict[str, int | str]] = []
        for case_index in self.selected_case_indices:
            for phase_index in np.flatnonzero(~self.feasible[int(case_index)]):
                failures.append(
                    {
                        "case_index": int(case_index),
                        "phase_index": int(phase_index),
                        "failure_class": str(
                            self.failure_class[int(case_index), int(phase_index)]
                        ),
                    }
                )
        return tuple(failures)

    def require_selected_feasible(self) -> None:
        """Fail closed before an atlas silently omits an infeasible state."""

        failures = self.selected_failures()
        if failures:
            raise RuntimeError(
                f"selected authority states are infeasible: {len(failures)} failure(s)"
            )

    def require_state_feasible(self, case_index: int, phase_index: int) -> None:
        """Reject one failed phase without invalidating other phases in its case."""

        self._require_selected_case(case_index)
        if not isinstance(phase_index, int) or not 0 <= phase_index < 13:
            raise ValueError("phase_index must lie in [0, 13)")
        if not self.feasible[case_index, phase_index]:
            failure = str(self.failure_class[case_index, phase_index])
            raise RuntimeError(
                "selected authority state is infeasible: "
                f"case={case_index}, phase={phase_index}, failure={failure}"
            )

    def feasible_states(
        self,
        case_indices: tuple[int, ...],
        phase_indices: tuple[int, ...],
    ) -> tuple[tuple[int, int], ...]:
        """Return the feasible subset of a registered Cartesian state design."""

        states: list[tuple[int, int]] = []
        for case_index in case_indices:
            self._require_selected_case(case_index)
            for phase_index in phase_indices:
                if not isinstance(phase_index, int) or not 0 <= phase_index < 13:
                    raise ValueError("phase_index must lie in [0, 13)")
                if self.feasible[case_index, phase_index]:
                    states.append((case_index, phase_index))
        return tuple(states)

    def model_hashes(self) -> dict[str, str]:
        """Return canonical hashes for every selected corner-consistent model."""

        hashes: dict[str, str] = {}
        for case_index in self.selected_case_indices:
            case = int(case_index)
            model, metadata = self.build_case_model(case)
            hashes[str(case)] = self.validate_case_model(case, model, metadata)
        return hashes

    def provenance_record(self) -> dict[str, Any]:
        """Describe the structural authority needed to bind an atlas execution."""

        return {
            "schema_version": "articulated-atlas-authority/v1",
            "authority_sha256": self.authority_sha256,
            "scales": {
                "height": self.height_scale,
                "body_mass": self.body_mass_scale,
                "joint_limit": self.joint_limit_scale,
            },
            "selected_case_indices": [
                int(value) for value in self.selected_case_indices
            ],
            "retained_failures": [dict(value) for value in self.selected_failures()],
            "model_sha256": self.model_hashes(),
        }

    def profile_for_case(self, case_index: int) -> SyntheticSubjectProfile:
        """Return the exact scaled synthetic profile for one selected case."""

        self._require_selected_case(case_index)
        base = default_synthetic_profiles()[int(self.profile_index[case_index])]
        return replace(
            base,
            profile_id=(
                f"{base.profile_id}-h{self.height_scale:.3f}"
                f"-m{self.body_mass_scale:.3f}"
            ),
            height_m=base.height_m * self.height_scale,
            mass_kg=base.mass_kg * self.body_mass_scale,
        )

    def build_case_model(self, case_index: int) -> tuple[SpatialModel, dict[str, Any]]:
        """Build the corner-consistent model independently of phase feasibility."""

        self._require_selected_case(case_index)
        return build_subject_scaled_model(self.profile_for_case(case_index))

    def validate_case_model(
        self,
        case_index: int,
        model: SpatialModel,
        metadata: dict[str, Any],
    ) -> str:
        """Reject a model whose profile transform differs from the authority."""

        expected_profile = asdict(self.profile_for_case(case_index))
        if metadata.get("profile") != expected_profile:
            raise RuntimeError("authority/model scaling does not match")
        if metadata.get("model_sha256") != model.canonical_hash:
            raise RuntimeError("authority/model hash does not match")
        return scientific_model_sha256(model)

    def _require_selected_case(self, case_index: int) -> None:
        if not isinstance(case_index, int) or case_index not in set(
            self.selected_case_indices.tolist()
        ):
            raise ValueError("case_index must be a selected case")


__all__ = ["ArticulatedAtlasAuthority", "scientific_model_sha256"]
