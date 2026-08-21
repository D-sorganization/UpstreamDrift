"""Compare native MuJoCo equality dynamics with projected compliant contact.

The two branches start from one identical closed articulated state.  The native
branch embeds two MuJoCo ``connect`` equalities and advances them with
``mj_step``.  The projected branch retains the repository's bilateral
Kelvin--Voigt generalized-force law and semi-implicit step.  Their disagreement
is therefore an observed formulation discrepancy, not an engine-equivalence
residual and not evidence about human anatomy or technique.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_contact_projection import (
    ArticulatedContactProjectionConfig,
    evaluate_contact_projection,
)
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    ForwardIntegrationCase,
    integrate_articulated_contact,
)
from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    ArticulatedForwardContactConfig,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
    _mujoco_xml,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"


@dataclass(frozen=True, slots=True)
class NativeConstraintDiscrepancyConfig:
    """Predeclared bounded comparison between distinct contact formulations."""

    duration_s: float = 0.004
    time_steps_s: tuple[float, ...] = (0.0005, 0.00025)
    case_index: int = 0
    sample_index: int = 6
    initial_club_displacement_m: float = 0.001
    contact_stiffness: float = 1800.0
    contact_damping: float = 20.0
    minimum_active_force: float = 1.0e-8
    killswitch_force_tolerance: float = 1.0e-12
    initial_state_tolerance: float = 1.0e-15
    minimum_nonzero_discrepancy: float = 1.0e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if not self.time_steps_s or any(
            not np.isfinite(step) or step <= 0.0 for step in self.time_steps_s
        ):
            raise ValueError("time_steps_s must contain finite positive values")
        if any(
            finer >= coarser
            for coarser, finer in zip(
                self.time_steps_s, self.time_steps_s[1:], strict=False
            )
        ):
            raise ValueError("time_steps_s must be ordered from coarse to fine")
        if any(
            not np.isclose(self.duration_s / step, round(self.duration_s / step))
            for step in self.time_steps_s
        ):
            raise ValueError(
                "duration_s must be an integer multiple of every time step"
            )
        if self.case_index < 0 or self.sample_index < 0:
            raise ValueError("case_index and sample_index must be nonnegative")
        for name in (
            "initial_club_displacement_m",
            "contact_stiffness",
            "contact_damping",
            "minimum_active_force",
            "killswitch_force_tolerance",
            "initial_state_tolerance",
            "minimum_nonzero_discrepancy",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class _Trace:
    time_s: FloatArray
    q: FloatArray
    qd: FloatArray
    generalized_force: FloatArray
    attachment_separation_m: FloatArray
    constraint_rows: NDArray[np.int64]


def _fmt(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _native_constraint_xml(
    model: SpatialModel,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    time_step_s: float,
    config: NativeConstraintDiscrepancyConfig,
) -> str:
    """Return the canonical tree with two site-based connect equalities."""

    if grip_span_m <= 0.0 or hand_contact_local_x_m <= 0.0:
        raise ValueError("grip and hand contact geometry must be positive")
    xml = _mujoco_xml(model)
    xml = xml.replace(
        '<option gravity="0 0 -9.80665" timestep="0.001" integrator="RK4"/>',
        (
            '<option gravity="0 0 -9.80665" '
            f'timestep="{time_step_s:.12g}" integrator="Euler" '
            'solver="Newton" iterations="100" tolerance="1e-12"/>'
        ),
    )
    site_specs = (
        (
            "joint_8_lead_wrist",
            '<site name="lead_hand_site" '
            f'pos="{_fmt((hand_contact_local_x_m, 0.0, 0.0))}" size="0.002"/>',
        ),
        (
            "joint_13_trail_wrist",
            '<site name="trail_hand_site" '
            f'pos="{_fmt((hand_contact_local_x_m, 0.0, 0.0))}" size="0.002"/>',
        ),
        (
            "joint_19_club_yaw",
            '<site name="lead_grip_site" '
            f'pos="{_fmt((0.0, 0.5 * grip_span_m, 0.0))}" size="0.002"/>'
            '<site name="trail_grip_site" '
            f'pos="{_fmt((0.0, -0.5 * grip_span_m, 0.0))}" size="0.002"/>',
        ),
    )
    for body_name, site_xml in site_specs:
        opening = f'<body name="{body_name}"'
        start = xml.find(opening)
        if start < 0:
            raise RuntimeError(f"canonical MuJoCo body is missing: {body_name}")
        end = xml.find(">", start)
        xml = xml[: end + 1] + site_xml + xml[end + 1 :]
    equality = (
        "  <equality>"
        '<connect name="lead_connect" site1="lead_hand_site" '
        'site2="lead_grip_site" '
        f'solref="{-config.contact_stiffness:.12g} '
        f'{-config.contact_damping:.12g}"/>'
        '<connect name="trail_connect" site1="trail_hand_site" '
        'site2="trail_grip_site" '
        f'solref="{-config.contact_stiffness:.12g} '
        f'{-config.contact_damping:.12g}"/>'
        "  </equality>"
    )
    return xml.replace("</mujoco>", equality + "</mujoco>")


def _site_separation(mujoco: Any, model: Any, data: Any) -> float:
    pairs = (
        ("lead_hand_site", "lead_grip_site"),
        ("trail_hand_site", "trail_grip_site"),
    )
    maximum = 0.0
    for first, second in pairs:
        first_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, first)
        second_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, second)
        if first_id < 0 or second_id < 0:
            raise RuntimeError("native constraint site lookup failed")
        maximum = max(
            maximum,
            float(np.linalg.norm(data.site_xpos[first_id] - data.site_xpos[second_id])),
        )
    return maximum


def _calibrate_grip_sites(
    mujoco: Any,
    native_model: Any,
    data: Any,
    calibration_q: FloatArray,
) -> None:
    """Place club sites under the hand sites at the common unperturbed state."""

    data.qpos[:] = calibration_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(native_model, data)
    for hand_name, grip_name in (
        ("lead_hand_site", "lead_grip_site"),
        ("trail_hand_site", "trail_grip_site"),
    ):
        hand_id = mujoco.mj_name2id(native_model, mujoco.mjtObj.mjOBJ_SITE, hand_name)
        grip_id = mujoco.mj_name2id(native_model, mujoco.mjtObj.mjOBJ_SITE, grip_name)
        if hand_id < 0 or grip_id < 0:
            raise RuntimeError("native site calibration lookup failed")
        body_id = int(native_model.site_bodyid[grip_id])
        body_rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        body_position = np.asarray(data.xpos[body_id], dtype=float)
        native_model.site_pos[grip_id] = body_rotation.T @ (
            np.asarray(data.site_xpos[hand_id], dtype=float) - body_position
        )
    mujoco.mj_forward(native_model, data)
    if _site_separation(mujoco, native_model, data) > 1.0e-10:
        raise RuntimeError("native sites do not close at the calibration state")


def _native_trace(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    time_step_s: float,
    config: NativeConstraintDiscrepancyConfig,
    equality_active: bool,
) -> _Trace:
    import mujoco

    native_model = mujoco.MjModel.from_xml_string(
        _native_constraint_xml(
            model,
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=hand_contact_local_x_m,
            time_step_s=time_step_s,
            config=config,
        )
    )
    if native_model.nq != model.nq or native_model.neq != 2:
        raise RuntimeError("native constraint model dimensions do not match")
    data = mujoco.MjData(native_model)
    calibration_q = np.asarray(q, dtype=float).copy()
    calibration_q[14] -= config.initial_club_displacement_m
    _calibrate_grip_sites(mujoco, native_model, data, calibration_q)
    data.qpos[:] = q
    data.qvel[:] = qd
    data.eq_active[:] = equality_active
    step_count = int(round(config.duration_s / time_step_s))
    position = np.empty((step_count + 1, model.nq), dtype=float)
    velocity = np.empty_like(position)
    force = np.empty_like(position)
    separation = np.empty(step_count + 1, dtype=float)
    rows = np.empty(step_count + 1, dtype=np.int64)
    for index in range(step_count + 1):
        mujoco.mj_forward(native_model, data)
        position[index] = data.qpos
        velocity[index] = data.qvel
        force[index] = data.qfrc_constraint
        separation[index] = _site_separation(mujoco, native_model, data)
        rows[index] = data.nefc
        if index < step_count:
            mujoco.mj_step(native_model, data)
    return _Trace(
        time_s=np.arange(step_count + 1, dtype=float) * time_step_s,
        q=position,
        qd=velocity,
        generalized_force=force,
        attachment_separation_m=separation,
        constraint_rows=rows,
    )


def _projected_trace(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    time_step_s: float,
    config: NativeConstraintDiscrepancyConfig,
) -> _Trace:
    case = ForwardIntegrationCase(
        q=q,
        qd=qd,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_local_x_m,
        time_step_s=time_step_s,
        contact_stiffness=config.contact_stiffness,
        contact_damping=config.contact_damping,
        initial_club_displacement_m=0.0,
        initial_club_velocity_m_s=0.0,
        engine="mujoco",
    )
    result = integrate_articulated_contact(
        model,
        case,
        ArticulatedForwardContactConfig(
            duration_s=config.duration_s,
            time_steps_s=(time_step_s, 0.5 * time_step_s),
            case_indices=(config.case_index,),
            sample_indices=(config.sample_index,),
            contact_stiffness=config.contact_stiffness,
            contact_damping=config.contact_damping,
            initial_club_displacement_m=config.initial_club_displacement_m,
        ),
    )
    position = np.asarray(result["q"], dtype=float)
    velocity = np.asarray(result["qd"], dtype=float)
    force = np.empty_like(position)
    for index, (sample_q, sample_qd) in enumerate(zip(position, velocity, strict=True)):
        force[index] = evaluate_contact_projection(
            model,
            sample_q,
            sample_qd,
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=hand_contact_local_x_m,
            perturb_contact=False,
            config=ArticulatedContactProjectionConfig(
                contact_stiffness=config.contact_stiffness,
                contact_damping=config.contact_damping,
            ),
        ).generalized_contact_force
    return _Trace(
        time_s=np.asarray(result["time_s"], dtype=float),
        q=position,
        qd=velocity,
        generalized_force=force,
        attachment_separation_m=np.asarray(
            result["maximum_attachment_separation_m"], dtype=float
        ),
        constraint_rows=np.zeros(position.shape[0], dtype=np.int64),
    )


def _load_initial_state(
    config: NativeConstraintDiscrepancyConfig,
) -> tuple[SpatialModel, FloatArray, float, float]:
    profiles = default_synthetic_profiles()
    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as source:
        profile_index = int(source["case_profile_index"][config.case_index])
        solution_q = np.asarray(
            source["solution_q"][config.case_index, config.sample_index], dtype=float
        )
        grip_span_m = float(source["case_grip_span_m"][config.case_index])
    model, metadata = build_subject_scaled_model(profiles[profile_index])
    solution_q = solution_q.copy()
    solution_q[14] += config.initial_club_displacement_m
    return (
        model,
        solution_q,
        grip_span_m,
        float(metadata["hand_contact_local_x_m"]),
    )


def _execute_resolutions(
    model: SpatialModel,
    initial_q: FloatArray,
    initial_qd: FloatArray,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    config: NativeConstraintDiscrepancyConfig,
) -> tuple[list[tuple[_Trace, _Trace, _Trace]], FloatArray, FloatArray]:
    resolutions: list[tuple[_Trace, _Trace, _Trace]] = []
    final_discrepancy = np.empty(len(config.time_steps_s), dtype=float)
    maximum_discrepancy = np.empty_like(final_discrepancy)
    for index, time_step_s in enumerate(config.time_steps_s):
        common = {
            "grip_span_m": grip_span_m,
            "hand_contact_local_x_m": hand_contact_local_x_m,
            "time_step_s": time_step_s,
            "config": config,
        }
        native = _native_trace(
            model, initial_q, initial_qd, equality_active=True, **common
        )
        projected = _projected_trace(model, initial_q, initial_qd, **common)
        killswitch = _native_trace(
            model, initial_q, initial_qd, equality_active=False, **common
        )
        discrepancy = np.max(np.abs(native.q - projected.q), axis=1)
        final_discrepancy[index] = discrepancy[-1]
        maximum_discrepancy[index] = np.max(discrepancy)
        resolutions.append((native, projected, killswitch))
    return resolutions, final_discrepancy, maximum_discrepancy


def _assemble_arrays(
    config: NativeConstraintDiscrepancyConfig,
    resolution: tuple[_Trace, _Trace, _Trace],
    final_discrepancy: FloatArray,
    maximum_discrepancy: FloatArray,
) -> dict[str, NDArray[Any]]:
    native, projected, killswitch = resolution
    return {
        "time_step_s": np.asarray(config.time_steps_s, dtype=float),
        "final_trajectory_absolute_discrepancy": final_discrepancy,
        "maximum_trajectory_absolute_discrepancy": maximum_discrepancy,
        "time_s": native.time_s,
        "native_q": native.q,
        "projected_q": projected.q,
        "native_qd": native.qd,
        "projected_qd": projected.qd,
        "native_generalized_constraint_force_n": native.generalized_force,
        "projected_generalized_contact_force_n": projected.generalized_force,
        "killswitch_generalized_constraint_force_n": killswitch.generalized_force,
        "native_attachment_separation_m": native.attachment_separation_m,
        "projected_attachment_separation_m": projected.attachment_separation_m,
        "native_constraint_rows": native.constraint_rows,
    }


def _evaluate_gates(
    config: NativeConstraintDiscrepancyConfig,
    arrays: dict[str, NDArray[Any]],
    resolution: tuple[_Trace, _Trace, _Trace],
    maximum_discrepancy: FloatArray,
) -> tuple[dict[str, bool], dict[str, float]]:
    native, projected, killswitch = resolution
    diagnostics = {
        "native_force": float(np.max(np.linalg.norm(native.generalized_force, axis=1))),
        "projected_force": float(
            np.max(np.linalg.norm(projected.generalized_force, axis=1))
        ),
        "killswitch_force": float(
            np.max(np.linalg.norm(killswitch.generalized_force, axis=1))
        ),
        "initial_discrepancy": float(np.max(np.abs(native.q[0] - projected.q[0]))),
        "maximum_discrepancy": float(np.max(maximum_discrepancy)),
    }
    finite = all(
        values.dtype.kind not in "fc" or np.all(np.isfinite(values))
        for values in arrays.values()
    )
    gates = {
        "finite": finite,
        "native_constraint_force_nonzero": diagnostics["native_force"]
        > config.minimum_active_force,
        "projected_contact_force_nonzero": diagnostics["projected_force"]
        > config.minimum_active_force,
        "killswitch_force_zero": diagnostics["killswitch_force"]
        <= config.killswitch_force_tolerance,
        "identical_initial_state": diagnostics["initial_discrepancy"]
        <= config.initial_state_tolerance,
        "formulations_distinct": diagnostics["maximum_discrepancy"]
        > config.minimum_nonzero_discrepancy,
        "six_native_constraint_rows": int(np.min(native.constraint_rows)) >= 6,
    }
    return gates, diagnostics


def _build_record(
    config: NativeConstraintDiscrepancyConfig,
    model: SpatialModel,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    resolution: tuple[_Trace, _Trace, _Trace],
    final_discrepancy: FloatArray,
    gates: dict[str, bool],
    diagnostics: dict[str, float],
) -> dict[str, Any]:
    native, projected, _ = resolution
    return {
        "schema_version": "articulated-native-constraint-discrepancy/v1",
        "study_id": "bilateral-native-connect-versus-projected-compliant-contact",
        "configuration": asdict(config),
        "model": {
            "coordinate_count": model.nq,
            "canonical_hash": model.canonical_hash,
            "case_index": config.case_index,
            "sample_index": config.sample_index,
            "grip_span_m": grip_span_m,
            "hand_contact_local_x_m": hand_contact_local_x_m,
        },
        "native_branch": {
            "engine": "mujoco",
            "constraint_type": "connect",
            "constraint_count": 2,
            "minimum_constraint_row_count": int(np.min(native.constraint_rows)),
            "integrator": "Euler",
            "integrator_operator": "mj_step",
            "generalized_force_source": "qfrc_constraint",
        },
        "projected_branch": {
            "contact_law": "bilateral Kelvin--Voigt point-force projection",
            "dynamics_operator": "MuJoCo mass and bias transport",
            "integrator": "project-authored semi-implicit Euler",
        },
        "results": {
            "maximum_native_generalized_constraint_force": diagnostics["native_force"],
            "maximum_projected_generalized_contact_force": diagnostics[
                "projected_force"
            ],
            "maximum_killswitch_generalized_constraint_force": diagnostics[
                "killswitch_force"
            ],
            "initial_state_absolute_discrepancy": diagnostics["initial_discrepancy"],
            "maximum_trajectory_absolute_discrepancy": diagnostics[
                "maximum_discrepancy"
            ],
            "finest_final_trajectory_absolute_discrepancy": float(
                final_discrepancy[-1]
            ),
            "native_initial_attachment_separation_m": float(
                native.attachment_separation_m[0]
            ),
            "native_final_attachment_separation_m": float(
                native.attachment_separation_m[-1]
            ),
            "projected_final_attachment_separation_m": float(
                projected.attachment_separation_m[-1]
            ),
            "registered_gates": gates,
            "all_registered_gates_passed": bool(all(gates.values())),
        },
        "claim_boundary": {
            "supported": (
                "MuJoCo independently solves two native connect equalities and "
                "advances its own integrator from the same state used by the "
                "projected compliant-contact branch"
            ),
            "engine_equivalence": "not_claimed",
            "constraint_formulation_equivalence": "not_claimed",
            "calibrated_grip_contact": "not_established",
            "anatomical_contact": "not_established",
            "human_transfer_or_strategy": "untested",
        },
    }


def run_native_constraint_discrepancy(
    config: NativeConstraintDiscrepancyConfig = NativeConstraintDiscrepancyConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Execute native and projected branches and retain their discrepancy.

    Postcondition: every returned numeric array is finite, both active branches
    produce nonzero generalized force, and the equality-disabled branch remains
    force-free.  No small-disagreement or anatomical-validity claim is implied.
    """

    if not isinstance(config, NativeConstraintDiscrepancyConfig):
        raise TypeError("config must be NativeConstraintDiscrepancyConfig")
    model, initial_q, grip_span_m, hand_contact_local_x_m = _load_initial_state(config)
    initial_qd = np.zeros(model.nq, dtype=float)
    resolutions, final_discrepancy, maximum_discrepancy = _execute_resolutions(
        model,
        initial_q,
        initial_qd,
        grip_span_m,
        hand_contact_local_x_m,
        config,
    )
    arrays = _assemble_arrays(
        config, resolutions[-1], final_discrepancy, maximum_discrepancy
    )
    gates, diagnostics = _evaluate_gates(
        config, arrays, resolutions[-1], maximum_discrepancy
    )
    record = _build_record(
        config,
        model,
        grip_span_m,
        hand_contact_local_x_m,
        resolutions[-1],
        final_discrepancy,
        gates,
        diagnostics,
    )
    if not gates["finite"]:
        raise RuntimeError("native constraint discrepancy produced nonfinite output")
    return record, arrays


__all__ = [
    "NativeConstraintDiscrepancyConfig",
    "run_native_constraint_discrepancy",
]
