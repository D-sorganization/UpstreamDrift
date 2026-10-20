"""Simscape physical grip contact scenario exporter, series loader, and parity adapter (#13334, HRG-GS-M12).

Provides:
1. Pure-NumPy forward kinematics of the full-body weld model's club
   (:func:`spec_weld_club_state`), bit-identical to Pinocchio's
   ``WeldClubKinematics.state`` without requiring the optional C++ runtime for
   input generation or schema checks.
2. Scenario input exporter (:func:`build_scenarios_payload`,
   :func:`export_scenarios_json`) producing the ``hold`` and ``short_swing``
   scenarios for ``overlap``, ``interlock``, and ``ten_finger`` that the
   R2025b Simscape model ``GS3DX_GripContact`` consumes.
3. Versioned series (``simscape-grip-contact-series/v1``) and receipt
   (``simscape-grip-contact-receipt/v1``) Design-by-Contract validators and
   converters to :class:`TopologyKineticsSeries`, :class:`GripKineticsSeries`,
   and :class:`ContactRun`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.grip_contact.bushing_law import cross3
from src.shared.python.grip_contact.club_dynamics import ClubDynamics
from src.shared.python.grip_contact.contact_run import ContactRun, slip_from_frames
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.matlab_kernel import (
    DEFAULT_SQUEEZE_PER_HAND_N,
    STYLE_IDS,
    build_kernel_params_payload,
    evaluate_kernel_reference,
    kernel_bundle_sha256,
    render_json_deterministic,
)
from src.shared.python.grip_contact.net_wrench_parity import (
    payloads_numerically_close,
    report_generated_drift,
)
from src.shared.python.grip_contact.parity import GripKineticsSeries
from src.shared.python.grip_contact.prescribed_motion import (
    CoordinateSpline,
    RigidBodyState,
    hand_frame_states,
)
from src.shared.python.grip_contact.swing_input import (
    CoordinateSwing,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.topology_contact import (
    HAND_HAND_IDENTIFIABILITY_NOTE,
    TopologyContactModel,
    build_topology_model,
)
from src.shared.python.grip_contact.topology_series import (
    PairLabel,
    TopologyKineticsSeries,
    topology_mapping_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
GS3DX_ROOT = (
    REPO_ROOT
    / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx"
)
DEFAULT_SPEC_PATH = (
    REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
DEFAULT_SWING_NPZ = REPO_ROOT / "tests/fixtures/club_face/swing_q_driver.npz"
DEFAULT_POSES_JSON = REPO_ROOT / "tests/fixtures/club_face/address_poses.json"
DEFAULT_SCENARIOS_JSON_PATH = (
    GS3DX_ROOT / "tools/generated/gs3dx_grip_contact_scenarios.json"
)
DEFAULT_RECEIPT_DIR = GS3DX_ROOT / "receipts/grip_contact"

SCENARIOS_SCHEMA_VERSION = "simscape-grip-contact-scenarios/v1"
SERIES_SCHEMA_VERSION = "simscape-grip-contact-series/v1"
RECEIPT_SCHEMA_VERSION = "simscape-grip-contact-receipt/v1"
REQUIRED_MATLAB_RELEASE = "2025b"
SIMSCAPE_CONTACT_ENGINE = "simscape_contact"
SIMSCAPE_MODEL_ID = "GS3DX_GripContact"
SCENARIO_NAMES = ("hold", "short_swing")
STYLE_NAMES = ("overlap", "interlock", "ten_finger")
DEFAULT_SCENARIO_DURATION_S = 0.1
SAMPLE_DT_S = 0.002


def _axis_rotation(axis_code: str, angle_rad: float) -> np.ndarray:
    """Elementary 3x3 rotation matrix for ``Rx``, ``Ry``, or ``Rz``."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    if axis_code == "Rx":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
    if axis_code == "Ry":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    if axis_code == "Rz":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    raise ValueError(f"Unsupported rotational primitive {axis_code!r}")


def _axis_unit(prim_code: str) -> np.ndarray:
    """Unit vector along ``x``, ``y``, or ``z`` for primitive ``Px/Py/Pz/Rx/Ry/Rz``."""
    axis = prim_code[-1]
    if axis == "x":
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if axis == "y":
        return np.array([0.0, 1.0, 0.0], dtype=float)
    if axis == "z":
        return np.array([0.0, 0.0, 1.0], dtype=float)
    raise ValueError(f"Unsupported primitive {prim_code!r}")


def spec_weld_club_state(
    spec: Mapping[str, Any],
    names: list[str] | tuple[str, ...],
    q: np.ndarray,
    qdot: np.ndarray,
) -> RigidBodyState:
    """Pure-NumPy forward kinematics of ``spec``'s weld club frame for ``(q, qdot)``.

    Matches :meth:`WeldClubKinematics.state` to machine precision (``< 1e-14``).
    """
    q_arr = np.asarray(q, dtype=float)
    qd_arr = np.asarray(qdot, dtype=float)
    if q_arr.shape != (len(names),) or qd_arr.shape != (len(names),):
        raise ValueError("q and qdot must match len(names)")
    q_map = {n: float(q_arr[i]) for i, n in enumerate(names)}
    qd_map = {n: float(qd_arr[i]) for i, n in enumerate(names)}

    states: dict[str, RigidBodyState] = {
        "world": RigidBodyState(
            np.eye(3, dtype=float),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
        )
    }
    pending = list(spec["joints"])
    while pending:
        progressed = False
        next_pending = []
        for jspec in pending:
            parent_name = jspec["parent"]
            if parent_name not in states:
                next_pending.append(jspec)
                continue
            p_state = states[parent_name]
            t_pb = np.asarray(jspec["parent_to_base"], dtype=float)
            arm_pb = p_state.rotation @ t_pb[:3, 3]
            rot = p_state.rotation @ t_pb[:3, :3]
            pos = p_state.position_m + arm_pb
            vel = p_state.velocity_m_s + cross3(p_state.omega_rad_s, arm_pb)
            omega = p_state.omega_rad_s.copy()

            for prim in jspec["primitives"]:
                cname = prim["coordinate"]
                ptype = prim["primitive"]
                qi = q_map.get(cname, 0.0)
                qdi = qd_map.get(cname, 0.0)
                u_local = _axis_unit(ptype)
                if ptype.startswith("P"):
                    dp_w = rot @ (qi * u_local)
                    dv_w = rot @ (qdi * u_local) + cross3(omega, dp_w)
                    pos = pos + dp_w
                    vel = vel + dv_w
                else:
                    omega = omega + rot @ (qdi * u_local)
                    rot = rot @ _axis_rotation(ptype, qi)

            t_cf = np.asarray(jspec["child_to_follower"], dtype=float)
            r_fc = t_cf[:3, :3].T
            p_fc = -r_fc @ t_cf[:3, 3]
            arm_fc = rot @ p_fc
            states[jspec["child"]] = RigidBodyState(
                rotation=rot @ r_fc,
                position_m=pos + arm_fc,
                velocity_m_s=vel + cross3(omega, arm_fc),
                omega_rad_s=omega,
            )
            progressed = True
        if not progressed:
            raise ValueError("Disconnected joint tree in spec")
        pending = next_pending

    club_body = str(spec["closure"]["body_b"])
    return states[club_body]


def _principal_inertia_decomposition(
    inertia_com: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(principal_moments, R_club_from_principal)`` in ``SO(3)``."""
    vals, vecs = np.linalg.eigh(np.asarray(inertia_com, dtype=float))
    rot = np.array(vecs, dtype=float)
    if float(np.linalg.det(rot)) < 0.0:
        rot[:, 0] = -rot[:, 0]
    return np.asarray(vals, dtype=float), rot


def _relative_6dof_hermite_trajectory(
    times: np.ndarray,
    pos_seq: np.ndarray,
    rot_seq: np.ndarray,
    vel_seq: np.ndarray,
    omega_seq: np.ndarray,
) -> dict[str, Any]:
    """Compute exact relative ``[dx, dy, dz, rx, ry, rz]`` Hermite spline coefficients.

    Relative coordinates are defined with respect to the initial hand pose
    ``(p0, R0)`` so ``q_rel(0) == 0``. The joint chain is ``Px -> Py -> Pz ->
    Rx -> Ry -> Rz`` (matching Simscape Multibody's ``Bushing Joint``).
    """
    n = times.size
    p0 = pos_seq[0]
    r0 = rot_seq[0]
    q_rel = np.zeros((n, 6), dtype=float)
    qd_rel = np.zeros((n, 6), dtype=float)

    for k in range(n):
        d_local = r0.T @ (pos_seq[k] - p0)
        vd_local = r0.T @ vel_seq[k]
        r_rel = r0.T @ rot_seq[k]
        ry = float(np.arcsin(np.clip(r_rel[0, 2], -1.0, 1.0)))
        rx = float(np.arctan2(-r_rel[1, 2], r_rel[2, 2]))
        rz = float(np.arctan2(-r_rel[0, 1], r_rel[0, 0]))
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        tany = sy / cy
        e_inv = np.array(
            [
                [1.0, sx * tany, -cx * tany],
                [0.0, cx, sx],
                [0.0, -sx / cy, cx / cy],
            ],
            dtype=float,
        )
        w_rel = r0.T @ omega_seq[k]
        q_rel[k, :3] = d_local
        q_rel[k, 3:] = [rx, ry, rz]
        qd_rel[k, :3] = vd_local
        qd_rel[k, 3:] = e_inv @ w_rel

    dt_h = np.diff(times)[:, None]
    b = qd_rel.copy()
    c = np.zeros_like(q_rel)
    d = np.zeros_like(q_rel)
    c[:-1] = (
        3.0 * (q_rel[1:] - q_rel[:-1]) / (dt_h * dt_h)
        - (2.0 * qd_rel[:-1] + qd_rel[1:]) / dt_h
    )
    d[:-1] = 2.0 * (q_rel[:-1] - q_rel[1:]) / (dt_h * dt_h * dt_h) + (
        qd_rel[:-1] + qd_rel[1:]
    ) / (dt_h * dt_h)
    c[-1] = c[-2] + 3.0 * d[-2] * dt_h[-1]
    d[-1] = d[-2]
    return {
        "q_rel": q_rel.tolist(),
        "qd_rel": qd_rel.tolist(),
        "hermite_b": b.tolist(),
        "hermite_c": c.tolist(),
        "hermite_d": d.tolist(),
    }


def _build_one_scenario(
    spec: Mapping[str, Any],
    swing: CoordinateSwing,
    interface: GripInterface,
    scenario: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the ``hold`` or ``short_swing`` scenario dictionary."""
    is_hold = scenario == "hold"
    n_samples = int(round(duration_s / SAMPLE_DT_S)) + 1
    times = np.round(np.arange(n_samples, dtype=float) * SAMPLE_DT_S, 10)
    names = list(swing.names)

    if is_hold:
        q0 = np.asarray(swing.q[0], dtype=float)
        zero_qd = np.zeros_like(q0)
        weld0 = spec_weld_club_state(spec, names, q0, zero_qd)
        weld_seq = [weld0 for _ in range(n_samples)]
    else:
        spline = CoordinateSpline(swing.time_s, swing.q)
        weld_seq = [
            spec_weld_club_state(spec, names, *spline.evaluate(float(t))) for t in times
        ]
        weld0 = weld_seq[0]

    r_c0 = weld0.rotation
    v_local_0 = r_c0.T @ weld0.velocity_m_s
    w_local_0 = r_c0.T @ weld0.omega_rad_s

    hand_trajs: dict[str, Any] = {}
    init_hands: dict[str, Any] = {}
    for side in ("L", "R"):
        h_states = [hand_frame_states(w, interface)[side] for w in weld_seq]
        pos_arr = np.array([h.position_m for h in h_states], dtype=float)
        rot_arr = np.array([h.rotation for h in h_states], dtype=float)
        vel_arr = np.array([h.velocity_m_s for h in h_states], dtype=float)
        om_arr = np.array([h.omega_rad_s for h in h_states], dtype=float)
        init_hands[side] = {
            "position_m": pos_arr[0].tolist(),
            "rotation_world_from_hand": rot_arr[0].tolist(),
            "velocity_world_m_s": vel_arr[0].tolist(),
            "omega_world_rad_s": om_arr[0].tolist(),
        }
        rel_hermite = _relative_6dof_hermite_trajectory(
            times, pos_arr, rot_arr, vel_arr, om_arr
        )
        hand_trajs[side] = {
            "position_m": pos_arr.tolist(),
            "rotation_world_from_hand": rot_arr.tolist(),
            "velocity_world_m_s": vel_arr.tolist(),
            "omega_world_rad_s": om_arr.tolist(),
            **rel_hermite,
        }

    return {
        "scenario": scenario,
        "hold": is_hold,
        "duration_s": float(duration_s),
        "sample_dt_s": SAMPLE_DT_S,
        "time_s": times.tolist(),
        "initial_club_state": {
            "position_m": weld0.position_m.tolist(),
            "rotation_world_from_club": r_c0.tolist(),
            "velocity_world_m_s": weld0.velocity_m_s.tolist(),
            "omega_world_rad_s": weld0.omega_rad_s.tolist(),
            "velocity_local_m_s": v_local_0.tolist(),
            "omega_local_rad_s": w_local_0.tolist(),
        },
        "initial_hand_states": init_hands,
        "hand_trajectories": hand_trajs,
    }


def build_scenarios_payload(
    *,
    spec_path: Path = DEFAULT_SPEC_PATH,
    swing_npz_path: Path = DEFAULT_SWING_NPZ,
    poses_json_path: Path = DEFAULT_POSES_JSON,
    squeeze_per_hand_n: float = DEFAULT_SQUEEZE_PER_HAND_N,
    duration_s: float = DEFAULT_SCENARIO_DURATION_S,
) -> dict[str, Any]:
    """Build the canonical ``simscape-grip-contact-scenarios/v1`` JSON dictionary."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    names = list(spec["coordinate_order"])
    swing = load_coordinate_swing(swing_npz_path, poses_json_path, "driver", names)
    interface = GripInterface.from_spec(spec)
    club = ClubDynamics.from_spec(spec)
    p_moments, r_principal = _principal_inertia_decomposition(club.inertia_com_kg_m2)

    styles_meta: dict[str, Any] = {}
    from src.shared.python.motion_matching.grip_contract import GripSpec

    for style_name in STYLE_NAMES:
        topo = contact_topology(GripSpec(style=GripStyle(style_name)))
        t_model = build_topology_model(
            interface, topology=topo, squeeze_per_hand_n=squeeze_per_hand_n
        )
        phys_names = [
            p.name for s in ("L", "R") for p in t_model.physical_grip_pairs(s)
        ]
        alloc_names = [p.name for p in topo.hand_hand_pairs()]
        labels = {
            p.name: ("physical" if p.is_hand_grip else "allocated") for p in topo.pairs
        }
        styles_meta[style_name] = {
            "style": style_name,
            "style_id": STYLE_IDS[style_name],
            "topology_mapping_hash": topology_mapping_hash(t_model),
            "physical_pair_names": phys_names,
            "allocated_pair_names": alloc_names,
            "all_pair_names": [p.name for p in topo.pairs],
            "pair_labels": labels,
        }

    scenarios = {
        sc: _build_one_scenario(spec, swing, interface, sc, duration_s)
        for sc in SCENARIO_NAMES
    }
    payload: dict[str, Any] = {
        "schema_version": SCENARIOS_SCHEMA_VERSION,
        "matlab_release": REQUIRED_MATLAB_RELEASE,
        "squeeze_per_hand_n": float(squeeze_per_hand_n),
        "gravity_m_s2": [float(x) for x in spec["gravity_m_s2"]],
        "swing_sha256": swing.sha256,
        "kernel_hashes": kernel_bundle_sha256(),
        "solver": {
            "solver_name": "ode15s",
            "rel_tol": 1.0e-6,
            "abs_tol": 1.0e-8,
            "max_step_s": 2.0e-4,
            "refined_rel_tol": 1.0e-7,
            "refined_abs_tol": 1.0e-9,
            "refined_max_step_s": 1.0e-4,
        },
        "club": {
            "mass_kg": float(club.mass_kg),
            "com_m": [float(x) for x in club.com_m],
            "inertia_com_kg_m2": [
                [float(x) for x in row] for row in club.inertia_com_kg_m2
            ],
            "principal_moments_kg_m2": [float(x) for x in p_moments],
            "principal_axes_club": [[float(x) for x in row] for row in r_principal],
            "grip_radius_m": float(
                contact_topology(GripSpec(style=GripStyle.OVERLAP)).grip_radius_m
            ),
            "T_club_grip_L": [
                [float(x) for x in row] for row in interface.left.matrix()
            ],
            "T_club_grip_R": [
                [float(x) for x in row] for row in interface.right.matrix()
            ],
        },
        "styles": styles_meta,
        "scenarios": scenarios,
    }
    validate_scenarios_payload(payload)
    return payload


def validate_scenarios_payload(payload: Mapping[str, Any]) -> None:
    """Validate ``simscape-grip-contact-scenarios/v1`` payload (DbC)."""
    if not isinstance(payload, Mapping):
        raise TypeError("scenarios payload must be a Mapping")
    if payload.get("schema_version") != SCENARIOS_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCENARIOS_SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}"
        )
    if payload.get("matlab_release") != REQUIRED_MATLAB_RELEASE:
        raise ValueError(f"matlab_release must be {REQUIRED_MATLAB_RELEASE!r}")
    styles = payload.get("styles")
    if not isinstance(styles, Mapping) or set(styles) != set(STYLE_NAMES):
        raise ValueError(f"styles must have keys {STYLE_NAMES}")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, Mapping) or set(scenarios) != set(SCENARIO_NAMES):
        raise ValueError(f"scenarios must have keys {SCENARIO_NAMES}")
    club = payload.get("club")
    if not isinstance(club, Mapping) or float(club.get("mass_kg", 0.0)) <= 0.0:
        raise ValueError("club.mass_kg must be positive")


def export_scenarios_json(
    path: Path = DEFAULT_SCENARIOS_JSON_PATH,
    *,
    check: bool = False,
) -> int:
    """Write or check ``gs3dx_grip_contact_scenarios.json``."""
    payload = build_scenarios_payload()
    text = render_json_deterministic(payload)
    target = Path(path)
    if check:
        current = target.read_text(encoding="utf-8") if target.exists() else None
        if current == text:
            return 0
        if target.resolve() == DEFAULT_SCENARIOS_JSON_PATH.resolve():
            try:
                act = json.loads(current)
                exp = json.loads(text)
                if payloads_numerically_close(act, exp):
                    return 0
            except Exception:
                pass
        report_generated_drift(target, current, text)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return 0


@dataclass(frozen=True)
class SimscapeGripContactSeries:
    """Validated Simscape grip contact simulation time series (`simscape-grip-contact-series/v1`)."""

    schema_version: str
    matlab_release: str
    engine: str
    model_id: str
    style: str
    scenario: str
    mapping_hash: str
    time_s: np.ndarray
    club_position_w_m: np.ndarray
    club_rotation_w: np.ndarray
    club_velocity_w_m_s: np.ndarray
    club_omega_w_rad_s: np.ndarray
    hand_l_position_w_m: np.ndarray
    hand_l_rotation_w: np.ndarray
    hand_r_position_w_m: np.ndarray
    hand_r_rotation_w: np.ndarray
    net_club_force_w_n: np.ndarray
    net_club_moment_origin_w_nm: np.ndarray
    force_on_club_l_w_n: np.ndarray
    moment_on_club_l_w_nm: np.ndarray
    force_on_club_r_w_n: np.ndarray
    moment_on_club_r_w_nm: np.ndarray
    pair_normal_force_n: Mapping[str, np.ndarray]
    pair_tangential_force_n: Mapping[str, np.ndarray]
    pair_spin_moment_nm: Mapping[str, np.ndarray]
    pair_slip_speed_m_s: Mapping[str, np.ndarray]
    pair_labels: Mapping[str, PairLabel]
    unavailable_reasons: Mapping[str, str]


def _check_nan_reason(
    arr: np.ndarray,
    channel_key: str,
    reasons: Mapping[str, str],
) -> None:
    """Require a non-empty reason in ``reasons`` whenever ``arr`` contains NaN."""
    if bool(np.isinf(arr).any()):
        raise ValueError(f"Channel {channel_key!r} contains infinite values")
    if bool(np.isnan(arr).any()):
        reason = reasons.get(channel_key, "")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                f"Channel {channel_key!r} contains NaN but has no non-empty entry in unavailable_reasons"
            )


def validate_simscape_grip_contact_series(payload: Mapping[str, Any]) -> None:
    """Validate a ``simscape-grip-contact-series/v1`` JSON dictionary (DbC)."""
    if not isinstance(payload, Mapping):
        raise TypeError("series payload must be a Mapping")
    if payload.get("schema_version") != SERIES_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SERIES_SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}"
        )
    if payload.get("matlab_release") != REQUIRED_MATLAB_RELEASE:
        raise ValueError(
            f"matlab_release must be {REQUIRED_MATLAB_RELEASE!r}, got {payload.get('matlab_release')!r}"
        )
    if payload.get("style") not in STYLE_NAMES:
        raise ValueError(f"style must be one of {STYLE_NAMES}")
    if payload.get("scenario") not in SCENARIO_NAMES:
        raise ValueError(f"scenario must be one of {SCENARIO_NAMES}")
    m_hash = payload.get("mapping_hash")
    if not isinstance(m_hash, str) or not m_hash.startswith("sha256:"):
        raise ValueError("mapping_hash must be a non-empty sha256:... digest")

    t = np.asarray(payload.get("time_s", []), dtype=float)
    if (
        t.ndim != 1
        or t.size < 2
        or not np.all(np.isfinite(t))
        or not np.all(np.diff(t) > 0)
    ):
        raise ValueError("time_s must be 1-D strictly increasing with >= 2 samples")
    n = t.size

    reasons = payload.get("unavailable_reasons") or {}
    if not isinstance(reasons, Mapping):
        raise TypeError("unavailable_reasons must be a Mapping")

    vec3_keys = (
        "club_position_w_m",
        "club_velocity_w_m_s",
        "club_omega_w_rad_s",
        "hand_l_position_w_m",
        "hand_r_position_w_m",
        "net_club_force_w_n",
        "net_club_moment_origin_w_nm",
        "force_on_club_l_w_n",
        "moment_on_club_l_w_nm",
        "force_on_club_r_w_n",
        "moment_on_club_r_w_nm",
    )
    for k in vec3_keys:
        arr = np.asarray(payload.get(k, []), dtype=float)
        if arr.shape != (n, 3):
            raise ValueError(f"{k} must have shape ({n}, 3), got {arr.shape}")
        _check_nan_reason(arr, k, reasons)

    for rk in ("club_rotation_w", "hand_l_rotation_w", "hand_r_rotation_w"):
        rarr = np.asarray(payload.get(rk, []), dtype=float)
        if rarr.shape != (n, 3, 3):
            raise ValueError(f"{rk} must have shape ({n}, 3, 3), got {rarr.shape}")
        _check_nan_reason(rarr, rk, reasons)

    labels = payload.get("pair_labels")
    if not isinstance(labels, Mapping) or not labels:
        raise ValueError("pair_labels must be a non-empty Mapping")

    for map_key in (
        "pair_normal_force_n",
        "pair_tangential_force_n",
        "pair_spin_moment_nm",
        "pair_slip_speed_m_s",
    ):
        ch_map = payload.get(map_key)
        if not isinstance(ch_map, Mapping) or set(ch_map) != set(labels):
            raise ValueError(f"{map_key} keys must match pair_labels")
        for pname, vals in ch_map.items():
            parr = np.asarray(vals, dtype=float)
            if parr.shape != (n,):
                raise ValueError(f"{map_key}[{pname!r}] must have shape ({n},)")
            if bool(np.isnan(parr).any()):
                has_reason = bool(
                    str(reasons.get(pname, "")).strip()
                    or str(reasons.get(f"{map_key}:{pname}", "")).strip()
                )
                if not has_reason:
                    raise ValueError(
                        f"{map_key}[{pname!r}] contains NaN without entry in unavailable_reasons"
                    )


def parse_simscape_grip_contact_series(
    payload: Mapping[str, Any],
) -> SimscapeGripContactSeries:
    """Validate and parse a ``simscape-grip-contact-series/v1`` dictionary."""
    validate_simscape_grip_contact_series(payload)
    labels: dict[str, PairLabel] = {
        k: ("physical" if v == "physical" else "allocated")
        for k, v in payload["pair_labels"].items()
    }
    return SimscapeGripContactSeries(
        schema_version=str(payload["schema_version"]),
        matlab_release=str(payload["matlab_release"]),
        engine=str(payload.get("engine", SIMSCAPE_CONTACT_ENGINE)),
        model_id=str(payload.get("model_id", SIMSCAPE_MODEL_ID)),
        style=str(payload["style"]),
        scenario=str(payload["scenario"]),
        mapping_hash=str(payload["mapping_hash"]),
        time_s=np.asarray(payload["time_s"], dtype=float),
        club_position_w_m=np.asarray(payload["club_position_w_m"], dtype=float),
        club_rotation_w=np.asarray(payload["club_rotation_w"], dtype=float),
        club_velocity_w_m_s=np.asarray(payload["club_velocity_w_m_s"], dtype=float),
        club_omega_w_rad_s=np.asarray(payload["club_omega_w_rad_s"], dtype=float),
        hand_l_position_w_m=np.asarray(payload["hand_l_position_w_m"], dtype=float),
        hand_l_rotation_w=np.asarray(payload["hand_l_rotation_w"], dtype=float),
        hand_r_position_w_m=np.asarray(payload["hand_r_position_w_m"], dtype=float),
        hand_r_rotation_w=np.asarray(payload["hand_r_rotation_w"], dtype=float),
        net_club_force_w_n=np.asarray(payload["net_club_force_w_n"], dtype=float),
        net_club_moment_origin_w_nm=np.asarray(
            payload["net_club_moment_origin_w_nm"], dtype=float
        ),
        force_on_club_l_w_n=np.asarray(payload["force_on_club_l_w_n"], dtype=float),
        moment_on_club_l_w_nm=np.asarray(payload["moment_on_club_l_w_nm"], dtype=float),
        force_on_club_r_w_n=np.asarray(payload["force_on_club_r_w_n"], dtype=float),
        moment_on_club_r_w_nm=np.asarray(payload["moment_on_club_r_w_nm"], dtype=float),
        pair_normal_force_n={
            k: np.asarray(v, dtype=float)
            for k, v in payload["pair_normal_force_n"].items()
        },
        pair_tangential_force_n={
            k: np.asarray(v, dtype=float)
            for k, v in payload["pair_tangential_force_n"].items()
        },
        pair_spin_moment_nm={
            k: np.asarray(v, dtype=float)
            for k, v in payload["pair_spin_moment_nm"].items()
        },
        pair_slip_speed_m_s={
            k: np.asarray(v, dtype=float)
            for k, v in payload["pair_slip_speed_m_s"].items()
        },
        pair_labels=labels,
        unavailable_reasons=dict(payload.get("unavailable_reasons") or {}),
    )


def load_simscape_grip_contact_series(path: Path) -> SimscapeGripContactSeries:
    """Load and validate a ``simscape-grip-contact-series/v1`` JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_simscape_grip_contact_series(data)


def simscape_series_to_topology_series(
    series: SimscapeGripContactSeries,
) -> TopologyKineticsSeries:
    """Convert :class:`SimscapeGripContactSeries` into :class:`TopologyKineticsSeries`."""
    reasons: dict[str, str] = {}
    for pname in series.pair_labels:
        if (
            bool(np.isnan(series.pair_normal_force_n[pname]).any())
            or bool(np.isnan(series.pair_tangential_force_n[pname]).any())
            or bool(np.isnan(series.pair_spin_moment_nm[pname]).any())
        ):
            reasons[pname] = series.unavailable_reasons.get(
                pname, HAND_HAND_IDENTIFIABILITY_NOTE
            )
    return TopologyKineticsSeries(
        engine=series.engine,
        model_id=series.model_id,
        mapping_hash=series.mapping_hash,
        time_s=series.time_s,
        pair_normal_force_n=series.pair_normal_force_n,
        pair_tangential_force_n=series.pair_tangential_force_n,
        pair_spin_moment_nm=series.pair_spin_moment_nm,
        force_on_club_n={
            "L": series.force_on_club_l_w_n,
            "R": series.force_on_club_r_w_n,
        },
        moment_on_club_nm={
            "L": series.moment_on_club_l_w_nm,
            "R": series.moment_on_club_r_w_nm,
        },
        labels=series.pair_labels,
        channel_reasons=reasons,
        applied_allocated_force_n=0.0,
    )


def _hand_and_club_poses(
    series: SimscapeGripContactSeries,
    interface: GripInterface,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    n = series.time_s.size
    off_l = interface.left.matrix()
    off_r = interface.right.matrix()
    club_pos_l = np.zeros((n, 3), dtype=float)
    club_rot_l = np.zeros((n, 3, 3), dtype=float)
    club_pos_r = np.zeros((n, 3), dtype=float)
    club_rot_r = np.zeros((n, 3, 3), dtype=float)
    for k in range(n):
        rc = series.club_rotation_w[k]
        pc = series.club_position_w_m[k]
        club_pos_l[k] = pc + rc @ off_l[:3, 3]
        club_rot_l[k] = rc @ off_l[:3, :3]
        club_pos_r[k] = pc + rc @ off_r[:3, 3]
        club_rot_r[k] = rc @ off_r[:3, :3]
    return (
        {
            "L": (series.hand_l_rotation_w, series.hand_l_position_w_m),
            "R": (series.hand_r_rotation_w, series.hand_r_position_w_m),
        },
        {
            "L": (club_rot_l, club_pos_l),
            "R": (club_rot_r, club_pos_r),
        },
    )


def simscape_series_to_kinetics_series(
    series: SimscapeGripContactSeries,
    interface: GripInterface | None = None,
) -> GripKineticsSeries:
    """Convert :class:`SimscapeGripContactSeries` into :class:`GripKineticsSeries`."""
    if interface is None:
        spec = json.loads(DEFAULT_SPEC_PATH.read_text(encoding="utf-8"))
        interface = GripInterface.from_spec(spec)

    hand_pose, club_pose = _hand_and_club_poses(series, interface)
    wrench = {
        "L": (series.force_on_club_l_w_n, series.moment_on_club_l_w_nm),
        "R": (series.force_on_club_r_w_n, series.moment_on_club_r_w_nm),
    }
    return GripKineticsSeries.from_frames(
        series.engine,
        series.time_s,
        wrench,
        hand_pose,
        club_pose,
        series.club_rotation_w,
        metadata={
            "model_id": series.model_id,
            "matlab_release": series.matlab_release,
            "style": series.style,
            "scenario": series.scenario,
            "mapping_hash": series.mapping_hash,
        },
    )


def simscape_series_to_contact_run(
    series: SimscapeGripContactSeries,
    interface: GripInterface | None = None,
) -> ContactRun:
    """Wrap :class:`SimscapeGripContactSeries` in a :class:`ContactRun` for parity reporting."""
    if interface is None:
        spec = json.loads(DEFAULT_SPEC_PATH.read_text(encoding="utf-8"))
        interface = GripInterface.from_spec(spec)
    kin = simscape_series_to_kinetics_series(series, interface=interface)
    hand_pose, club_pose = _hand_and_club_poses(series, interface)
    axial, roll = slip_from_frames(hand_pose, club_pose)
    from src.shared.python.motion_matching.grip_contract import GripSpec

    topo = contact_topology(GripSpec(style=GripStyle(series.style)))
    normal_side: dict[str, np.ndarray] = {}
    for side, side_val in (("L", "lead"), ("R", "trail")):
        pnames = [
            p.name
            for p in topo.hand_grip_pairs()
            if p.site_a.startswith(f"{side_val}_")
            and p.name in series.pair_normal_force_n
        ]
        normal_side[side] = np.sum(
            [series.pair_normal_force_n[p] for p in pnames], axis=0
        )
    reasons = {
        p: series.unavailable_reasons.get(p, HAND_HAND_IDENTIFIABILITY_NOTE)
        for p, lbl in series.pair_labels.items()
        if lbl == "allocated"
    }
    return ContactRun(
        series=kin,
        normal_force_n=normal_side,
        roll_slip_rad=roll,
        axial_slip_m=axial,
        pair_normal_force_n=dict(series.pair_normal_force_n),
        pair_tangential_force_n=dict(series.pair_tangential_force_n),
        channel_reasons=reasons,
    )


def synthesize_reference_simscape_series(
    style: str = "overlap",
    scenario: str = "hold",
    n_samples: int = 6,
) -> dict[str, Any]:
    """Synthesize a schema-valid series dictionary from the Python kernel for loader unit tests."""
    params = build_kernel_params_payload()
    scenarios = build_scenarios_payload(duration_s=(n_samples - 1) * SAMPLE_DT_S)
    sc_data = scenarios["scenarios"][scenario]
    st_meta = scenarios["styles"][style]
    p_names, a_names = st_meta["physical_pair_names"], st_meta["allocated_pair_names"]
    all_names, labels = st_meta["all_pair_names"], st_meta["pair_labels"]
    c_init = sc_data["initial_club_state"]
    hl_traj, hr_traj = (
        sc_data["hand_trajectories"]["L"],
        sc_data["hand_trajectories"]["R"],
    )
    times = sc_data["time_s"]

    f_club_seq, m_club_seq, f_l_seq, m_l_seq, f_r_seq, m_r_seq = [], [], [], [], [], []
    p_norm: dict[str, list[float]] = {p: [] for p in all_names}
    p_tang: dict[str, list[float]] = {p: [] for p in all_names}
    p_spin: dict[str, list[float]] = {p: [] for p in all_names}
    p_slip: dict[str, list[float]] = {p: [] for p in all_names}
    tbl = np.asarray(params["styles"][style]["pair_table"], dtype=float)
    mask_l, mask_r = tbl[:, 1] == 1.0, tbl[:, 1] == 2.0

    for k in range(len(times)):
        inp = {
            "p_club_m": c_init["position_m"],
            "r_club": c_init["rotation_world_from_club"],
            "v_club_m_s": c_init["velocity_world_m_s"],
            "omega_club_rad_s": c_init["omega_world_rad_s"],
            "p_hand_l_m": hl_traj["position_m"][k],
            "r_hand_l": hl_traj["rotation_world_from_hand"][k],
            "v_hand_l_m_s": hl_traj["velocity_world_m_s"][k],
            "omega_hand_l_rad_s": hl_traj["omega_world_rad_s"][k],
            "p_hand_r_m": hr_traj["position_m"][k],
            "r_hand_r": hr_traj["rotation_world_from_hand"][k],
            "v_hand_r_m_s": hr_traj["velocity_world_m_s"][k],
            "omega_hand_r_rad_s": hr_traj["omega_world_rad_s"][k],
        }
        out = evaluate_kernel_reference(params, style, inp)
        f_club_seq.append(out["f_club_w_n"].tolist())
        m_club_seq.append(out["m_club_w_nm"].tolist())
        pf, pm = out["pair_force_club_w_n"], out["pair_moment_club_w_nm"]
        f_l_seq.append(np.sum(pf[mask_l], axis=0).tolist())
        m_l_seq.append(np.sum(pm[mask_l], axis=0).tolist())
        f_r_seq.append(np.sum(pf[mask_r], axis=0).tolist())
        m_r_seq.append(np.sum(pm[mask_r], axis=0).tolist())
        for idx, pname in enumerate(p_names):
            p_norm[pname].append(float(out["pair_normal_n"][idx]))
            p_tang[pname].append(float(out["pair_tangent_n"][idx]))
            p_spin[pname].append(float(out["pair_spin_nm"][idx]))
            p_slip[pname].append(float(out["pair_slip_speed_m_s"][idx]))
        for pname in a_names:
            for ch in (p_norm, p_tang, p_spin, p_slip):
                ch[pname].append(float("nan"))

    return {
        "schema_version": SERIES_SCHEMA_VERSION,
        "matlab_release": REQUIRED_MATLAB_RELEASE,
        "engine": SIMSCAPE_CONTACT_ENGINE,
        "model_id": SIMSCAPE_MODEL_ID,
        "style": style,
        "scenario": scenario,
        "mapping_hash": st_meta["topology_mapping_hash"],
        "time_s": times,
        "club_position_w_m": [c_init["position_m"] for _ in times],
        "club_rotation_w": [c_init["rotation_world_from_club"] for _ in times],
        "club_velocity_w_m_s": [c_init["velocity_world_m_s"] for _ in times],
        "club_omega_w_rad_s": [c_init["omega_world_rad_s"] for _ in times],
        "hand_l_position_w_m": hl_traj["position_m"],
        "hand_l_rotation_w": hl_traj["rotation_world_from_hand"],
        "hand_r_position_w_m": hr_traj["position_m"],
        "hand_r_rotation_w": hr_traj["rotation_world_from_hand"],
        "net_club_force_w_n": f_club_seq,
        "net_club_moment_origin_w_nm": m_club_seq,
        "force_on_club_l_w_n": f_l_seq,
        "moment_on_club_l_w_nm": m_l_seq,
        "force_on_club_r_w_n": f_r_seq,
        "moment_on_club_r_w_nm": m_r_seq,
        "pair_normal_force_n": p_norm,
        "pair_tangential_force_n": p_tang,
        "pair_spin_moment_nm": p_spin,
        "pair_slip_speed_m_s": p_slip,
        "pair_labels": labels,
        "unavailable_reasons": dict.fromkeys(a_names, HAND_HAND_IDENTIFIABILITY_NOTE),
    }


def validate_simscape_grip_contact_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate a ``simscape-grip-contact-receipt/v1`` dictionary (DbC)."""
    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a Mapping")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {RECEIPT_SCHEMA_VERSION!r}, got {receipt.get('schema_version')!r}"
        )
    if receipt.get("matlab_release") != REQUIRED_MATLAB_RELEASE:
        raise ValueError(
            f"matlab_release must be {REQUIRED_MATLAB_RELEASE!r}, got {receipt.get('matlab_release')!r}"
        )
    for hkey in ("model_sha256", "input_sha256", "kernel_sha256"):
        hval = receipt.get(hkey)
        if not isinstance(hval, str) or len(hval) != 64:
            raise ValueError(f"{hkey} must be a 64-char SHA-256 hex string")

    budget = receipt.get("block_budget")
    if not isinstance(budget, Mapping):
        raise ValueError("block_budget must be a Mapping")
    ceiling = int(budget.get("license_ceiling", 1000))
    compiled = int(budget.get("compiled_total", 1001))
    if compiled > ceiling:
        raise ValueError(
            f"compiled_total ({compiled}) exceeds license_ceiling ({ceiling})"
        )

    for check_key in (
        "timestep_convergence",
        "save_reload_check",
        "frame_probes",
        "force_sign_checks",
    ):
        chk = receipt.get(check_key)
        if not isinstance(chk, Mapping) or not bool(chk.get("passed", False)):
            raise ValueError(f"Receipt qualification check {check_key!r} did not pass")


def load_simscape_grip_contact_receipt(path: Path) -> dict[str, Any]:
    """Load and validate a ``simscape-grip-contact-receipt/v1`` JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_simscape_grip_contact_receipt(data)
    return dict(data)


__all__ = [
    "DEFAULT_RECEIPT_DIR",
    "DEFAULT_SCENARIOS_JSON_PATH",
    "DEFAULT_SCENARIO_DURATION_S",
    "RECEIPT_SCHEMA_VERSION",
    "REQUIRED_MATLAB_RELEASE",
    "SCENARIOS_SCHEMA_VERSION",
    "SCENARIO_NAMES",
    "SERIES_SCHEMA_VERSION",
    "SIMSCAPE_CONTACT_ENGINE",
    "SIMSCAPE_MODEL_ID",
    "STYLE_NAMES",
    "SimscapeGripContactSeries",
    "build_scenarios_payload",
    "export_scenarios_json",
    "load_simscape_grip_contact_receipt",
    "load_simscape_grip_contact_series",
    "parse_simscape_grip_contact_series",
    "simscape_series_to_contact_run",
    "simscape_series_to_kinetics_series",
    "simscape_series_to_topology_series",
    "spec_weld_club_state",
    "synthesize_reference_simscape_series",
    "validate_scenarios_payload",
    "validate_simscape_grip_contact_receipt",
    "validate_simscape_grip_contact_series",
]
