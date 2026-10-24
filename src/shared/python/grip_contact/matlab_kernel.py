"""Canonical Python -> MATLAB codegen kernel for Simscape grip contact (HRG-GS-M12, #13334).

Generates:
1. ``gs3dx_grip_contact_kernel.m`` — a ``%#codegen``-compatible MATLAB function
   with fixed-size arrays, no containers/strings/dynamic memory, and no
   persistent or continuous state.  The shared pairwise contact law in
   :func:`pair_contact.sphere_capsule`, :func:`pair_contact.capsule_capsule`,
   and :func:`pair_contact._sample_contact` is stateless (Hunt-Crossley normal
   force, regularized tanh/Stribeck tangential friction with no tangential
   spring, and torsional spin moment).  Evaluates all physical topology pairs
   of a style and returns per-pair channels and net wrenches on the club and
   each hand in World axes about their body origins.
2. ``gs3dx_grip_contact_params.json`` — law parameters, club grip frame offsets,
   and per-style physical pair tables exported from the Python source of truth.
3. ``gs3dx_grip_pair_golden.json`` — constitutive golden vectors (randomized 3D
   states plus separated, just touching, near-collinear capsules, stick, slip,
   zero relative velocity, coincident axis, and rebounding clamp edge cases)
   cross-checked against :func:`topology_side_wrench`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.grip_contact.bushing_law import BushingState, cross3
from src.shared.python.grip_contact.generated_export_lock import (
    serialize_committed_export,
)
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.generated_freshness import generated_json_matches
from src.shared.python.grip_contact.interface import SIDES, GripInterface
from src.shared.python.grip_contact.matlab_kernel_source import (
    MATLAB_HELPER_FUNCTIONS,
    MATLAB_LOOP_BODY,
)
from src.shared.python.grip_contact.net_wrench_parity import (
    canonicalize_export_tree,
    matlab_export_literal,
    report_generated_drift,
)
from src.shared.python.grip_contact.pair_contact import (
    closest_point_segment,
    line_contact_node_count,
)
from src.shared.python.grip_contact.prescribed_motion import RigidBodyState
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    map_site_to_hand_frame,
    topology_side_wrench,
)
from src.shared.python.grip_contact.topology_series import topology_mapping_hash

REPO_ROOT = Path(__file__).resolve().parents[4]
GS3DX_ROOT = (
    REPO_ROOT
    / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx"
)
DEFAULT_SPEC_PATH = (
    REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
DEFAULT_GENERATED_DIR = GS3DX_ROOT / "tools/generated"
DEFAULT_KERNEL_M_PATH = DEFAULT_GENERATED_DIR / "gs3dx_grip_contact_kernel.m"
DEFAULT_PARAMS_JSON_PATH = DEFAULT_GENERATED_DIR / "gs3dx_grip_contact_params.json"
DEFAULT_GOLDEN_JSON_PATH = GS3DX_ROOT / "tests/fixtures/gs3dx_grip_pair_golden.json"

KERNEL_SCHEMA_VERSION = "simscape-grip-contact-kernel/v2"
GOLDEN_SCHEMA_VERSION = "simscape-grip-pair-golden/v1"
MAX_PHYSICAL_PAIRS = 12
PAIR_TABLE_COLS = 18
STYLE_IDS: dict[str, int] = {"overlap": 1, "interlock": 2, "ten_finger": 3}


def _fmt(val: float) -> str:
    return matlab_export_literal(val)


def _orthogonal_unit(axis: np.ndarray) -> np.ndarray:
    """Deterministic unit vector orthogonal to ``axis`` (matches ``pair_contact``)."""
    nrm = float(np.linalg.norm(axis))
    if nrm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    unit = axis / nrm
    ref = (
        np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(unit[2])) < 0.9
        else np.array([1.0, 0.0, 0.0], dtype=float)
    )
    ortho = cross3(unit, ref)
    norm_v = float(np.linalg.norm(ortho))
    return ortho / norm_v if norm_v > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=float)


def _build_style_table(model: TopologyContactModel, style_name: str) -> dict[str, Any]:
    """Build fixed-size ``(MAX_PHYSICAL_PAIRS, 18)`` numeric table for ``style_name``.

    Column 18 is the node count of the pair's line-contact quadrature: 1 for a
    sphere, :func:`line_contact_node_count` of the hand capsule for a capsule.

    Preconditions:
        Every hand capsule is strictly shorter than its grip capsule, so it is
        the axis :func:`pair_contact.capsule_capsule` integrates (#13508).
    """
    topo = model.topology
    rows: list[list[float]] = []
    pair_names: list[str] = []
    pair_sides: list[str] = []
    pair_primitives: list[str] = []
    for side in SIDES:
        side_code = 1.0 if side == "L" else 2.0
        cap_b = model.grip_capsule(side)
        b0 = [float(x) for x in cap_b.p0_m]
        b1 = [float(x) for x in cap_b.p1_m]
        rb = float(cap_b.radius_m)
        for pair in model.physical_grip_pairs(side):
            mapped = map_site_to_hand_frame(topo, pair.site_a, side)
            ra = float(mapped.radius_m)
            if mapped.primitive == "sphere":
                prim_code, nodes = 1.0, 1.0
                a0 = [float(x) for x in mapped.centre_m]
                a1 = list(a0)
            elif mapped.primitive == "capsule":
                prim_code = 2.0
                a0, a1 = ([float(x) for x in p] for p in mapped.endpoints_m)
                nodes = float(_hand_capsule_node_count(a0, a1, ra, b0, b1, pair.name))
            else:
                raise ValueError(f"Unsupported primitive {mapped.primitive!r}")
            row = [1.0, side_code, prim_code, *a0, *a1, ra, *b0, *b1, rb, nodes]
            rows.append(row)
            pair_names.append(pair.name)
            pair_sides.append(side)
            pair_primitives.append(mapped.primitive)
    active_count = len(rows)
    if active_count > MAX_PHYSICAL_PAIRS:
        raise ValueError(
            f"Style {style_name!r} has {active_count} pairs > MAX_PHYSICAL_PAIRS={MAX_PHYSICAL_PAIRS}"
        )
    while len(rows) < MAX_PHYSICAL_PAIRS:
        rows.append([0.0] * PAIR_TABLE_COLS)
    return {
        "style_id": STYLE_IDS[style_name],
        "active_pairs": active_count,
        "topology_mapping_hash": topology_mapping_hash(model),
        "pair_names": pair_names,
        "pair_sides": pair_sides,
        "pair_primitives": pair_primitives,
        "pair_table": rows,
    }


def _hand_capsule_node_count(
    a0: list[float],
    a1: list[float],
    r_a: float,
    b0: list[float],
    b1: list[float],
    name: str,
) -> int:
    """Line-contact node count of a hand capsule that integrates its pair."""
    len_a = math.dist(a0, a1)
    if not len_a < math.dist(b0, b1):
        raise ValueError(
            f"pair {name!r}: hand capsule ({len_a} m) must be shorter than the grip "
            "capsule; the kernel integrates along the hand capsule"
        )
    return line_contact_node_count(0.5 * len_a, r_a)


DEFAULT_SQUEEZE_PER_HAND_N = 1100.0


def _model_for_style(
    spec: Mapping[str, Any],
    style: GripStyle | str,
    squeeze_per_hand_n: float = DEFAULT_SQUEEZE_PER_HAND_N,
) -> TopologyContactModel:
    """Build :class:`TopologyContactModel` for ``spec`` and ``style``."""
    from src.shared.python.motion_matching.grip_contract import GripSpec

    gs = style if isinstance(style, GripStyle) else GripStyle(style)
    interface = GripInterface.from_spec(spec)
    topo = contact_topology(GripSpec(style=gs))
    return build_topology_model(
        interface, topology=topo, squeeze_per_hand_n=squeeze_per_hand_n
    )


def build_kernel_params_payload(
    spec_path: Path = DEFAULT_SPEC_PATH,
    squeeze_per_hand_n: float = DEFAULT_SQUEEZE_PER_HAND_N,
) -> dict[str, Any]:
    """Build the canonical ``simscape-grip-contact-kernel/v2`` parameters dictionary."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    interface = GripInterface.from_spec(spec)
    base_model = _model_for_style(spec, GripStyle.OVERLAP, squeeze_per_hand_n)
    law = base_model.law

    styles_dict: dict[str, Any] = {}
    for gs in GripStyle:
        mdl = _model_for_style(spec, gs, squeeze_per_hand_n)
        styles_dict[gs.value] = _build_style_table(mdl, gs.value)

    left_f = interface.left
    right_f = interface.right
    payload: dict[str, Any] = {
        "schema_version": KERNEL_SCHEMA_VERSION,
        "max_physical_pairs": MAX_PHYSICAL_PAIRS,
        "pair_table_columns": [
            "active",
            "side_code",
            "prim_code",
            "a0_x",
            "a0_y",
            "a0_z",
            "a1_x",
            "a1_y",
            "a1_z",
            "r_a",
            "b0_x",
            "b0_y",
            "b0_z",
            "b1_x",
            "b1_y",
            "b1_z",
            "r_b",
            "line_nodes",
        ],
        "law": {
            "normal_stiffness_n_m": float(law.stiffness_n_m),
            "damping_s_m": float(law.dissipation_s_m),
            "static_friction": float(law.static_friction),
            "dynamic_friction": float(law.dynamic_friction),
            "viscous_friction_s_m": float(law.viscous_friction),
            "stribeck_velocity_m_s": float(law.transition_velocity_m_s),
            "patch_radius_m": float(base_model.patch_radius_m),
            "spin_transition_rad_s": float(base_model.spin_transition_rad_s),
        },
        "club_grip_frames": {
            "L": {
                "position_m": [float(x) for x in left_f.position_m],
                "rotation": [[float(x) for x in row] for row in left_f.rotation],
            },
            "R": {
                "position_m": [float(x) for x in right_f.position_m],
                "rotation": [[float(x) for x in row] for row in right_f.rotation],
            },
        },
        "styles": styles_dict,
    }
    validate_kernel_params_payload(payload)
    return payload


def validate_kernel_params_payload(payload: Mapping[str, Any]) -> None:
    """Validate ``simscape-grip-contact-kernel/v2`` parameters dictionary (DbC)."""
    if not isinstance(payload, Mapping):
        raise TypeError("kernel params payload must be a Mapping")
    if payload.get("schema_version") != KERNEL_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {KERNEL_SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}"
        )
    if payload.get("max_physical_pairs") != MAX_PHYSICAL_PAIRS:
        raise ValueError(f"max_physical_pairs must be {MAX_PHYSICAL_PAIRS}")
    law = payload.get("law")
    if (
        not isinstance(law, Mapping)
        or float(law.get("normal_stiffness_n_m", 0.0)) <= 0.0
    ):
        raise ValueError("law.normal_stiffness_n_m must be positive")
    styles = payload.get("styles")
    if not isinstance(styles, Mapping) or set(styles) != {
        "overlap",
        "interlock",
        "ten_finger",
    }:
        raise ValueError("styles must contain overlap, interlock, and ten_finger")
    for sname, sdata in styles.items():
        tbl = np.asarray(sdata["pair_table"], dtype=float)
        if tbl.shape != (MAX_PHYSICAL_PAIRS, PAIR_TABLE_COLS):
            raise ValueError(f"style {sname!r} pair_table has wrong shape {tbl.shape}")
        if not np.all(np.isfinite(tbl)):
            raise ValueError(f"style {sname!r} pair_table contains non-finite values")


def _format_matlab_matrix(mat: list[list[float]], indent: str = "        ") -> str:
    """Render a 2D numeric array as a literal MATLAB matrix."""
    lines = ["["]
    for k, row in enumerate(mat):
        sep = ";" if k + 1 < len(mat) else ""
        row_str = ", ".join(_fmt(v) for v in row)
        lines.append(f"{indent}{row_str}{sep}")
    lines.append("    ]")
    return "\n".join(lines)


def render_matlab_kernel_source(params: Mapping[str, Any]) -> str:
    """Render ``gs3dx_grip_contact_kernel.m`` from ``params`` (single source of truth)."""
    validate_kernel_params_payload(params)
    law = params["law"]
    frames = params["club_grip_frames"]
    styles = params["styles"]
    p_l, r_l = frames["L"]["position_m"], frames["L"]["rotation"]
    p_r, r_r = frames["R"]["position_m"], frames["R"]["rotation"]
    tbl_overlap = _format_matlab_matrix(styles["overlap"]["pair_table"])
    tbl_interlock = _format_matlab_matrix(styles["interlock"]["pair_table"])
    tbl_ten_finger = _format_matlab_matrix(styles["ten_finger"]["pair_table"])

    header = f"""function [F_club_w, M_club_w, F_hand_L_w, M_hand_L_w, F_hand_R_w, M_hand_R_w, ...
    pair_normal_n, pair_tangent_n, pair_spin_nm, pair_penetration_m, ...
    pair_slip_speed_m_s, pair_signed_dist_m, pair_force_club_w, ...
    pair_moment_club_w, pair_contact_point_w] = gs3dx_grip_contact_kernel( ...
    p_club, R_club, v_club, omega_club, ...
    p_hand_L, R_hand_L, v_hand_L, omega_hand_L, ...
    p_hand_R, R_hand_R, v_hand_R, omega_hand_R, ...
    style_id) %#codegen
%GS3DX_GRIP_CONTACT_KERNEL  Generated canonical Simscape grip contact kernel (#13334).
%
%   DO NOT EDIT MANUALLY. Generated by scripts/export_simscape_grip_kernel.py
%   from src/shared/python/grip_contact/matlab_kernel.py.
%   Schema: {KERNEL_SCHEMA_VERSION}
%   Stateless constitutive law mirroring sphere_capsule, the capsule_capsule
%   line-contact quadrature, and _sample_contact in
%   src/shared/python/grip_contact/pair_contact.py.
%
%   Inputs:
%     p_club (3x1), R_club (3x3), v_club (3x1), omega_club (3x1)
%     p_hand_L (3x1), R_hand_L (3x3), v_hand_L (3x1), omega_hand_L (3x1)
%     p_hand_R (3x1), R_hand_R (3x3), v_hand_R (3x1), omega_hand_R (3x1)
%     style_id (1x1) — 1: overlap, 2: interlock, 3: ten_finger
%
%   Outputs (all forces/moments in World axes; moments about body origins):
%     F_club_w (3x1), M_club_w (3x1)       — net wrench on club about p_club
%     F_hand_L_w (3x1), M_hand_L_w (3x1)   — net wrench on hand_L about p_hand_L
%     F_hand_R_w (3x1), M_hand_R_w (3x1)   — net wrench on hand_R about p_hand_R
%     pair_normal_n ({MAX_PHYSICAL_PAIRS}x1), pair_tangent_n ({MAX_PHYSICAL_PAIRS}x1), pair_spin_nm ({MAX_PHYSICAL_PAIRS}x1)
%     pair_penetration_m ({MAX_PHYSICAL_PAIRS}x1), pair_slip_speed_m_s ({MAX_PHYSICAL_PAIRS}x1)
%     pair_signed_dist_m ({MAX_PHYSICAL_PAIRS}x1)
%     pair_force_club_w ({MAX_PHYSICAL_PAIRS}x3), pair_moment_club_w ({MAX_PHYSICAL_PAIRS}x3)
%     pair_contact_point_w ({MAX_PHYSICAL_PAIRS}x3)

    % Constitutive law constants from Python source of truth
    k_n = {_fmt(law["normal_stiffness_n_m"])};
    c_n = {_fmt(law["damping_s_m"])};
    mu_s = {_fmt(law["static_friction"])};
    mu_d = {_fmt(law["dynamic_friction"])};
    mu_v = {_fmt(law["viscous_friction_s_m"])};
    v_tr = {_fmt(law["stribeck_velocity_m_s"])};
    r_patch = {_fmt(law["patch_radius_m"])};
    omega_tr = {_fmt(law["spin_transition_rad_s"])};

    % Club-local grip frame placements (L and R)
    p_club_L_local = [{_fmt(p_l[0])}; {_fmt(p_l[1])}; {_fmt(p_l[2])}];
    R_club_L_local = {_format_matlab_matrix(r_l)};
    p_club_R_local = [{_fmt(p_r[0])}; {_fmt(p_r[1])}; {_fmt(p_r[2])}];
    R_club_R_local = {_format_matlab_matrix(r_r)};

    % Select fixed-size (12x18) pair table for style_id
    if style_id == 2
        pair_table = {tbl_interlock};
    elseif style_id == 3
        pair_table = {tbl_ten_finger};
    else
        pair_table = {tbl_overlap};
    end

"""
    loop = MATLAB_LOOP_BODY.replace("@MAXP@", str(MAX_PHYSICAL_PAIRS))
    return header + loop + MATLAB_HELPER_FUNCTIONS


def _eval_plane_contact_py(
    u_normal: np.ndarray,
    penetration: float,
    v_rel: np.ndarray,
    omega_rel: np.ndarray,
    law: Mapping[str, float],
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Mirror of ``local_plane_contact`` in ``gs3dx_grip_contact_kernel.m``."""
    v_n = float(np.dot(u_normal, v_rel))
    v_t = v_rel - u_normal * v_n
    slip_speed = float(np.linalg.norm(v_t))
    if penetration <= 0.0:
        z3 = np.zeros(3, dtype=float)
        return 0.0, 0.0, 0.0, slip_speed, z3, z3.copy()
    rate = -v_n
    k_n = float(law["normal_stiffness_n_m"])
    c_n = float(law["damping_s_m"])
    mu_s = float(law["static_friction"])
    mu_d = float(law["dynamic_friction"])
    mu_v = float(law["viscous_friction_s_m"])
    v_tr = float(law["stribeck_velocity_m_s"])
    r_patch = float(law["patch_radius_m"])
    omega_tr = float(law["spin_transition_rad_s"])

    f_n_mag = max(0.0, k_n * penetration * (1.0 + c_n * rate))
    normal_force = u_normal * f_n_mag
    if slip_speed > 0.0 and f_n_mag > 0.0:
        ratio = slip_speed / v_tr
        mu = mu_d + (mu_s - mu_d) * math.exp(-(ratio * ratio)) + mu_v * slip_speed
        f_t_mag = mu * f_n_mag * math.tanh(ratio)
        friction_force = (-v_t / slip_speed) * f_t_mag
    else:
        f_t_mag = 0.0
        friction_force = np.zeros(3, dtype=float)
    force_vec = normal_force + friction_force
    if f_n_mag > 0.0 and r_patch > 0.0:
        spin_rate = float(np.dot(u_normal, omega_rel))
        spin_signed = mu_s * f_n_mag * r_patch * math.tanh(spin_rate / omega_tr)
        spin_mag = abs(spin_signed)
        spin_vec = -u_normal * spin_signed
    else:
        spin_mag = 0.0
        spin_vec = np.zeros(3, dtype=float)
    return f_n_mag, f_t_mag, spin_mag, slip_speed, force_vec, spin_vec


def _eval_sphere_node_py(
    center_w: np.ndarray,
    r_a: float,
    seg_b: tuple[np.ndarray, np.ndarray, float],
    pose_a: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    pose_b: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    law: Mapping[str, float],
) -> tuple[Any, ...]:
    """Mirror of ``local_sphere_capsule`` in ``gs3dx_grip_contact_kernel.m``."""
    b0_w, b1_w, r_b = seg_b
    p_a, _, v_a0, om_a = pose_a
    p_b, _, v_b0, om_b = pose_b
    p_axis, _ = closest_point_segment(center_w, b0_w, b1_w)
    diff_v = p_axis - center_w
    dist = float(np.linalg.norm(diff_v))
    normal = diff_v / dist if dist > 1e-12 else _orthogonal_unit(b1_w - b0_w)
    sdist = dist - r_a - r_b
    pen = max(0.0, -sdist)
    cpt = center_w + r_a * normal
    v_a = v_a0 + cross3(om_a, cpt - p_a)
    v_b = v_b0 + cross3(om_b, cpt - p_b)
    fn, ft, sp, slp, force_a, spin_a = _eval_plane_contact_py(
        -normal, pen, v_a - v_b, om_a - om_b, law
    )
    moment_a = cross3(cpt - p_a, force_a) + spin_a
    moment_b = cross3(cpt - p_b, -force_a) - spin_a
    return fn, ft, sp, slp, pen, sdist, force_a, moment_a, -force_a, moment_b, cpt


def _eval_pair_row_py(
    row: np.ndarray,
    pose_a: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    pose_b: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    law: Mapping[str, float],
) -> tuple[Any, ...]:
    """Evaluate one ``pair_table`` row in Python, node by node like the kernel.

    Postconditions:
        Forces, moments and the normal, tangential and spin magnitudes are
        ``1 / n`` weighted node sums; penetration is the deepest node's and the
        signed distance the smallest; contact point and slip speed are
        normal-load weighted node means (unweighted when no node is loaded).
    """
    a0, a1, r_a = row[3:6], row[6:9], float(row[9])
    p_b, r_b_m = pose_b[0], pose_b[1]
    seg_b = (p_b + r_b_m @ row[10:13], p_b + r_b_m @ row[13:16], float(row[16]))
    n_nodes = int(row[17])
    w = 1.0 / n_nodes
    fn = ft = sp = pen = load_slip = mean_slip = 0.0
    sdist = math.inf
    fa, ma, fb, mb, cp_load, cp_mean = (np.zeros(3, dtype=float) for _ in range(6))
    for j in range(n_nodes):
        node = a0 + (j / (n_nodes - 1)) * (a1 - a0) if n_nodes > 1 else a0
        n_fn, n_ft, n_sp, n_slp, n_pen, n_sd, n_fa, n_ma, n_fb, n_mb, n_cpt = (
            _eval_sphere_node_py(
                pose_a[0] + pose_a[1] @ node, r_a, seg_b, pose_a, pose_b, law
            )
        )
        fn, ft, sp = fn + w * n_fn, ft + w * n_ft, sp + w * n_sp
        pen, sdist = max(pen, n_pen), min(sdist, n_sd)
        fa, ma = fa + w * n_fa, ma + w * n_ma
        fb, mb = fb + w * n_fb, mb + w * n_mb
        cp_load, cp_mean = cp_load + (w * n_fn) * n_cpt, cp_mean + w * n_cpt
        load_slip += (w * n_fn) * n_slp
        mean_slip += w * n_slp
    if fn > 0.0:
        cpt, slp = cp_load / fn, load_slip / fn
    else:
        cpt, slp = cp_mean, mean_slip
    return fn, ft, sp, pen, slp, sdist, fa, ma, fb, mb, cpt


def evaluate_kernel_reference(
    params: Mapping[str, Any],
    style: str,
    inputs: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Evaluate the exact ``gs3dx_grip_contact_kernel`` outputs in Python."""
    validate_kernel_params_payload(params)
    if style not in params["styles"]:
        raise ValueError(f"Unknown style {style!r}")
    p_club = np.asarray(inputs["p_club_m"], dtype=float)
    r_club = np.asarray(inputs["r_club"], dtype=float)
    v_club = np.asarray(inputs["v_club_m_s"], dtype=float)
    omega_club = np.asarray(inputs["omega_club_rad_s"], dtype=float)
    pose_hl = (
        np.asarray(inputs["p_hand_l_m"], dtype=float),
        np.asarray(inputs["r_hand_l"], dtype=float),
        np.asarray(inputs["v_hand_l_m_s"], dtype=float),
        np.asarray(inputs["omega_hand_l_rad_s"], dtype=float),
    )
    pose_hr = (
        np.asarray(inputs["p_hand_r_m"], dtype=float),
        np.asarray(inputs["r_hand_r"], dtype=float),
        np.asarray(inputs["v_hand_r_m_s"], dtype=float),
        np.asarray(inputs["omega_hand_r_rad_s"], dtype=float),
    )

    frames = params["club_grip_frames"]
    dp_l = r_club @ np.asarray(frames["L"]["position_m"], dtype=float)
    pose_bl = (
        p_club + dp_l,
        r_club @ np.asarray(frames["L"]["rotation"], dtype=float),
        v_club + cross3(omega_club, dp_l),
        omega_club,
    )
    dp_r = r_club @ np.asarray(frames["R"]["position_m"], dtype=float)
    pose_br = (
        p_club + dp_r,
        r_club @ np.asarray(frames["R"]["rotation"], dtype=float),
        v_club + cross3(omega_club, dp_r),
        omega_club,
    )

    law = params["law"]
    table = np.asarray(params["styles"][style]["pair_table"], dtype=float)
    f_hl, m_hl = np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    f_hr, m_hr = np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    f_sl, m_sl = np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    f_sr, m_sr = np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    pair_normal = np.zeros(MAX_PHYSICAL_PAIRS, dtype=float)
    pair_tangent = np.zeros(MAX_PHYSICAL_PAIRS, dtype=float)
    pair_spin = np.zeros(MAX_PHYSICAL_PAIRS, dtype=float)
    pair_pen = np.zeros(MAX_PHYSICAL_PAIRS, dtype=float)
    pair_slip = np.zeros(MAX_PHYSICAL_PAIRS, dtype=float)
    pair_sdist = np.zeros(MAX_PHYSICAL_PAIRS, dtype=float)
    pair_f_club = np.zeros((MAX_PHYSICAL_PAIRS, 3), dtype=float)
    pair_m_club = np.zeros((MAX_PHYSICAL_PAIRS, 3), dtype=float)
    pair_cpt = np.zeros((MAX_PHYSICAL_PAIRS, 3), dtype=float)

    for k in range(MAX_PHYSICAL_PAIRS):
        row = table[k]
        if row[0] <= 0.0:
            continue
        pa_s, pb_s = (pose_hl, pose_bl) if row[1] == 1.0 else (pose_hr, pose_br)
        fn, ft, sp, pen, slp, sdist, fa, ma, fb, mb, cpt = _eval_pair_row_py(
            row, pa_s, pb_s, law
        )
        if row[1] == 1.0:
            f_hl, m_hl, f_sl, m_sl = f_hl + fa, m_hl + ma, f_sl + fb, m_sl + mb
        else:
            f_hr, m_hr, f_sr, m_sr = f_hr + fa, m_hr + ma, f_sr + fb, m_sr + mb
        pair_normal[k], pair_tangent[k], pair_spin[k] = fn, ft, sp
        pair_pen[k], pair_slip[k], pair_sdist[k] = pen, slp, sdist
        pair_f_club[k], pair_m_club[k], pair_cpt[k] = fb, mb, cpt

    f_club = f_sl + f_sr
    m_club = (m_sl + cross3(pose_bl[0] - p_club, f_sl)) + (
        m_sr + cross3(pose_br[0] - p_club, f_sr)
    )
    return {
        "f_club_w_n": f_club,
        "m_club_w_nm": m_club,
        "f_hand_l_w_n": f_hl,
        "m_hand_l_w_nm": m_hl,
        "f_hand_r_w_n": f_hr,
        "m_hand_r_w_nm": m_hr,
        "pair_normal_n": pair_normal,
        "pair_tangent_n": pair_tangent,
        "pair_spin_nm": pair_spin,
        "pair_penetration_m": pair_pen,
        "pair_slip_speed_m_s": pair_slip,
        "pair_signed_dist_m": pair_sdist,
        "pair_force_club_w_n": pair_f_club,
        "pair_moment_club_w_nm": pair_m_club,
        "pair_contact_point_w_m": pair_cpt,
    }


def _verify_against_topology_side_wrench(
    spec: Mapping[str, Any],
    style: str,
    inputs: Mapping[str, Any],
    out: Mapping[str, np.ndarray],
) -> None:
    """Cross-check ``evaluate_kernel_reference`` against ``topology_side_wrench``."""
    model = _model_for_style(spec, style)
    interface = GripInterface.from_spec(spec)
    club = RigidBodyState(
        np.asarray(inputs["r_club"], dtype=float),
        np.asarray(inputs["p_club_m"], dtype=float),
        np.asarray(inputs["v_club_m_s"], dtype=float),
        np.asarray(inputs["omega_club_rad_s"], dtype=float),
    )
    hand_l = BushingState(
        np.asarray(inputs["r_hand_l"], dtype=float),
        np.asarray(inputs["p_hand_l_m"], dtype=float),
        np.asarray(inputs["v_hand_l_m_s"], dtype=float),
        np.asarray(inputs["omega_hand_l_rad_s"], dtype=float),
    )
    hand_r = BushingState(
        np.asarray(inputs["r_hand_r"], dtype=float),
        np.asarray(inputs["p_hand_r_m"], dtype=float),
        np.asarray(inputs["v_hand_r_m_s"], dtype=float),
        np.asarray(inputs["omega_hand_r_rad_s"], dtype=float),
    )
    c_l = club.frame(interface.left.matrix())
    c_r = club.frame(interface.right.matrix())
    w_l = topology_side_wrench(model, "L", hand_l, c_l)
    w_r = topology_side_wrench(model, "R", hand_r, c_r)

    f_club_ref = w_l.force_n + w_r.force_n
    m_club_ref = (
        w_l.moment_nm
        + cross3(c_l.position_m - club.position_m, w_l.force_n)
        + w_r.moment_nm
        + cross3(c_r.position_m - club.position_m, w_r.force_n)
    )
    np.testing.assert_allclose(out["f_club_w_n"], f_club_ref, rtol=1e-11, atol=1e-9)
    np.testing.assert_allclose(out["m_club_w_nm"], m_club_ref, rtol=1e-11, atol=1e-9)
    np.testing.assert_allclose(out["f_hand_l_w_n"], -w_l.force_n, rtol=1e-11, atol=1e-9)
    np.testing.assert_allclose(out["f_hand_r_w_n"], -w_r.force_n, rtol=1e-11, atol=1e-9)


def _rot_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix for ``axis`` and ``angle_rad``."""
    u = axis / float(np.linalg.norm(axis))
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    ux = np.array(
        [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]], dtype=float
    )
    return np.eye(3) * c + (1.0 - c) * np.outer(u, u) + s * ux


def _align_vector_rotation(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Proper rotation matrix mapping unit direction of ``v_from`` onto ``v_to``."""
    u1 = v_from / float(np.linalg.norm(v_from))
    u2 = v_to / float(np.linalg.norm(v_to))
    cr = np.cross(u1, u2)
    nrm = float(np.linalg.norm(cr))
    dt = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    if nrm <= 1e-12:
        if dt > 0.0:
            return np.eye(3)
        ortho = _orthogonal_unit(u1)
        return _rot_from_axis_angle(ortho, math.pi)
    return _rot_from_axis_angle(cr / nrm, math.atan2(nrm, dt))


def _make_case(
    spec: Mapping[str, Any],
    params: Mapping[str, Any],
    reference_params: Mapping[str, Any],
    case_id: str,
    style: str,
    category: str,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate and cross-verify one golden vector case.

    ``expected`` is the kernel evaluated with the exported ``params`` (what the
    generated MATLAB kernel computes). The law cross-check evaluates the same
    kernel algorithm with the unrounded ``reference_params`` against
    :func:`topology_side_wrench`, which is built from the same unrounded spec.
    """
    out = evaluate_kernel_reference(params, style, inputs)
    ref_out = (
        out
        if reference_params is params
        else evaluate_kernel_reference(reference_params, style, inputs)
    )
    _verify_against_topology_side_wrench(spec, style, inputs, ref_out)
    expected = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in out.items()
    }
    return {
        "case_id": case_id,
        "style": style,
        "style_id": STYLE_IDS[style],
        "category": category,
        "inputs": inputs,
        "expected": expected,
    }


def _base_held_inputs(params: Mapping[str, Any], style: str) -> dict[str, Any]:
    """Nominal gripped configuration with club at origin and hands at club grip frames."""
    frames = params["club_grip_frames"]
    p_l = np.asarray(frames["L"]["position_m"], dtype=float)
    r_l = np.asarray(frames["L"]["rotation"], dtype=float)
    p_r = np.asarray(frames["R"]["position_m"], dtype=float)
    r_r = np.asarray(frames["R"]["rotation"], dtype=float)
    return {
        "p_club_m": [0.0, 0.0, 0.0],
        "r_club": np.eye(3).tolist(),
        "v_club_m_s": [0.0, 0.0, 0.0],
        "omega_club_rad_s": [0.0, 0.0, 0.0],
        "p_hand_l_m": p_l.tolist(),
        "r_hand_l": r_l.tolist(),
        "v_hand_l_m_s": [0.0, 0.0, 0.0],
        "omega_hand_l_rad_s": [0.0, 0.0, 0.0],
        "p_hand_r_m": p_r.tolist(),
        "r_hand_r": r_r.tolist(),
        "v_hand_r_m_s": [0.0, 0.0, 0.0],
        "omega_hand_r_rad_s": [0.0, 0.0, 0.0],
    }


def _clone_inputs(base: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    """Deep-copy ``base`` input dict and apply ``overrides``."""
    out = json.loads(json.dumps(base))
    out.update(overrides)
    return out


def _build_style_edge_cases(
    spec: Mapping[str, Any],
    params: Mapping[str, Any],
    reference_params: Mapping[str, Any],
    style: str,
) -> list[dict[str, Any]]:
    """Build deterministic constitutive edge cases for one ``style``."""
    base = _base_held_inputs(params, style)
    tbl = np.asarray(params["styles"][style]["pair_table"], dtype=float)
    r_grip = np.asarray(params["club_grip_frames"]["L"]["rotation"], dtype=float)
    p_grip_l = np.asarray(params["club_grip_frames"]["L"]["position_m"], dtype=float)
    p_grip_r = np.asarray(params["club_grip_frames"]["R"]["position_m"], dtype=float)
    a0_1, b_mid_1 = tbl[0, 3:6], 0.5 * (tbl[0, 10:13] + tbl[0, 13:16])
    ra_1, rb_1 = float(tbl[0, 9]), float(tbl[0, 16])
    far_r = (p_grip_r + np.array([0.3, 0.3, 0.3])).tolist()

    sep_in = _clone_inputs(
        base,
        p_hand_l_m=(p_grip_l + np.array([0.25, 0.25, 0.25])).tolist(),
        p_hand_r_m=(p_grip_r + np.array([-0.25, -0.25, -0.25])).tolist(),
    )
    jt_offset = b_mid_1 + np.array([0.0, ra_1 + rb_1 + 1.0e-9, 0.0]) - a0_1
    jt_in = _clone_inputs(
        base, p_hand_l_m=(p_grip_l + r_grip @ jt_offset).tolist(), p_hand_r_m=far_r
    )
    co_in = _clone_inputs(
        base,
        p_hand_l_m=(p_grip_l + r_grip @ (b_mid_1 - a0_1)).tolist(),
        p_hand_r_m=far_r,
    )
    v_co = [0.65, -0.35, 0.85]
    comov_in = _clone_inputs(
        base, v_club_m_s=v_co, v_hand_l_m_s=v_co, v_hand_r_m_s=v_co
    )
    stick_in = _clone_inputs(
        base, v_club_m_s=[1.0e-5, -0.5e-5, 0.2e-5], omega_club_rad_s=[1.0e-4, 0.0, 0.0]
    )
    slip_in = _clone_inputs(
        base, v_club_m_s=[1.25, -0.75, 0.45], omega_club_rad_s=[14.0, -3.5, 2.5]
    )
    pen_target = b_mid_1 + np.array([0.0, ra_1 + rb_1 - 0.002, 0.0]) - a0_1
    reb_in = _clone_inputs(
        jt_in,
        p_hand_l_m=(p_grip_l + r_grip @ pen_target).tolist(),
        v_hand_l_m_s=(r_grip @ np.array([0.0, 3.0, 0.0])).tolist(),
    )
    col_in = _clone_inputs(base)
    if style == "ten_finger":
        names = params["styles"][style]["pair_names"]
        cap_row = tbl[names.index("ten_finger_little_finger_grip")]
        ca0, ca1 = cap_row[3:6], cap_row[6:9]
        cb_mid = 0.5 * (cap_row[10:13] + cap_row[13:16])
        ra_c, rb_c = float(cap_row[9]), float(cap_row[16])
        r_hr_new = r_grip @ _align_vector_rotation(ca1 - ca0, np.array([1.0, 0.0, 0.0]))
        target_mid = cb_mid + np.array([0.0, ra_c + rb_c - 0.004, 0.0])
        p_hr_new = p_grip_r + r_grip @ target_mid - r_hr_new @ (0.5 * (ca0 + ca1))
        col_in = _clone_inputs(
            base,
            r_hand_r=r_hr_new.tolist(),
            p_hand_r_m=p_hr_new.tolist(),
            v_club_m_s=[0.15, 0.0, 0.0],
        )

    specs = (
        ("separated", "separated", sep_in),
        ("just_touching", "just_touching", jt_in),
        ("coincident_axis", "coincident_axis", co_in),
        ("zero_rel_vel_rest", "zero_relative_velocity", _clone_inputs(base)),
        ("zero_rel_vel_comoving", "zero_relative_velocity", comov_in),
        ("stick", "stick", stick_in),
        ("slip", "slip", slip_in),
        ("rebounding_clamp", "rebounding_clamp", reb_in),
        ("near_collinear_capsules", "near_collinear_capsules", col_in),
    )
    return [
        _make_case(spec, params, reference_params, f"{style}_{cid}", style, cat, inp)
        for cid, cat, inp in specs
    ]


def build_golden_vectors_payload(
    params: Mapping[str, Any] | None = None,
    spec_path: Path = DEFAULT_SPEC_PATH,
    *,
    reference_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate constitutive golden vectors across styles, edge cases, and random states.

    ``params`` is the exported (canonicalized) kernel parameter payload; case
    inputs and expected outputs are computed from it, so the generated kernel
    reproduces its own goldens (#13552). ``reference_params`` is the unrounded
    payload used only for the 1e-11 law cross-check against
    :func:`topology_side_wrench`; it defaults to ``params``.

    When both are omitted, ``reference_params`` is built from ``spec_path`` and
    ``params`` is its :func:`canonicalize_export_tree` export.
    """
    if params is None:
        if reference_params is None:
            reference_params = build_kernel_params_payload(spec_path)
        params = canonicalize_export_tree(reference_params)
    if reference_params is None:
        reference_params = params
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    rng = np.random.default_rng(13334)

    for style in ("overlap", "interlock", "ten_finger"):
        base = _base_held_inputs(params, style)
        r_grip = np.asarray(params["club_grip_frames"]["L"]["rotation"], dtype=float)
        p_grip_l = np.asarray(
            params["club_grip_frames"]["L"]["position_m"], dtype=float
        )
        p_grip_r = np.asarray(
            params["club_grip_frames"]["R"]["position_m"], dtype=float
        )
        cases.extend(_build_style_edge_cases(spec, params, reference_params, style))

        for r_idx in range(8):
            r_c = _rot_from_axis_angle(
                rng.normal(size=3), float(rng.uniform(-0.35, 0.35))
            )
            p_c = rng.uniform(-0.05, 0.05, size=3)
            dp_l_rnd = rng.uniform(-0.004, 0.004, size=3)
            dp_r_rnd = rng.uniform(-0.004, 0.004, size=3)
            dr_l = _rot_from_axis_angle(
                rng.normal(size=3), float(rng.uniform(-0.08, 0.08))
            )
            dr_r = _rot_from_axis_angle(
                rng.normal(size=3), float(rng.uniform(-0.08, 0.08))
            )
            rnd_in = _clone_inputs(
                base,
                p_club_m=p_c.tolist(),
                r_club=r_c.tolist(),
                v_club_m_s=rng.uniform(-0.8, 0.8, size=3).tolist(),
                omega_club_rad_s=rng.uniform(-6.0, 6.0, size=3).tolist(),
                p_hand_l_m=(p_c + r_c @ (p_grip_l + dp_l_rnd)).tolist(),
                r_hand_l=(r_c @ r_grip @ dr_l).tolist(),
                v_hand_l_m_s=rng.uniform(-0.8, 0.8, size=3).tolist(),
                omega_hand_l_rad_s=rng.uniform(-6.0, 6.0, size=3).tolist(),
                p_hand_r_m=(p_c + r_c @ (p_grip_r + dp_r_rnd)).tolist(),
                r_hand_r=(r_c @ r_grip @ dr_r).tolist(),
                v_hand_r_m_s=rng.uniform(-0.8, 0.8, size=3).tolist(),
                omega_hand_r_rad_s=rng.uniform(-6.0, 6.0, size=3).tolist(),
            )
            cases.append(
                _make_case(
                    spec,
                    params,
                    reference_params,
                    f"{style}_random_{r_idx:02d}",
                    style,
                    "randomized",
                    rnd_in,
                )
            )

    payload = {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "kernel_schema_version": KERNEL_SCHEMA_VERSION,
        "relative_tolerance": 1.0e-10,
        "absolute_tolerance_n": 1.0e-9,
        "cases": cases,
    }
    validate_golden_vectors_payload(payload)
    return payload


def validate_golden_vectors_payload(payload: Mapping[str, Any]) -> None:
    """Validate ``simscape-grip-pair-golden/v1`` golden vectors payload (DbC)."""
    if not isinstance(payload, Mapping):
        raise TypeError("golden vectors payload must be a Mapping")
    if payload.get("schema_version") != GOLDEN_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {GOLDEN_SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) < 27:
        raise ValueError("golden vectors payload must contain at least 27 cases")
    for c in cases:
        if c.get("style") not in STYLE_IDS:
            raise ValueError(f"Invalid style in golden case {c.get('case_id')!r}")
        if "inputs" not in c or "expected" not in c:
            raise ValueError(
                f"Golden case {c.get('case_id')!r} missing inputs/expected"
            )
        if "z_state" in c["inputs"] or "dz_dt" in c["expected"]:
            raise ValueError("Golden case must not include z_state or dz_dt")


def render_json_deterministic(payload: Mapping[str, Any]) -> str:
    """Render deterministic JSON text with 2-space indent, sorted keys, trailing newline."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def kernel_bundle_sha256(
    params: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Compute SHA-256 digests of the generated MATLAB kernel and params JSON."""
    if params is None:
        params = canonicalize_export_tree(build_kernel_params_payload())
    m_src = render_matlab_kernel_source(params)
    p_src = render_json_deterministic(params)
    return {
        "kernel_m_sha256": hashlib.sha256(m_src.encode("utf-8")).hexdigest(),
        "params_json_sha256": hashlib.sha256(p_src.encode("utf-8")).hexdigest(),
    }


def export_matlab_kernel_bundle(
    *,
    kernel_m_path: Path = DEFAULT_KERNEL_M_PATH,
    params_json_path: Path = DEFAULT_PARAMS_JSON_PATH,
    golden_json_path: Path = DEFAULT_GOLDEN_JSON_PATH,
    check: bool = False,
) -> int:
    """Write or check ``gs3dx_grip_contact_kernel.m``, params JSON, and golden JSON.

    The golden vectors are computed from the same canonicalized (rounded)
    parameters that are rendered into the kernel and params JSON, so the
    exported kernel reproduces its own goldens to the golden tolerance.

    ``check`` compares the kernel and params byte-for-byte and the golden
    vectors with :func:`generated_json_matches` (cross-environment ulp noise).
    """

    def _run() -> int:
        raw_params = build_kernel_params_payload()
        export_params = canonicalize_export_tree(raw_params)
        golden = build_golden_vectors_payload(
            export_params, reference_params=raw_params
        )
        m_text = render_matlab_kernel_source(export_params)
        p_text = render_json_deterministic(export_params)
        g_text = render_json_deterministic(golden)

        targets = (
            (Path(kernel_m_path), m_text),
            (Path(params_json_path), p_text),
            (Path(golden_json_path), g_text),
        )
        if check:
            for path, text in targets:
                current = path.read_text(encoding="utf-8") if path.exists() else None
                same = current == text
                if not same and current is not None and path is targets[2][0]:
                    same = generated_json_matches(json.loads(current), golden)
                if not same:
                    report_generated_drift(path, current, text)
                    return 1
            return 0

        for path, expected_text in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected_text, encoding="utf-8")
        return 0

    lock_targets = (
        Path(kernel_m_path),
        Path(params_json_path),
        Path(golden_json_path),
    )
    committed = (
        DEFAULT_KERNEL_M_PATH,
        DEFAULT_PARAMS_JSON_PATH,
        DEFAULT_GOLDEN_JSON_PATH,
    )
    return serialize_committed_export(lock_targets, committed, _run)


__all__ = [
    "DEFAULT_GENERATED_DIR",
    "DEFAULT_GOLDEN_JSON_PATH",
    "DEFAULT_KERNEL_M_PATH",
    "DEFAULT_PARAMS_JSON_PATH",
    "GOLDEN_SCHEMA_VERSION",
    "KERNEL_SCHEMA_VERSION",
    "MAX_PHYSICAL_PAIRS",
    "PAIR_TABLE_COLS",
    "STYLE_IDS",
    "build_golden_vectors_payload",
    "build_kernel_params_payload",
    "evaluate_kernel_reference",
    "export_matlab_kernel_bundle",
    "kernel_bundle_sha256",
    "render_json_deterministic",
    "render_matlab_kernel_source",
    "validate_golden_vectors_payload",
    "validate_kernel_params_payload",
]
