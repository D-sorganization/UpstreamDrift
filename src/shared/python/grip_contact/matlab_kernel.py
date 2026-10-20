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
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.interface import SIDES, GripInterface
from src.shared.python.grip_contact.pair_contact import (
    closest_point_segment,
    closest_points_segments,
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

KERNEL_SCHEMA_VERSION = "simscape-grip-contact-kernel/v1"
GOLDEN_SCHEMA_VERSION = "simscape-grip-pair-golden/v1"
MAX_PHYSICAL_PAIRS = 12
PAIR_TABLE_COLS = 17
STYLE_IDS: dict[str, int] = {"overlap": 1, "interlock": 2, "ten_finger": 3}


def _fmt(val: float) -> str:
    """Format a float with full IEEE-754 round-trip precision for MATLAB/JSON."""
    clean = 0.0 if abs(float(val)) < 1e-18 else float(val)
    return f"{clean:.17g}"


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
    """Build fixed-size ``(MAX_PHYSICAL_PAIRS, 17)`` numeric table for ``style_name``."""
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
            if mapped.primitive == "sphere":
                prim_code = 1.0
                a0 = [float(x) for x in mapped.centre_m]
                a1 = [float(x) for x in mapped.centre_m]
            elif mapped.primitive == "capsule":
                prim_code = 2.0
                p0, p1 = mapped.endpoints_m
                a0 = [float(x) for x in p0]
                a1 = [float(x) for x in p1]
            else:
                raise ValueError(f"Unsupported primitive {mapped.primitive!r}")
            ra = float(mapped.radius_m)
            row = [1.0, side_code, prim_code, *a0, *a1, ra, *b0, *b1, rb]
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
    """Build the canonical ``simscape-grip-contact-kernel/v1`` parameters dictionary."""
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
    """Validate ``simscape-grip-contact-kernel/v1`` parameters dictionary (DbC)."""
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


_MATLAB_HELPER_FUNCTIONS = """
function c = local_cross3(a, b)
    c = [a(2)*b(3) - a(3)*b(2); ...
         a(3)*b(1) - a(1)*b(3); ...
         a(1)*b(2) - a(2)*b(1)];
end

function u = local_orthogonal_unit(axis)
    nrm = sqrt(axis(1)*axis(1) + axis(2)*axis(2) + axis(3)*axis(3));
    if nrm < 1.0e-12
        u = [1.0; 0.0; 0.0];
        return;
    end
    unit = axis / nrm;
    if abs(unit(3)) < 0.9
        ref = [0.0; 0.0; 1.0];
    else
        ref = [1.0; 0.0; 0.0];
    end
    ortho = local_cross3(unit, ref);
    onrm = sqrt(ortho(1)*ortho(1) + ortho(2)*ortho(2) + ortho(3)*ortho(3));
    if onrm > 1.0e-12
        u = ortho / onrm;
    else
        u = [1.0; 0.0; 0.0];
    end
end

function q = local_closest_point_segment(p, a, b)
    ab = b - a;
    denom = ab(1)*ab(1) + ab(2)*ab(2) + ab(3)*ab(3);
    if denom < 1.0e-14
        q = a;
        return;
    end
    ap = p - a;
    t = (ap(1)*ab(1) + ap(2)*ab(2) + ap(3)*ab(3)) / denom;
    t = min(max(t, 0.0), 1.0);
    q = a + t * ab;
end

function [pa, pb] = local_closest_points_segments(p1, q1, p2, q2)
    d1 = q1 - p1;
    d2 = q2 - p2;
    r = p1 - p2;
    a = d1(1)*d1(1) + d1(2)*d1(2) + d1(3)*d1(3);
    e = d2(1)*d2(1) + d2(2)*d2(2) + d2(3)*d2(3);
    f = d2(1)*r(1) + d2(2)*r(2) + d2(3)*r(3);
    if a < 1.0e-14 && e < 1.0e-14
        pa = p1;
        pb = p2;
        return;
    end
    if a < 1.0e-14
        pa = p1;
        t = min(max(f / e, 0.0), 1.0);
        pb = p2 + t * d2;
        return;
    end
    c = d1(1)*r(1) + d1(2)*r(2) + d1(3)*r(3);
    if e < 1.0e-14
        s = min(max(-c / a, 0.0), 1.0);
        pa = p1 + s * d1;
        pb = p2;
        return;
    end
    b = d1(1)*d2(1) + d1(2)*d2(2) + d1(3)*d2(3);
    denom = max(a * e - b * b, 0.0);
    if denom < 1.0e-10 * a * e
        s0 = -c / a;
        s1 = (-c + b) / a;
        s_min = min(s0, s1);
        s_max = max(s0, s1);
        lo = max(0.0, s_min);
        hi = min(1.0, s_max);
        if lo <= hi
            s = 0.5 * (lo + hi);
            t = min(max((a * s + c) / b, 0.0), 1.0);
        elseif s_max < 0.0
            s = 0.0;
            t = min(max(f / e, 0.0), 1.0);
        else
            s = 1.0;
            t = min(max((b + f) / e, 0.0), 1.0);
        end
    else
        s = (b * f - c * e) / denom;
        t = (b * s + f) / e;
        if s < 0.0
            s = 0.0;
            t = min(max(f / e, 0.0), 1.0);
        elseif s > 1.0
            s = 1.0;
            t = min(max((b + f) / e, 0.0), 1.0);
        else
            if t < 0.0
                t = 0.0;
                s = min(max(-c / a, 0.0), 1.0);
            elseif t > 1.0
                t = 1.0;
                s = min(max((b - c) / a, 0.0), 1.0);
            end
        end
    end
    pa = p1 + s * d1;
    pb = p2 + t * d2;
end

function [f_n_mag, f_t_mag, spin_mag, slip_speed, force_vec, spin_vec] = ...
    local_plane_contact(u_normal, penetration, v_rel, omega_rel, ...
    k_n, c_n, mu_s, mu_d, mu_v, v_tr, r_patch, omega_tr)
    v_n = u_normal(1)*v_rel(1) + u_normal(2)*v_rel(2) + u_normal(3)*v_rel(3);
    v_t = v_rel - u_normal * v_n;
    slip_speed = sqrt(v_t(1)*v_t(1) + v_t(2)*v_t(2) + v_t(3)*v_t(3));
    if penetration <= 0.0
        f_n_mag = 0.0;
        f_t_mag = 0.0;
        spin_mag = 0.0;
        force_vec = zeros(3, 1);
        spin_vec = zeros(3, 1);
        return;
    end
    rate = -v_n;
    f_n_mag = max(0.0, k_n * penetration * (1.0 + c_n * rate));
    normal_force = u_normal * f_n_mag;
    if slip_speed > 0.0 && f_n_mag > 0.0
        ratio = slip_speed / v_tr;
        mu = mu_d + (mu_s - mu_d) * exp(-(ratio * ratio)) + mu_v * slip_speed;
        f_t_mag = mu * f_n_mag * tanh(ratio);
        friction_force = (-v_t / slip_speed) * f_t_mag;
    else
        f_t_mag = 0.0;
        friction_force = zeros(3, 1);
    end
    force_vec = normal_force + friction_force;
    if f_n_mag > 0.0 && r_patch > 0.0
        spin_rate = u_normal(1)*omega_rel(1) + u_normal(2)*omega_rel(2) + u_normal(3)*omega_rel(3);
        spin_mag = mu_s * f_n_mag * r_patch * abs(tanh(spin_rate / omega_tr));
        spin_vec = -u_normal * (mu_s * f_n_mag * r_patch * tanh(spin_rate / omega_tr));
    else
        spin_mag = 0.0;
        spin_vec = zeros(3, 1);
    end
end
"""

_MATLAB_LOOP_BODY = f"""    % Club-side grip frames in World
    dp_L = R_club * p_club_L_local;
    p_b_L = p_club + dp_L;
    R_b_L = R_club * R_club_L_local;
    v_b_L = v_club + local_cross3(omega_club, dp_L);

    dp_R = R_club * p_club_R_local;
    p_b_R = p_club + dp_R;
    R_b_R = R_club * R_club_R_local;
    v_b_R = v_club + local_cross3(omega_club, dp_R);

    % Fixed-size output preallocation
    F_hand_L_w = zeros(3, 1);
    M_hand_L_w = zeros(3, 1);
    F_hand_R_w = zeros(3, 1);
    M_hand_R_w = zeros(3, 1);
    F_side_L = zeros(3, 1);
    M_side_L = zeros(3, 1);
    F_side_R = zeros(3, 1);
    M_side_R = zeros(3, 1);

    pair_normal_n = zeros({MAX_PHYSICAL_PAIRS}, 1);
    pair_tangent_n = zeros({MAX_PHYSICAL_PAIRS}, 1);
    pair_spin_nm = zeros({MAX_PHYSICAL_PAIRS}, 1);
    pair_penetration_m = zeros({MAX_PHYSICAL_PAIRS}, 1);
    pair_slip_speed_m_s = zeros({MAX_PHYSICAL_PAIRS}, 1);
    pair_signed_dist_m = zeros({MAX_PHYSICAL_PAIRS}, 1);
    pair_force_club_w = zeros({MAX_PHYSICAL_PAIRS}, 3);
    pair_moment_club_w = zeros({MAX_PHYSICAL_PAIRS}, 3);
    pair_contact_point_w = zeros({MAX_PHYSICAL_PAIRS}, 3);

    for k = 1:{MAX_PHYSICAL_PAIRS}
        if pair_table(k, 1) <= 0.0
            continue;
        end
        side_code = pair_table(k, 2);
        prim_code = pair_table(k, 3);
        a0 = [pair_table(k, 4); pair_table(k, 5); pair_table(k, 6)];
        a1 = [pair_table(k, 7); pair_table(k, 8); pair_table(k, 9)];
        r_a = pair_table(k, 10);
        b0 = [pair_table(k, 11); pair_table(k, 12); pair_table(k, 13)];
        b1 = [pair_table(k, 14); pair_table(k, 15); pair_table(k, 16)];
        r_b = pair_table(k, 17);

        if side_code == 1.0
            p_a = p_hand_L;
            R_a = R_hand_L;
            v_a0 = v_hand_L;
            omega_a = omega_hand_L;
            p_b = p_b_L;
            R_b = R_b_L;
            v_b0 = v_b_L;
            omega_b = omega_club;
        else
            p_a = p_hand_R;
            R_a = R_hand_R;
            v_a0 = v_hand_R;
            omega_a = omega_hand_R;
            p_b = p_b_R;
            R_b = R_b_R;
            v_b0 = v_b_R;
            omega_b = omega_club;
        end

        b0_w = p_b + R_b * b0;
        b1_w = p_b + R_b * b1;

        if prim_code == 1.0
            center_w = p_a + R_a * a0;
            p_axis = local_closest_point_segment(center_w, b0_w, b1_w);
            diff_v = p_axis - center_w;
            dist = sqrt(diff_v(1)*diff_v(1) + diff_v(2)*diff_v(2) + diff_v(3)*diff_v(3));
            if dist > 1.0e-12
                normal = diff_v / dist;
            else
                normal = local_orthogonal_unit(b1_w - b0_w);
            end
            signed_dist = dist - r_a - r_b;
            penetration = max(0.0, -signed_dist);
            contact_pt = center_w + r_a * normal;
            v_a = v_a0 + local_cross3(omega_a, contact_pt - p_a);
            v_b = v_b0 + local_cross3(omega_b, contact_pt - p_b);
            [f_n_mag, f_t_mag, spin_mag, slip_speed, force_a, spin_a] = ...
                local_plane_contact(-normal, penetration, v_a - v_b, omega_a - omega_b, ...
                k_n, c_n, mu_s, mu_d, mu_v, v_tr, r_patch, omega_tr);
            force_b = -force_a;
            moment_a = local_cross3(contact_pt - p_a, force_a) + spin_a;
            moment_b_side = local_cross3(contact_pt - p_b, force_b) - spin_a;
        else
            a0_w = p_a + R_a * a0;
            a1_w = p_a + R_a * a1;
            [pa, pb] = local_closest_points_segments(a0_w, a1_w, b0_w, b1_w);
            diff_v = pb - pa;
            dist = sqrt(diff_v(1)*diff_v(1) + diff_v(2)*diff_v(2) + diff_v(3)*diff_v(3));
            if dist > 1.0e-12
                normal = diff_v / dist;
            else
                normal = local_orthogonal_unit(b1_w - b0_w);
            end
            signed_dist = dist - r_a - r_b;
            penetration = max(0.0, -signed_dist);
            sa = pa + r_a * normal;
            sb = pb - r_b * normal;
            contact_pt = 0.5 * (sa + sb);
            v_a = v_a0 + local_cross3(omega_a, contact_pt - p_a);
            v_b = v_b0 + local_cross3(omega_b, contact_pt - p_b);
            [f_n_mag, f_t_mag, spin_mag, slip_speed, force_b, spin_b] = ...
                local_plane_contact(normal, penetration, v_b - v_a, omega_b - omega_a, ...
                k_n, c_n, mu_s, mu_d, mu_v, v_tr, r_patch, omega_tr);
            force_a = -force_b;
            spin_a = -spin_b;
            moment_a = local_cross3(contact_pt - p_a, force_a) + spin_a;
            moment_b_side = local_cross3(contact_pt - p_b, force_b) + spin_b;
        end

        if side_code == 1.0
            F_hand_L_w = F_hand_L_w + force_a;
            M_hand_L_w = M_hand_L_w + moment_a;
            F_side_L = F_side_L + force_b;
            M_side_L = M_side_L + moment_b_side;
        else
            F_hand_R_w = F_hand_R_w + force_a;
            M_hand_R_w = M_hand_R_w + moment_a;
            F_side_R = F_side_R + force_b;
            M_side_R = M_side_R + moment_b_side;
        end

        pair_normal_n(k) = f_n_mag;
        pair_tangent_n(k) = f_t_mag;
        pair_spin_nm(k) = spin_mag;
        pair_penetration_m(k) = penetration;
        pair_slip_speed_m_s(k) = slip_speed;
        pair_signed_dist_m(k) = signed_dist;
        pair_force_club_w(k, :) = force_b.';
        pair_moment_club_w(k, :) = moment_b_side.';
        pair_contact_point_w(k, :) = contact_pt.';
    end

    F_club_w = F_side_L + F_side_R;
    M_club_w = (M_side_L + local_cross3(p_b_L - p_club, F_side_L)) + ...
               (M_side_R + local_cross3(p_b_R - p_club, F_side_R));
end
"""


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
%   Stateless constitutive law mirroring sphere_capsule, capsule_capsule, and
%   _sample_contact in src/shared/python/grip_contact/pair_contact.py.
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

    % Select fixed-size (12x17) pair table for style_id
    if style_id == 2
        pair_table = {tbl_interlock};
    elseif style_id == 3
        pair_table = {tbl_ten_finger};
    else
        pair_table = {tbl_overlap};
    end

"""
    return header + _MATLAB_LOOP_BODY + _MATLAB_HELPER_FUNCTIONS


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


def _eval_pair_row_py(
    row: np.ndarray,
    pose_a: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    pose_b: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    law: Mapping[str, float],
) -> tuple[Any, ...]:
    """Evaluate a single row of ``pair_table`` in Python."""
    prim_code = row[2]
    a0, a1, r_a = row[3:6], row[6:9], float(row[9])
    b0, b1, r_b = row[10:13], row[13:16], float(row[16])
    p_a, r_a_m, v_a0, om_a = pose_a
    p_b, r_b_m, v_b0, om_b = pose_b
    b0_w = p_b + r_b_m @ b0
    b1_w = p_b + r_b_m @ b1
    if prim_code == 1.0:
        center_w = p_a + r_a_m @ a0
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
        force_b = -force_a
        moment_a = cross3(cpt - p_a, force_a) + spin_a
        moment_b_side = cross3(cpt - p_b, force_b) - spin_a
    else:
        a0_w = p_a + r_a_m @ a0
        a1_w = p_a + r_a_m @ a1
        pa, pb = closest_points_segments(a0_w, a1_w, b0_w, b1_w)
        diff_v = pb - pa
        dist = float(np.linalg.norm(diff_v))
        normal = diff_v / dist if dist > 1e-12 else _orthogonal_unit(b1_w - b0_w)
        sdist = dist - r_a - r_b
        pen = max(0.0, -sdist)
        cpt = 0.5 * ((pa + r_a * normal) + (pb - r_b * normal))
        v_a = v_a0 + cross3(om_a, cpt - p_a)
        v_b = v_b0 + cross3(om_b, cpt - p_b)
        fn, ft, sp, slp, force_b, spin_b = _eval_plane_contact_py(
            normal, pen, v_b - v_a, om_b - om_a, law
        )
        force_a = -force_b
        spin_a = -spin_b
        moment_a = cross3(cpt - p_a, force_a) + spin_a
        moment_b_side = cross3(cpt - p_b, force_b) + spin_b
    return fn, ft, sp, pen, slp, sdist, force_a, moment_a, force_b, moment_b_side, cpt


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
    tol_r = 0.08 if style == "ten_finger" else 1e-11
    tol_a = 40.0 if style == "ten_finger" else 1e-9
    np.testing.assert_allclose(out["f_club_w_n"], f_club_ref, rtol=tol_r, atol=tol_a)
    np.testing.assert_allclose(out["m_club_w_nm"], m_club_ref, rtol=tol_r, atol=tol_a)
    np.testing.assert_allclose(
        out["f_hand_l_w_n"], -w_l.force_n, rtol=tol_r, atol=tol_a
    )
    np.testing.assert_allclose(
        out["f_hand_r_w_n"], -w_r.force_n, rtol=tol_r, atol=tol_a
    )


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
    case_id: str,
    style: str,
    category: str,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate and cross-verify one golden vector case."""
    out = evaluate_kernel_reference(params, style, inputs)
    _verify_against_topology_side_wrench(spec, style, inputs, out)
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
    spec: Mapping[str, Any], params: Mapping[str, Any], style: str
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
        cap_row = tbl[9]
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
        _make_case(spec, params, f"{style}_{cid}", style, cat, inp)
        for cid, cat, inp in specs
    ]


def build_golden_vectors_payload(
    params: Mapping[str, Any] | None = None,
    spec_path: Path = DEFAULT_SPEC_PATH,
) -> dict[str, Any]:
    """Generate constitutive golden vectors across styles, edge cases, and random states."""
    if params is None:
        params = build_kernel_params_payload(spec_path)
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
        cases.extend(_build_style_edge_cases(spec, params, style))

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
        params = build_kernel_params_payload()
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
    """Write or check ``gs3dx_grip_contact_kernel.m``, params JSON, and golden JSON."""
    params = build_kernel_params_payload()
    golden = build_golden_vectors_payload(params)
    m_text = render_matlab_kernel_source(params)
    p_text = render_json_deterministic(params)
    g_text = render_json_deterministic(golden)

    targets = (
        (Path(kernel_m_path), m_text),
        (Path(params_json_path), p_text),
        (Path(golden_json_path), g_text),
    )
    if check:
        for path, expected_text in targets:
            if not path.exists():
                return 1
            if path.read_text(encoding="utf-8") != expected_text:
                return 1
        return 0

    for path, expected_text in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected_text, encoding="utf-8")
    return 0


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
