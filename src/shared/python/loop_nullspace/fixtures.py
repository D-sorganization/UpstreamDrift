"""Analytic closed-loop oracle fixtures for NSX (#12557, NSX-12).

Provides deterministic ground-truth ClosedLoopSnapshot and GoldenValues pairs
backed by the SymPy derivations in NSX-00 (#12549), plus the rank-2 audit of
pendulum_simulator.golfer_constraints.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.linalg

from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
    FloatArray,
    FootContact,
    GripFrame,
    InputProvenance,
    LoopTopology,
    ModelProvenance,
    SpatialInertia,
    _readonly_float_array,
)
from src.shared.python.pendulum_simulator.golfer_constraints import (
    analytical_constraint_jacobian,
    constraint_vector,
)
from src.shared.python.pendulum_simulator.physics_golfer import GolferParams

GOLDEN_FIXTURES_PATH: Path = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "research"
    / "loop_nullspace"
    / "derivations"
    / "golden_fixtures.json"
)

_NSX00_FALLBACK_FIXTURES: dict[str, Any] = {
    "planar_two_arm_fixed_base": {
        "description": "5-DOF planar two-arm + club loop with full actuation and 2-D point closure (PLANAR_POINT2)",
        "closure_kind": "PLANAR_POINT2",
        "A": [
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.75],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.25, 0.5],
            [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, -0.5, -0.75],
        ],
        "b": [2.4, -1.4, 0.6, 1.8, -0.8],
        "J_c": [
            [1.0, 0.5, 0.25, -1.0, -0.5],
            [0.0, 0.75, 0.5, 0.0, -0.75],
        ],
        "Jdot_v": [0.125, -0.08333333333333333],
        "M": [
            [2.5, 0.5, 0.2, 0.0, 0.0],
            [0.5, 1.5, 0.25, 0.0, 0.0],
            [0.2, 0.25, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.4],
            [0.0, 0.0, 0.0, 0.4, 1.2],
        ],
        "h": [0.5, -0.25, 0.1, 0.3, -0.2],
        "x_particular": [
            2.244060913705584,
            -1.3875126903553299,
            0.6213197969543147,
            1.9559390862944162,
            -0.8124873096446701,
            0.15593908629441625,
            -0.12060913705583756,
        ],
        "null_basis": [
            [-1.0, 0.0],
            [-0.5, -0.75],
            [-0.25, -0.5],
            [1.0, 0.0],
            [0.5, 0.75],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        "tau_min_Minv": [
            2.3619958606030096,
            -1.4929238686580522,
            0.5412177658443811,
            1.8380041393969906,
            -0.7070761313419478,
        ],
        "a_ztcf": [
            -0.30723191366755725,
            0.324021687883074,
            -0.0858085808580858,
            -0.11952564304049453,
            0.1557048561999057,
        ],
        "lambda_zero_feasible": True,
        "dimensions": {
            "n_v": 5,
            "n_tau": 5,
            "n_u": 0,
            "m_c": 2,
            "m_g": 0,
            "dim_x": 7,
            "rank_A": 5,
            "equality_nullity": 2,
            "multiplier_gauge": 0,
            "rank_Jc": 2,
            "rank_Jc_u": 0,
            "attainable_torque_variation": 2,
            "feasible_set_dim": 2,
        },
    },
    "planar_two_arm_underactuated": {
        "description": "5-DOF planar two-arm + unactuated club angle (n_tau=4, n_u=1, m_c=2)",
        "closure_kind": "PLANAR_POINT2",
        "actuated_indices": [0, 1, 3, 4],
        "unactuated_indices": [2],
        "A": [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.5, 0.75],
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.5],
            [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -0.5, -0.75],
        ],
        "b": [2.4, -1.4, 0.6, 1.8, -0.8],
        "J_c": [
            [1.0, 0.5, 0.25, -1.0, -0.5],
            [0.0, 0.75, 0.5, 0.0, -0.75],
        ],
        "x_particular": [
            2.125714285714286,
            -2.334285714285714,
            2.0742857142857143,
            0.13428571428571429,
            0.2742857142857143,
            1.062857142857143,
        ],
        "null_basis": [[2.0], [0.25], [-2.0], [-0.25], [-2.0], [1.0]],
        "lambda_zero_feasible": False,
        "dimensions": {
            "n_v": 5,
            "n_tau": 4,
            "n_u": 1,
            "m_c": 2,
            "m_g": 0,
            "dim_x": 6,
            "rank_A": 5,
            "equality_nullity": 1,
            "multiplier_gauge": 0,
            "rank_Jc": 2,
            "rank_Jc_u": 1,
            "attainable_torque_variation": 1,
            "feasible_set_dim": 1,
        },
    },
    "floating_base_two_arm": {
        "description": "7-DOF planar floating-base model (n_u=3, n_tau=4, m_c=2 hand weld, m_g=4 two-foot contact)",
        "closure_kind": "PLANAR_POINT2",
        "actuated_indices": [3, 4, 5, 6],
        "unactuated_indices": [0, 1, 2],
        "A": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.5,
                -0.3333333333333333,
                0.5,
                0.3333333333333333,
            ],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.5, 0.75, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -0.5, -0.75, 0.0, 0.0, 0.0, 0.0],
        ],
        "b": [1.5, 9.5, 0.5, 2.0, -1.0, 1.0, -0.5],
        "J_c": [
            [0.0, 0.0, 0.0, 1.0, 0.5, -1.0, -0.5],
            [0.0, 0.0, 0.0, 0.0, 0.75, 0.0, -0.75],
        ],
        "J_g": [
            [1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -0.3333333333333333, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0],
        ],
        "x_particular": [
            1.7272727272727273,
            -0.9318181818181818,
            1.2727272727272727,
            -0.5681818181818182,
            0.2727272727272727,
            -0.2727272727272727,
            0.40384615384615385,
            4.519230769230769,
            1.0961538461538463,
            4.980769230769231,
        ],
        "null_basis": [
            [-1.0, 0.0, 0.0],
            [-0.5, -0.75, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.75, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.6666666666666666],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -0.6666666666666666],
            [0.0, 0.0, 1.0],
        ],
        "base_centroidal_wrench_invariant": [1.5, 9.5, 0.5],
        "dimensions": {
            "n_v": 7,
            "n_tau": 4,
            "n_u": 3,
            "m_c": 2,
            "m_g": 4,
            "dim_x": 10,
            "rank_A": 7,
            "equality_nullity": 3,
            "multiplier_gauge": 0,
            "rank_Jc": 2,
            "rank_Jc_u": 0,
            "attainable_torque_variation": 2,
            "feasible_set_dim": 3,
        },
    },
    "straight_arm_rank_deficient": {
        "description": "4-DOF straight-arm singular configuration where m_c=2 but rank(J_c)=1 (multiplier_gauge=1)",
        "closure_kind": "PLANAR_POINT2",
        "A": [
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, -0.5],
        ],
        "b": [1.0, 0.5, -1.0, -0.5],
        "J_c": [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, -1.0, -0.5],
        ],
        "x_particular": [
            0.2857142857142857,
            0.14285714285714285,
            -0.2857142857142857,
            -0.14285714285714285,
            0.0,
            0.7142857142857143,
        ],
        "null_basis": [
            [0.0, -1.0],
            [0.0, -0.5],
            [0.0, 1.0],
            [0.0, 0.5],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        "dimensions": {
            "n_v": 4,
            "n_tau": 4,
            "n_u": 0,
            "m_c": 2,
            "m_g": 0,
            "dim_x": 6,
            "rank_A": 4,
            "equality_nullity": 2,
            "multiplier_gauge": 1,
            "rank_Jc": 1,
            "rank_Jc_u": 0,
            "attainable_torque_variation": 1,
            "feasible_set_dim": 2,
        },
    },
    "bare_club_partition": {
        "bare_club_mass_kg": 0.313,
        "welded_hands_mass_kg": 1.1082,
        "composite_mass_kg": 1.4212,
    },
    "c_plus_minus_identity": {
        "v_admissible": [
            -0.8333333333333334,
            -1.3333333333333333,
            2.0,
            -1.0,
            0.0,
        ],
    },
    "analytic_envelope_case": {
        "z_min": [-2.0, 1.0],
        "z_max": [4.0, 5.0],
        "sign_class": ["INDETERMINATE", "POSITIVE"],
        "chebyshev_center": [1.0, 3.0],
        "chebyshev_radius": 2.0,
    },
}


def load_golden_fixtures(path: Path | None = None) -> dict[str, Any]:
    """Load NSX-00 golden fixtures from JSON on disk or deterministic fallback."""
    target = GOLDEN_FIXTURES_PATH if path is None else path
    if target.is_file():
        payload = json.loads(target.read_text(encoding="utf-8"))
        return dict(payload.get("fixtures", payload))
    return dict(_NSX00_FALLBACK_FIXTURES)


@dataclass(frozen=True)
class GoldenValues(Mapping[str, Any]):
    """Typed and subscriptable ground-truth record for an NSX oracle fixture."""

    fixture_id: str
    description: str
    closure_kind: ClosureKind
    A: FloatArray
    b: FloatArray
    J_c: FloatArray
    Jdot_v: FloatArray
    M: FloatArray
    h: FloatArray
    x_particular: FloatArray
    null_basis: FloatArray
    dimensions: Mapping[str, int]
    tau_min_Minv: FloatArray | None = None
    a_ztcf: FloatArray | None = None
    lambda_zero_feasible: bool | None = None
    J_g: FloatArray | None = None
    base_centroidal_wrench_invariant: FloatArray | None = None
    z_min: FloatArray | None = None
    z_max: FloatArray | None = None
    tau_min: FloatArray | None = None
    tau_max: FloatArray | None = None
    lambda_min: FloatArray | None = None
    lambda_max: FloatArray | None = None
    sign_class: tuple[str, ...] | None = None
    chebyshev_center: FloatArray | None = None
    chebyshev_radius: float | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "closure_kind", ClosureKind(self.closure_kind))
        for field_name, ndim in (
            ("A", 2),
            ("b", 1),
            ("J_c", 2),
            ("Jdot_v", 1),
            ("M", 2),
            ("h", 1),
            ("x_particular", 1),
            ("null_basis", 2),
        ):
            val = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                _readonly_float_array(val, field_name, ndim=ndim),
            )
        for opt_name, ndim in (
            ("tau_min_Minv", 1),
            ("a_ztcf", 1),
            ("J_g", 2),
            ("base_centroidal_wrench_invariant", 1),
            ("z_min", 1),
            ("z_max", 1),
            ("tau_min", 1),
            ("tau_max", 1),
            ("lambda_min", 1),
            ("lambda_max", 1),
            ("chebyshev_center", 1),
        ):
            opt_val = getattr(self, opt_name)
            if opt_val is not None:
                object.__setattr__(
                    self,
                    opt_name,
                    _readonly_float_array(opt_val, opt_name, ndim=ndim),
                )

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key) and key != "raw":
            return getattr(self, key)
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        keys = set(self.raw.keys()) | {
            "fixture_id",
            "description",
            "closure_kind",
            "A",
            "b",
            "J_c",
            "Jdot_v",
            "M",
            "h",
            "x_particular",
            "null_basis",
            "dimensions",
            "tau_min_Minv",
            "a_ztcf",
            "lambda_zero_feasible",
            "J_g",
            "base_centroidal_wrench_invariant",
            "z_min",
            "z_max",
            "tau_min",
            "tau_max",
            "lambda_min",
            "lambda_max",
            "sign_class",
            "chebyshev_center",
            "chebyshev_radius",
        }
        return iter(sorted(keys))

    def __len__(self) -> int:
        return sum(1 for _ in self)


@dataclass(frozen=True)
class GolferConstraintRankAudit(Mapping[str, Any]):
    """Audit result proving golfer_constraints has rank 2 and zero wrist columns."""

    full_shape: tuple[int, int]
    rank: int
    independent_rows: tuple[int, ...]
    redundant_rows: tuple[int, ...]
    zero_columns: tuple[int, ...]
    projection_residual: float
    jacobian_projection_residual: float
    independent_jacobian: FloatArray

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(
            (
                "full_shape",
                "rank",
                "independent_rows",
                "redundant_rows",
                "zero_columns",
                "projection_residual",
                "jacobian_projection_residual",
                "independent_jacobian",
            )
        )

    def __len__(self) -> int:
        return 8


def _default_closed_loop_golfer_state() -> tuple[FloatArray, GolferParams]:
    """Return a closed-loop (q, params) pair for pendulum_simulator.golfer_constraints."""
    l_up, l_fore = 0.30, 0.28
    a_rs, a_re = -0.60, -0.15
    a_ls, a_le = 0.35, 0.10
    dx_r = l_up * float(np.sin(a_rs)) + l_fore * float(np.sin(a_rs + a_re))
    dy_r = -l_up * float(np.cos(a_rs)) - l_fore * float(np.cos(a_rs + a_re))
    dx_l = l_up * float(np.sin(a_ls)) + l_fore * float(np.sin(a_ls + a_le))
    dy_l = -l_up * float(np.cos(a_ls)) - l_fore * float(np.cos(a_ls + a_le))
    d_shoulder = 0.5 * (dx_l - dx_r)
    grip_r = 0.05
    grip_l = grip_r + (dy_r - dy_l)
    params = GolferParams(
        m_hub=0.5,
        m_r_upper=2.0,
        m_r_fore=1.2,
        m_l_upper=2.0,
        m_l_fore=1.2,
        m_club=0.313,
        L_hub=0.2,
        L_r_upper=l_up,
        L_r_fore=l_fore,
        L_l_upper=l_up,
        L_l_fore=l_fore,
        L_club=1.1,
        d_rs=d_shoulder,
        d_ls=d_shoulder,
        grip_right=grip_r,
        grip_left=grip_l,
    )
    q = np.array(
        [0.0, a_rs, a_re, 0.12, a_ls, a_le, -0.08, 0.0],
        dtype=np.float64,
    )
    return q, params


def audit_golfer_constraints_rank(
    q: np.ndarray | None = None,
    params: GolferParams | None = None,
    *,
    tol: float = 1e-10,
) -> GolferConstraintRankAudit:
    """Audit pendulum_simulator.golfer_constraints and select the 2 independent rows."""
    q_default, p_default = _default_closed_loop_golfer_state()
    q_eval = np.asarray(q_default if q is None else q, dtype=np.float64)[:8]
    p_eval = p_default if params is None else params

    phi = constraint_vector(q_eval, p_eval)
    phi_q = analytical_constraint_jacobian(q_eval, p_eval)

    th_club = float(q_eval[7])
    c_club, s_club = float(np.cos(th_club)), float(np.sin(th_club))
    rot_club = np.array([[c_club, s_club], [s_club, -c_club]], dtype=np.float64)
    club_dir = np.array([s_club, -c_club], dtype=np.float64)
    grip_sep = float(p_eval.grip_left - p_eval.grip_right)
    # Reconstruct (lh - rh) - grip_sep * club_dir from phi[:2] accounting for golfer_kinematics y-sign
    delta_grip = np.array([phi[0], phi[1] + 2.0 * grip_sep * c_club], dtype=np.float64)

    proj_res = float(np.linalg.norm(rot_club @ delta_grip - phi[2:4]))
    jac_proj_res = float(np.linalg.norm(rot_club @ phi_q[:2, :] - phi_q[2:4, :]))
    rank = int(np.linalg.matrix_rank(phi_q, tol=tol))
    zero_cols = tuple(
        int(j)
        for j in range(phi_q.shape[1])
        if float(np.linalg.norm(phi_q[:, j])) <= tol
    )
    indep_jac = _readonly_float_array(
        phi_q[:2, :], "independent_jacobian", shape=(2, 8)
    )
    return GolferConstraintRankAudit(
        full_shape=(int(phi_q.shape[0]), int(phi_q.shape[1])),
        rank=rank,
        independent_rows=(0, 1),
        redundant_rows=(2, 3),
        zero_columns=zero_cols,
        projection_residual=proj_res,
        jacobian_projection_residual=jac_proj_res,
        independent_jacobian=indep_jac,
    )


def _standard_grip_and_inertias() -> tuple[
    GripFrame,
    SpatialInertia,
    tuple[SpatialInertia, ...],
    ModelProvenance,
    InputProvenance,
]:
    """Return canonical grip frame, bare-club inertia, welded hand inertias, and provenance."""
    grip_frame = GripFrame(
        rotation=np.eye(3, dtype=np.float64),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )
    bare_club = SpatialInertia(
        mass_kg=0.313,
        com_m=np.array([0.38, 0.0, 0.0], dtype=np.float64),
        inertia_com_kg_m2=np.diag([0.0008, 0.042, 0.042]),
        solid_names=("Clubhead", "Rigid Shaft", "Grip"),
    )
    hand_l = SpatialInertia(
        mass_kg=0.5591,
        com_m=np.array([-0.04, 0.0, 0.0], dtype=np.float64),
        inertia_com_kg_m2=np.diag([0.0004, 0.0007, 0.0007]),
        solid_names=("LHand", "LHandStandoff"),
    )
    hand_r = SpatialInertia(
        mass_kg=0.5491,
        com_m=np.array([0.04, 0.0, 0.0], dtype=np.float64),
        inertia_com_kg_m2=np.diag([0.0004, 0.0007, 0.0007]),
        solid_names=("RHand",),
    )
    model_prov = ModelProvenance(
        spec_sha256="12549" + "0" * 59,
        inertial_source="nsx_00_sympy_golden",
        bare_club_bench_id="nsx-00-driver-bare-club",
        engine_name="sympy_oracle",
    )
    input_prov = InputProvenance(
        kinematic_source="nsx_00_analytic_fixture",
        derivative_method="exact_symbolic",
        time_sync_status="SYNCHRONIZED",
    )
    return grip_frame, bare_club, (hand_l, hand_r), model_prov, input_prov


def _selection_matrix(actuated_indices: tuple[int, ...], n_v: int) -> FloatArray:
    """Build (n_tau, n_v) selection matrix S from actuated DOF indices."""
    S = np.zeros((len(actuated_indices), n_v), dtype=np.float64)
    for row_idx, dof_idx in enumerate(actuated_indices):
        S[row_idx, dof_idx] = 1.0
    return S


def planar_golfer_snapshot(
    config: Mapping[str, Any] | None = None,
    *,
    actuation: Literal["full", "club_unactuated"] = "full",
    closure_kind: ClosureKind
    | Literal["PLANAR_POINT2", "PLANAR_WELD3"] = ClosureKind.PLANAR_POINT2,
) -> tuple[ClosedLoopSnapshot, GoldenValues]:
    """Return the 5-DOF planar two-arm + club snapshot and NSX-00 golden values."""
    cfg = dict(config or {})
    act_mode = str(cfg.get("actuation", actuation))
    kind = ClosureKind(cfg.get("closure_kind", closure_kind))
    if act_mode not in ("full", "club_unactuated"):
        raise ValueError(
            f"actuation must be 'full' or 'club_unactuated', got {act_mode!r}"
        )
    if kind not in (ClosureKind.PLANAR_POINT2, ClosureKind.PLANAR_WELD3):
        raise ValueError(
            f"planar_golfer_snapshot requires PLANAR_POINT2 or PLANAR_WELD3, got {kind}"
        )

    fixtures = load_golden_fixtures()
    base_data = fixtures["planar_two_arm_fixed_base"]
    under_data = fixtures["planar_two_arm_underactuated"]
    raw_data = base_data if act_mode == "full" else under_data
    fixture_id = (
        "planar_two_arm_fixed_base"
        if act_mode == "full"
        else "planar_two_arm_underactuated"
    )

    actuated = (0, 1, 2, 3, 4) if act_mode == "full" else (0, 1, 3, 4)
    unactuated: tuple[int, ...] = () if act_mode == "full" else (2,)
    topology = LoopTopology(
        closure_kind=kind,
        closing_side="right",
        holding_side="left",
        actuated_indices=actuated,
        unactuated_indices=unactuated,
        loop_support_indices=(0, 1, 2, 3, 4),
        coordinate_order=("q_L1", "q_L2", "q_club", "q_R1", "q_R2"),
    )

    M = np.asarray(base_data["M"], dtype=np.float64)
    h = np.asarray(base_data["h"], dtype=np.float64)
    b = np.asarray(base_data["b"], dtype=np.float64)
    a = np.linalg.solve(M, b - h)
    v = np.asarray(fixtures["c_plus_minus_identity"]["v_admissible"], dtype=np.float64)
    q = np.zeros(5, dtype=np.float64)
    S = _selection_matrix(actuated, 5)

    J_c_2 = np.asarray(base_data["J_c"], dtype=np.float64)
    Jdot_2 = np.asarray(base_data["Jdot_v"], dtype=np.float64)
    if kind is ClosureKind.PLANAR_POINT2:
        J_c = J_c_2
        Jdot_v = Jdot_2
        A = np.asarray(raw_data["A"], dtype=np.float64)
        x_p = np.asarray(raw_data["x_particular"], dtype=np.float64)
        null_basis = np.asarray(raw_data["null_basis"], dtype=np.float64)
        dims = dict(raw_data["dimensions"])
    else:
        orient_row = np.array([[1.0, 1.0, 1.0, -1.0, -1.0]], dtype=np.float64)
        J_c = np.vstack([J_c_2, orient_row])
        Jdot_v = np.append(Jdot_2, 0.0)
        A = np.hstack([S.T, J_c.T])
        x_p = np.linalg.lstsq(A, b, rcond=None)[0]
        null_basis = scipy.linalg.null_space(A)
        rank_jc = int(np.linalg.matrix_rank(J_c))
        rank_jcu = int(np.linalg.matrix_rank(J_c[:, unactuated])) if unactuated else 0
        dims = {
            "n_v": 5,
            "n_tau": len(actuated),
            "n_u": len(unactuated),
            "m_c": 3,
            "m_g": 0,
            "dim_x": len(actuated) + 3,
            "rank_A": int(np.linalg.matrix_rank(A)),
            "equality_nullity": int(null_basis.shape[1]),
            "multiplier_gauge": 3 - rank_jc,
            "rank_Jc": rank_jc,
            "rank_Jc_u": rank_jcu,
            "attainable_torque_variation": rank_jc - rank_jcu,
            "feasible_set_dim": int(null_basis.shape[1]),
        }

    grip_frame, bare_club, hands, model_prov, input_prov = _standard_grip_and_inertias()
    snapshot = ClosedLoopSnapshot(
        q=q,
        v=v,
        a=a,
        M=M,
        h=h,
        S=S,
        J_c=J_c,
        Jdot_v=Jdot_v,
        grip_frame=grip_frame,
        bare_club=bare_club,
        welded_hand_inertias=hands,
        model_provenance=model_prov,
        input_provenance=input_prov,
        topology=topology,
    )
    golden = GoldenValues(
        fixture_id=fixture_id,
        description=str(raw_data["description"]),
        closure_kind=kind,
        A=A,
        b=b,
        J_c=J_c,
        Jdot_v=Jdot_v,
        M=M,
        h=h,
        x_particular=x_p,
        null_basis=null_basis,
        dimensions=dims,
        tau_min_Minv=base_data.get("tau_min_Minv")
        if act_mode == "full" and kind is ClosureKind.PLANAR_POINT2
        else None,
        a_ztcf=base_data.get("a_ztcf")
        if act_mode == "full" and kind is ClosureKind.PLANAR_POINT2
        else None,
        lambda_zero_feasible=bool(raw_data["lambda_zero_feasible"]),
        raw=raw_data,
    )
    return snapshot, golden


def straight_arm_snapshot() -> tuple[ClosedLoopSnapshot, GoldenValues]:
    """Return the 4-DOF rank-deficient straight-arm snapshot (rank(J_c) == 1 < m_c == 2)."""
    fixtures = load_golden_fixtures()
    raw = fixtures["straight_arm_rank_deficient"]
    topology = LoopTopology(
        closure_kind=ClosureKind.PLANAR_POINT2,
        closing_side="right",
        holding_side="left",
        actuated_indices=(0, 1, 2, 3),
        unactuated_indices=(),
        loop_support_indices=(0, 1, 2, 3),
        coordinate_order=("q_L1", "q_L2", "q_R1", "q_R2"),
    )
    M = np.eye(4, dtype=np.float64)
    h = np.zeros(4, dtype=np.float64)
    b = np.asarray(raw["b"], dtype=np.float64)
    J_c = np.asarray(raw["J_c"], dtype=np.float64)
    Jdot_v = np.zeros(2, dtype=np.float64)
    S = np.eye(4, dtype=np.float64)
    grip_frame, bare_club, hands, model_prov, input_prov = _standard_grip_and_inertias()
    snapshot = ClosedLoopSnapshot(
        q=np.zeros(4, dtype=np.float64),
        v=np.zeros(4, dtype=np.float64),
        a=b.copy(),
        M=M,
        h=h,
        S=S,
        J_c=J_c,
        Jdot_v=Jdot_v,
        grip_frame=grip_frame,
        bare_club=bare_club,
        welded_hand_inertias=hands,
        model_provenance=model_prov,
        input_provenance=input_prov,
        topology=topology,
    )
    golden = GoldenValues(
        fixture_id="straight_arm_rank_deficient",
        description=str(raw["description"]),
        closure_kind=ClosureKind.PLANAR_POINT2,
        A=np.asarray(raw["A"], dtype=np.float64),
        b=b,
        J_c=J_c,
        Jdot_v=Jdot_v,
        M=M,
        h=h,
        x_particular=np.asarray(raw["x_particular"], dtype=np.float64),
        null_basis=np.asarray(raw["null_basis"], dtype=np.float64),
        dimensions=dict(raw["dimensions"]),
        raw=raw,
    )
    return snapshot, golden


def spatial_two_arm_club_snapshot(
    *,
    floating_base: bool = False,
    include_feet: bool = False,
) -> tuple[ClosedLoopSnapshot, GoldenValues]:
    """Return a 3-D spatial two-arm (2x3-link) + rigid bare club WELD6 snapshot."""
    arm_coords = ("q_L1", "q_L2", "q_L3", "q_R1", "q_R2", "q_R3")
    base_coords = (
        ("base_vx", "base_vy", "base_vz", "base_wx", "base_wy", "base_wz")
        if floating_base
        else ()
    )
    coords = base_coords + arm_coords
    n_u = len(base_coords)
    n_v = len(coords)
    unactuated = tuple(range(n_u))
    actuated = tuple(range(n_u, n_v))

    topology = LoopTopology(
        closure_kind=ClosureKind.WELD6,
        closing_side="right",
        holding_side="left",
        actuated_indices=actuated,
        unactuated_indices=unactuated,
        loop_support_indices=actuated,
        coordinate_order=coords,
    )

    J_L = np.array(
        [
            [1.0, 0.5, 0.25],
            [0.0, 0.8, 0.4],
            [0.2, -0.3, 0.6],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.5],
        ],
        dtype=np.float64,
    )
    J_R = np.array(
        [
            [1.0, 0.4, 0.2],
            [0.1, 0.75, 0.35],
            [-0.2, 0.25, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.2],
            [1.0, 0.1, 0.4],
        ],
        dtype=np.float64,
    )
    J_c_arm = np.hstack([J_L, -J_R])
    J_c = (
        np.hstack([np.zeros((6, n_u), dtype=np.float64), J_c_arm])
        if floating_base
        else J_c_arm
    )
    Jdot_v = np.array([0.1, -0.05, 0.08, 0.02, -0.03, 0.04], dtype=np.float64)

    diag_m = np.linspace(1.5, 3.0, n_v, dtype=np.float64)
    M = np.diag(diag_m) + 0.1 * np.eye(n_v, k=1) + 0.1 * np.eye(n_v, k=-1)
    h = np.linspace(-0.4, 0.6, n_v, dtype=np.float64)
    S = _selection_matrix(actuated, n_v)

    b_arm = np.array([2.2, -1.5, 0.9, 1.7, -1.1, 0.6], dtype=np.float64)
    foot_contacts: tuple[FootContact, ...] = ()
    J_g: FloatArray | None = None
    if floating_base and include_feet:
        J_foot_l = np.zeros((3, n_v), dtype=np.float64)
        J_foot_r = np.zeros((3, n_v), dtype=np.float64)
        J_foot_l[:, :3] = np.eye(3)
        J_foot_r[:, :3] = np.eye(3)
        J_foot_l[:, 3:6] = np.array(
            [[0.0, 0.0, 0.2], [0.0, 0.0, -0.1], [-0.2, 0.1, 0.0]]
        )
        J_foot_r[:, 3:6] = np.array(
            [[0.0, 0.0, -0.2], [0.0, 0.0, -0.1], [0.2, 0.1, 0.0]]
        )
        J_g = np.vstack([J_foot_l, J_foot_r])
        foot_contacts = (
            FootContact(
                foot_id="left_foot",
                point_world_m=np.array([-0.1, 0.2, 0.0]),
                normal_world=np.array([0.0, 0.0, 1.0]),
                jacobian=J_foot_l,
                friction_coeff=0.6,
            ),
            FootContact(
                foot_id="right_foot",
                point_world_m=np.array([-0.1, -0.2, 0.0]),
                normal_world=np.array([0.0, 0.0, 1.0]),
                jacobian=J_foot_r,
                friction_coeff=0.6,
            ),
        )
        f_g_true = np.array([30.0, -20.0, 380.0, 25.0, 20.0, 400.0])
        b_base = J_g[:, :6].T @ f_g_true
        b = np.concatenate([b_base, b_arm])
        A = np.hstack([S.T, J_c.T, J_g.T])
    elif floating_base:
        b = np.concatenate([np.zeros(6, dtype=np.float64), b_arm])
        A = np.hstack([S.T, J_c.T])
    else:
        b = b_arm
        A = np.hstack([S.T, J_c.T])

    a = np.linalg.solve(M, b - h)
    v = np.zeros(n_v, dtype=np.float64)
    q = np.zeros(n_v, dtype=np.float64)
    x_p = np.linalg.lstsq(A, b, rcond=None)[0]
    null_basis = scipy.linalg.null_space(A)

    grip_frame, bare_club, hands, model_prov, input_prov = _standard_grip_and_inertias()
    snapshot = ClosedLoopSnapshot(
        q=q,
        v=v,
        a=a,
        M=M,
        h=h,
        S=S,
        J_c=J_c,
        Jdot_v=Jdot_v,
        grip_frame=grip_frame,
        bare_club=bare_club,
        welded_hand_inertias=hands,
        model_provenance=model_prov,
        input_provenance=input_prov,
        topology=topology,
        J_g=J_g,
        foot_contacts=foot_contacts,
    )
    m_g = 0 if J_g is None else int(J_g.shape[0])
    rank_jc = int(np.linalg.matrix_rank(J_c))
    rank_jcu = int(np.linalg.matrix_rank(J_c[:, unactuated])) if unactuated else 0
    dims = {
        "n_v": n_v,
        "n_tau": len(actuated),
        "n_u": n_u,
        "m_c": 6,
        "m_g": m_g,
        "dim_x": int(A.shape[1]),
        "rank_A": int(np.linalg.matrix_rank(A)),
        "equality_nullity": int(null_basis.shape[1]),
        "multiplier_gauge": 6 - rank_jc,
        "rank_Jc": rank_jc,
        "rank_Jc_u": rank_jcu,
        "attainable_torque_variation": rank_jc - rank_jcu,
        "feasible_set_dim": int(null_basis.shape[1]),
    }
    golden = GoldenValues(
        fixture_id="spatial_two_arm_club",
        description="Spatial 2x3-link arms + rigid bare club with WELD6 closure",
        closure_kind=ClosureKind.WELD6,
        A=A,
        b=b,
        J_c=J_c,
        Jdot_v=Jdot_v,
        M=M,
        h=h,
        x_particular=x_p,
        null_basis=null_basis,
        dimensions=dims,
        J_g=J_g,
    )
    return snapshot, golden


def analytic_envelope_case() -> tuple[ClosedLoopSnapshot, GoldenValues]:
    """Return the closed-form static envelope oracle fixture (z in [-2, 4] x [1, 5])."""
    snapshot, base_golden = planar_golfer_snapshot(actuation="full")
    fixtures = load_golden_fixtures()
    env_raw = fixtures["analytic_envelope_case"]

    z_min = np.asarray(env_raw["z_min"], dtype=np.float64)
    z_max = np.asarray(env_raw["z_max"], dtype=np.float64)
    x_p = base_golden.x_particular
    N = base_golden.null_basis

    term_min = np.minimum(N * z_min[None, :], N * z_max[None, :]).sum(axis=1)
    term_max = np.maximum(N * z_min[None, :], N * z_max[None, :]).sum(axis=1)
    x_min = x_p + term_min
    x_max = x_p + term_max

    golden = GoldenValues(
        fixture_id="analytic_envelope_case",
        description="5-DOF planar closed loop with closed-form null-space box z in [-2, 4] x [1, 5]",
        closure_kind=base_golden.closure_kind,
        A=base_golden.A,
        b=base_golden.b,
        J_c=base_golden.J_c,
        Jdot_v=base_golden.Jdot_v,
        M=base_golden.M,
        h=base_golden.h,
        x_particular=x_p,
        null_basis=N,
        dimensions=base_golden.dimensions,
        tau_min_Minv=base_golden.tau_min_Minv,
        a_ztcf=base_golden.a_ztcf,
        lambda_zero_feasible=base_golden.lambda_zero_feasible,
        z_min=z_min,
        z_max=z_max,
        tau_min=x_min[:5],
        tau_max=x_max[:5],
        lambda_min=x_min[5:],
        lambda_max=x_max[5:],
        sign_class=tuple(str(s) for s in env_raw["sign_class"]),
        chebyshev_center=np.asarray(env_raw["chebyshev_center"], dtype=np.float64),
        chebyshev_radius=float(env_raw["chebyshev_radius"]),
        raw={**dict(base_golden.raw), **dict(env_raw)},
    )
    return snapshot, golden
