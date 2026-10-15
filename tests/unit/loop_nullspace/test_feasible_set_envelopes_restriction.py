"""Unit tests for NSX-21 FeasibleSet, NSX-22 static envelopes, and NSX-26 restriction (#12564, #12565, #12569)."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np
import pytest

from src.shared.python.loop_nullspace.envelopes import (
    EnvelopeSeries,
    LinearQuantitySpec,
    QuantityEnvelope,
    chebyshev_ball,
    chebyshev_margin,
    classify_sign,
    static_envelopes,
)
from src.shared.python.loop_nullspace.feasible_set import (
    FeasibleSet,
    LinearProgramSpec,
    PolyhedralRowBlock,
    SecondOrderConeBlock,
    SecondOrderConeProgramSpec,
    cone_linearization_error_bound,
)
from src.shared.python.loop_nullspace.restriction import (
    REACTION_RESTRICTION_SENSITIVITY,
    RESTRICTION_ROW_INDICES,
    RestrictedFrameResult,
    RestrictionReport,
    restrict,
    restriction_sensitivity,
)
from src.shared.python.motion_matching.force_nullspace import (
    ForceNullSpace,
    FrictionCone,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _MockTopology:
    """Minimal topology carrier for welded 6-DOF closure testing."""

    closure_kind: str
    actuated_indices: tuple[int, ...]
    unactuated_indices: tuple[int, ...]
    coordinate_order: tuple[str, ...]


@dataclass(frozen=True)
class _MockSnapshot:
    """Minimal closed-loop snapshot carrier for restriction sensitivity tests."""

    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    M: np.ndarray
    h: np.ndarray
    S: np.ndarray
    J_c: np.ndarray
    Jdot_v: np.ndarray
    topology: _MockTopology
    J_g: np.ndarray | None = None

    def generalized_load(self) -> np.ndarray:
        """Return generalized inverse-dynamics load b = M @ a + h."""
        return np.asarray(self.M @ self.a + self.h, dtype=np.float64)

    def closure_rows(self) -> int:
        """Return number of closure rows."""
        return int(self.J_c.shape[0])


def _make_spatial_two_arm_snapshot(scale_moment_load: float = 1.0) -> _MockSnapshot:
    """Build a 6-DOF actuated two-arm WELD6 snapshot with full-rank 6x6 J_c."""
    n_v = 6
    j_l = np.array(
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
    j_r = np.array(
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
    j_c = np.hstack([j_l, -j_r])
    m_mat = np.diag(np.linspace(1.5, 3.0, n_v))
    h_vec = np.array([0.2, -0.1, 0.15, -0.2, 0.1, -0.15], dtype=np.float64)
    b_base = np.array([2.0, -1.5, 1.0, -2.5, 1.8, -1.2], dtype=np.float64)
    # Axial shaft moment load (row 3 of J_c): absorbed with zero joint torque under WELD6,
    # but unabsorbable under SHAFT_FREE5 / BALL3 once row 3 is restricted out.
    b_vec = b_base + scale_moment_load * (j_c[3, :] * 10.0)
    a_vec = np.linalg.solve(m_mat, b_vec - h_vec)
    topo = _MockTopology(
        closure_kind="WELD6",
        actuated_indices=(0, 1, 2, 3, 4, 5),
        unactuated_indices=(),
        coordinate_order=("q_L1", "q_L2", "q_L3", "q_R1", "q_R2", "q_R3"),
    )
    return _MockSnapshot(
        q=np.zeros(n_v, dtype=np.float64),
        v=np.zeros(n_v, dtype=np.float64),
        a=a_vec,
        M=m_mat,
        h=h_vec,
        S=np.eye(n_v, dtype=np.float64),
        J_c=j_c,
        Jdot_v=np.zeros(6, dtype=np.float64),
        topology=topo,
    )


def test_two_arm_oracle_analytic_envelope_minus5_to_7() -> None:
    """Two-arm oracle with tau_1 + tau_2 = 2 and box bounds matches [-5, 7] to < 1e-9."""
    # Balance equation: tau_1 + tau_2 = 2, with tau_3 = lambda where tau_1 - lambda = 2, tau_2 + lambda = 0
    # Equivalent A x = b: tau_1 + tau_2 = 2 -> particular [1, 1], null basis [1, -1]^T
    matrix_a = np.array([[1.0, 1.0]], dtype=np.float64)
    rhs_b = np.array([2.0], dtype=np.float64)
    decomp = ForceNullSpace.from_balance(matrix_a, rhs_b)

    # Under |tau_1| <= 10 and -5 <= tau_2 <= 7 (or tau_1 >= -5, tau_2 >= -5, |tau_i| <= 10):
    # tau_1(z) = 1 + n_0 z, tau_2(z) = 1 - n_0 z.
    # With tau_min = [-5.0, -5.0], tau_max = [10.0, 10.0]:
    # tau_1 >= -5 and tau_2 <= 10 => tau_1 = 2 - tau_2 >= -8 => tau_1 >= -5
    # tau_2 >= -5 and tau_1 <= 10 => tau_1 = 2 - tau_2 <= 7 => tau_1 in [-5, 7]!
    fs = FeasibleSet.from_hypotheses(
        decomp,
        ("H-TORQUE-CAPACITY-BOX",),
        capacity_bounds=(
            np.array([-5.0, -5.0], dtype=np.float64),
            np.array([10.0, 10.0], dtype=np.float64),
        ),
    )
    assert not fs.is_empty()
    assert fs.conflict_set() == ()

    series = static_envelopes([fs])
    assert series.series_label == "STATIC"
    assert series.evidence_class == "ENVELOPE"
    env_tau1 = series.envelopes[0]
    env_tau2 = series.envelopes[1]
    assert abs(float(env_tau1.lower[0]) - (-5.0)) < 1e-9
    assert abs(float(env_tau1.upper[0]) - 7.0) < 1e-9
    assert abs(float(env_tau2.lower[0]) - (-5.0)) < 1e-9
    assert abs(float(env_tau2.upper[0]) - 7.0) < 1e-9
    assert env_tau1.lower_status == ("BOUNDED",)
    assert env_tau1.upper_status == ("BOUNDED",)
    assert env_tau1.sign_class == ("INDETERMINATE",)


def test_nsx00_analytic_envelope_golden_and_chebyshev() -> None:
    """NSX-00 analytic envelope golden (z in [-2, 4] x [1, 5]) matches bounds, signs, and Chebyshev ball."""
    x_p = np.array(
        [
            2.244060913705584,
            -1.3875126903553299,
            0.6213197969543147,
            1.9559390862944162,
            -0.8124873096446701,
            0.15593908629441625,
            -0.12060913705583756,
        ],
        dtype=np.float64,
    )
    null_basis = np.array(
        [
            [-1.0, 0.0],
            [-0.5, -0.75],
            [-0.25, -0.5],
            [1.0, 0.0],
            [0.5, 0.75],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Box z in [-2, 4] x [1, 5]
    a_ub = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    b_ub = np.array([2.0, 4.0, -1.0, 5.0], dtype=np.float64)
    block = PolyhedralRowBlock(
        A_ub=a_ub,
        b_ub=b_ub,
        hypothesis_id="H-TORQUE-CAPACITY-BOX",
        label="z_box",
    )
    fs = FeasibleSet(
        x_particular=x_p,
        null_basis=null_basis,
        polyhedral_blocks=(block,),
        n_tau=5,
    )

    z_series = static_envelopes([fs], quantities="z")
    assert np.allclose(z_series.lower_matrix[0], [-2.0, 1.0], atol=1e-9)
    assert np.allclose(z_series.upper_matrix[0], [4.0, 5.0], atol=1e-9)
    assert z_series.envelopes[0].sign_class == ("INDETERMINATE",)
    assert z_series.envelopes[1].sign_class == ("POSITIVE",)

    radius, center = chebyshev_ball(fs)
    assert abs(radius - 2.0) < 1e-9
    assert np.allclose(center, [1.0, 3.0], atol=1e-9)
    assert abs(chebyshev_margin(fs) - 2.0) < 1e-9

    tau_series = static_envelopes([fs], quantities="tau")
    z_min = np.array([-2.0, 1.0])
    z_max = np.array([4.0, 5.0])
    n_tau = null_basis[:5, :]
    expected_min = x_p[:5] + np.minimum(
        n_tau * z_min[None, :], n_tau * z_max[None, :]
    ).sum(axis=1)
    expected_max = x_p[:5] + np.maximum(
        n_tau * z_min[None, :], n_tau * z_max[None, :]
    ).sum(axis=1)
    assert np.allclose(tau_series.lower_matrix[0], expected_min, atol=1e-9)
    assert np.allclose(tau_series.upper_matrix[0], expected_max, atol=1e-9)


def test_inclusion_minimal_conflict_set_and_row_provenance() -> None:
    """conflict_set() returns only the inclusion-minimal contradictory hypothesis IDs."""
    # 2D z-space with three hypotheses:
    # H-TORQUE-CAPACITY-BOX: -5 <= z_0 <= 5, -5 <= z_1 <= 5
    # H-BENIGN-SPLIT-PRIOR: z_1 >= -2 (completely compatible)
    # H-CONTRADICTORY-PRELOAD: z_0 >= 8 (contradicts z_0 <= 5 in H-TORQUE-CAPACITY-BOX)
    b_cap = PolyhedralRowBlock(
        A_ub=np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]),
        b_ub=np.array([5.0, 5.0, 5.0, 5.0]),
        hypothesis_id="H-TORQUE-CAPACITY-BOX",
    )
    b_benign = PolyhedralRowBlock(
        A_ub=np.array([[0.0, -1.0], [0.0, 1.0]]),
        b_ub=np.array([2.0, 4.0]),
        hypothesis_id="H-BENIGN-SPLIT-PRIOR",
    )
    b_conflict = PolyhedralRowBlock(
        A_ub=np.array([[-1.0, 0.0]]),
        b_ub=np.array([-8.0]),
        hypothesis_id="H-CONTRADICTORY-PRELOAD",
    )
    fs = FeasibleSet(
        x_particular=np.zeros(2),
        null_basis=np.eye(2),
        polyhedral_blocks=(b_cap, b_benign, b_conflict),
    )

    assert len(fs.row_hypothesis_ids) == fs.A_ub.shape[0] == 7
    assert all(isinstance(hid, str) and hid for hid in fs.row_hypothesis_ids)
    assert fs.is_empty()

    iis = fs.conflict_set()
    assert set(iis) == {"H-TORQUE-CAPACITY-BOX", "H-CONTRADICTORY-PRELOAD"}
    assert "H-BENIGN-SPLIT-PRIOR" not in iis


def test_cone_linearization_bound_and_lp_socp_specs() -> None:
    """Linearized friction cones enforce k >= 4, record 1 - cos(pi/k), and export LP/SOCP specs."""
    with pytest.raises(ValueError, match="cone_facets_k"):
        cone_linearization_error_bound(3)

    for k in (4, 8, 16):
        expected = 1.0 - math.cos(math.pi / k)
        assert abs(cone_linearization_error_bound(k) - expected) < 1e-15

    # 3-DOF contact force vector x = [f_t1, f_t2, f_n] parameterized directly by z in R^3
    cone = FrictionCone(
        indices=(0, 1, 2),
        frame=np.eye(3, dtype=np.float64),
        mu=0.6,
    )
    decomp = (
        np.array([0.0, 0.0, 20.0], dtype=np.float64),
        np.eye(3, dtype=np.float64),
    )
    fs = FeasibleSet.from_hypotheses(
        decomp,
        ("H-TORQUE-CAPACITY-BOX", "H-GRIP-FRICTION-CONE"),
        capacity_bounds=(
            np.array([-50.0, -50.0, -10.0]),
            np.array([50.0, 50.0, 100.0]),
        ),
        friction_cones=(cone,),
        cone_facets_k=8,
    )
    assert not fs.is_empty()
    assert all(hid for hid in fs.row_hypothesis_ids)
    assert len(fs.row_hypothesis_ids) == fs.A_ub.shape[0]

    cone_blocks = [b for b in fs.polyhedral_blocks if b.cone_facets_k is not None]
    assert len(cone_blocks) == 1
    assert cone_blocks[0].cone_facets_k == 8
    assert (
        abs(cone_blocks[0].approximation_error_bound - (1.0 - math.cos(math.pi / 8)))
        < 1e-15
    )

    lp_spec = fs.to_lp()
    assert isinstance(lp_spec, LinearProgramSpec)
    assert lp_spec.A_ub.shape == fs.A_ub.shape
    assert len(lp_spec.row_hypothesis_ids) == fs.A_ub.shape[0]

    socp_spec = fs.to_socp()
    assert isinstance(socp_spec, SecondOrderConeProgramSpec)
    assert len(socp_spec.soc_blocks) == 1
    soc_blk = socp_spec.soc_blocks[0]
    assert isinstance(soc_blk, SecondOrderConeBlock)
    assert soc_blk.hypothesis_id == "H-GRIP-FRICTION-CONE"
    # At z = [6, 8, 0], f = [6, 8, 20], ||f_t||_2 = 10 <= 0.6 * 20 = 12 -> residual = -2 <= 0
    assert abs(soc_blk.residual(np.array([6.0, 8.0, 0.0])) - (-2.0)) < 1e-12


def test_unbounded_direction_returns_nan_and_sign_classification() -> None:
    """Unbounded LP directions return NaN with status UNBOUNDED, never a large finite number."""
    # z_0 >= 2 (unbounded above), z_1 <= -3 (unbounded below), z_2 in [0, 0] (ZERO), z_3 in [-1, 2]
    a_ub = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],  # -z_0 <= -2 => z_0 >= 2
            [0.0, 1.0, 0.0, 0.0],  # z_1 <= -3
            [0.0, 0.0, -1.0, 0.0],  # z_2 >= 0
            [0.0, 0.0, 1.0, 0.0],  # z_2 <= 0
            [0.0, 0.0, 0.0, -1.0],  # z_3 >= -1
            [0.0, 0.0, 0.0, 1.0],  # z_3 <= 2
        ],
        dtype=np.float64,
    )
    b_ub = np.array([-2.0, -3.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float64)
    fs = FeasibleSet(
        x_particular=np.zeros(4),
        null_basis=np.eye(4),
        polyhedral_blocks=(
            PolyhedralRowBlock(A_ub=a_ub, b_ub=b_ub, hypothesis_id="H-TEST"),
        ),
    )
    series = static_envelopes([fs])
    e0, e1, e2, e3 = series.envelopes

    assert abs(float(e0.lower[0]) - 2.0) < 1e-9
    assert e0.lower_status == ("BOUNDED",)
    assert math.isnan(float(e0.upper[0]))
    assert e0.upper_status == ("UNBOUNDED",)
    assert e0.sign_class == ("POSITIVE",)

    assert math.isnan(float(e1.lower[0]))
    assert e1.lower_status == ("UNBOUNDED",)
    assert abs(float(e1.upper[0]) - (-3.0)) < 1e-9
    assert e1.upper_status == ("BOUNDED",)
    assert e1.sign_class == ("NEGATIVE",)

    assert abs(float(e2.lower[0])) < 1e-9
    assert abs(float(e2.upper[0])) < 1e-9
    assert e2.sign_class == ("ZERO",)

    assert e3.sign_class == ("INDETERMINATE",)
    assert classify_sign(0.5, 2.0) == "POSITIVE"
    assert classify_sign(-3.0, -0.1) == "NEGATIVE"
    assert classify_sign(-1e-12, 1e-12) == "ZERO"
    assert classify_sign(-1.0, 1.0) == "INDETERMINATE"


def test_warm_start_equivalence_across_frames() -> None:
    """Cold and warm-started static_envelopes produce identical bounds across multi-frame sequences."""
    rng = np.random.default_rng(42)
    matrix_a = np.array(
        [
            [1.0, 0.0, 0.0, 1.0, 0.5],
            [0.0, 1.0, 0.0, 0.5, -0.5],
            [0.0, 0.0, 1.0, -0.5, 1.0],
        ],
        dtype=np.float64,
    )
    feasible_sets: list[FeasibleSet] = []
    for k in range(25):
        phase = 0.1 * k
        rhs = np.array(
            [2.0 + math.sin(phase), -1.0 + math.cos(phase), 0.5 * math.sin(2 * phase)],
            dtype=np.float64,
        )
        decomp = ForceNullSpace.from_balance(matrix_a, rhs)
        tau_min = np.full(3, -12.0, dtype=np.float64) + 0.1 * rng.standard_normal(3)
        tau_max = np.full(3, 12.0, dtype=np.float64) + 0.1 * rng.standard_normal(3)
        fs = FeasibleSet.from_hypotheses(
            decomp,
            ("H-TORQUE-CAPACITY-BOX",),
            capacity_bounds=(tau_min, tau_max),
        )
        feasible_sets.append(fs)

    warm = static_envelopes(feasible_sets, warm_start=True)
    cold = static_envelopes(feasible_sets, warm_start=False)

    assert warm.warm_start_used is True
    assert cold.warm_start_used is False
    assert np.allclose(warm.lower_matrix, cold.lower_matrix, atol=1e-9)
    assert np.allclose(warm.upper_matrix, cold.upper_matrix, atol=1e-9)
    assert np.allclose(warm.chebyshev_margins, cold.chebyshev_margins, atol=1e-9)
    for eq_w, eq_c in zip(warm.envelopes, cold.envelopes, strict=True):
        assert eq_w.sign_class == eq_c.sign_class
        assert eq_w.lower_status == eq_c.lower_status
        assert eq_w.upper_status == eq_c.upper_status


def test_reaction_restriction_row_selection_nesting_and_emptiness() -> None:
    """restrict() selects exact canonical grip rows and restriction_sensitivity verifies WELD6 >= SHAFT_FREE5 >= BALL3."""
    snap_easy = _make_spatial_two_arm_snapshot(scale_moment_load=0.2)
    snap_hard = _make_spatial_two_arm_snapshot(scale_moment_load=2.5)

    for kind, expected_rows in RESTRICTION_ROW_INDICES.items():
        r_snap = restrict(snap_easy, kind)
        assert r_snap.J_c.shape == (len(expected_rows), 6)
        assert np.array_equal(r_snap.J_c, snap_easy.J_c[list(expected_rows), :])
        assert np.array_equal(r_snap.Jdot_v, snap_easy.Jdot_v[list(expected_rows)])
        assert str(r_snap.topology.closure_kind) == kind

    invalid_kind: Any = "PLANAR_POINT2"
    with pytest.raises(ValueError, match="kind"):
        restrict(snap_easy, invalid_kind)

    bad_snap = replace(
        snap_easy,
        J_c=snap_easy.J_c[:3, :],
        Jdot_v=snap_easy.Jdot_v[:3],
        topology=replace(snap_easy.topology, closure_kind="BALL3"),
    )
    with pytest.raises(ValueError, match="6"):
        restrict(bad_snap, "SHAFT_FREE5")

    # Frame 0 (snap_easy) is feasible for all 3 restrictions;
    # Frame 1 (snap_hard) is feasible for WELD6 (which can use grip moments tx, ty, tz)
    # but infeasible for BALL3 under tight torque capacity bounds |tau_i| <= 6.0!
    cap_bounds = (
        np.full(6, -6.0, dtype=np.float64),
        np.full(6, 6.0, dtype=np.float64),
    )
    report = restriction_sensitivity(
        [snap_easy, snap_hard],
        kinds=("WELD6", "SHAFT_FREE5", "BALL3"),
        capacity_bounds=cap_bounds,
    )
    assert isinstance(report, RestrictionReport)
    assert report.label == REACTION_RESTRICTION_SENSITIVITY
    assert report.nesting_holds is True
    assert report.max_nesting_violation <= 1e-8

    for kind in ("WELD6", "SHAFT_FREE5", "BALL3"):
        for fr in report.frame_results_by_kind[kind]:
            assert isinstance(fr, RestrictedFrameResult)
            assert fr.label == REACTION_RESTRICTION_SENSITIVITY

    # Verify strict nesting on Frame 0 (all 3 kinds feasible):
    # WELD6 lower <= SHAFT_FREE5 lower <= BALL3 lower <= BALL3 upper <= SHAFT_FREE5 upper <= WELD6 upper
    w6_lo = report.envelopes_by_kind["WELD6"].lower_matrix[0]
    sf5_lo = report.envelopes_by_kind["SHAFT_FREE5"].lower_matrix[0]
    b3_lo = report.envelopes_by_kind["BALL3"].lower_matrix[0]
    w6_hi = report.envelopes_by_kind["WELD6"].upper_matrix[0]
    sf5_hi = report.envelopes_by_kind["SHAFT_FREE5"].upper_matrix[0]
    b3_hi = report.envelopes_by_kind["BALL3"].upper_matrix[0]

    assert np.all(w6_lo <= sf5_lo + 1e-8)
    assert np.all(sf5_lo <= b3_lo + 1e-8)
    assert np.all(b3_hi <= sf5_hi + 1e-8)
    assert np.all(sf5_hi <= w6_hi + 1e-8)

    # Verify Frame 1 is feasible for WELD6 but empty for BALL3 with non-empty conflict set
    assert 1 not in report.empty_frames_by_kind["WELD6"]
    assert 1 in report.empty_frames_by_kind["BALL3"]
    ball3_conflict = report.conflict_sets_by_kind["BALL3"][1]
    assert len(ball3_conflict) >= 1
    assert "H-TORQUE-CAPACITY-BOX" in ball3_conflict
