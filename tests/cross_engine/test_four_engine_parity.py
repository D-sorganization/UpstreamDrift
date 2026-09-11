"""Automated Cross-Engine Parity Benchmark Suite and Tolerance Gate.

Matches CROSS_ENGINE_GOLF_EQUIVALENCE_SPEC.md (§1, §3, §5) for Issue #9969
(Parent Epic: #9964).

Validates that given the canonical multibody golfer specification and an
identical continuous polynomial torque profile theta* (6th-order Bernstein basis
across 27 canonical actuation channels), the reference Simscape Multibody
baseline and the three open-source engines (MuJoCo, Pinocchio, and Drake) produce
identical forward-dynamics trajectories satisfying the cross-engine tolerance gates:
    - Grip point RMSE vs Simscape baseline: < 5.0 mm
    - Clubhead point RMSE vs Simscape baseline: < 10.0 mm

All native physics engines are dynamically guarded via is_mujoco_available(),
is_pinocchio_available(), and is_drake_available() so tests gracefully skip or
verify contract compliance when compiled shared objects are not installed on host.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from src.engines.physics_engines.mujoco.python.simulate_with_coefficients import (
    DEFAULT_GOLFER_XML as MUJOCO_GOLFER_XML,
)
from src.engines.physics_engines.pinocchio.python.simulate_with_coefficients import (
    DEFAULT_GOLFER_URDF as PINOCCHIO_GOLFER_URDF,
)
from src.engines.physics_engines.drake.python.simulate_with_coefficients import (
    DEFAULT_GOLFER_URDF as DRAKE_GOLFER_URDF,
)
from src.shared.python.core.contracts.decorators import postcondition, precondition
from src.shared.python.engine_core.engine_availability import (
    is_drake_available,
    is_mujoco_available,
    is_pinocchio_available,
)

logger = logging.getLogger(__name__)

# Suite marker contract: unit and cross_engine
pytestmark = [pytest.mark.unit, pytest.mark.cross_engine]

# Parity tolerance thresholds from CROSS_ENGINE_GOLF_EQUIVALENCE_SPEC.md §5.1
GRIP_RMSE_TOL_MM: float = 5.0
CLUBHEAD_RMSE_TOL_MM: float = 10.0
BENCHMARK_HORIZON_S: float = 0.02
BENCHMARK_TIMESTEP_S: float = 0.001
BENCHMARK_SAMPLE_RATE_HZ: float = 1000.0
EXPECTED_SAMPLE_COUNT: int = 21

CANONICAL_BASELINE_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cross_engine_parity"
    / "simscape_canonical_baseline.npz"
)


# --------------------------------------------------------------------------- #
# Data structures & Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParityGateResult:
    """Evaluation result against cross-engine parity tolerances."""

    engine: str
    grip_rmse_mm: float
    clubhead_rmse_mm: float
    grip_passed: bool
    clubhead_passed: bool
    passed: bool
    n_samples: int
    horizon_s: float
    details: str


@dataclass(frozen=True)
class SimscapeGoldenBaseline:
    """Canonical ground-truth Simscape Multibody forward-dynamics baseline."""

    time: NDArray[np.float64]
    grip: NDArray[np.float64]
    clubhead: NDArray[np.float64]
    grip_quat: NDArray[np.float64]
    club_quat: NDArray[np.float64]
    theta_star: NDArray[np.float64]


# --------------------------------------------------------------------------- #
# Benchmark Torque & Baseline Loaders
# --------------------------------------------------------------------------- #


def get_certified_benchmark_theta() -> NDArray[np.float64]:
    """Return certified 6th-order continuous Bernstein torque polynomial theta*.

    Shape: (27 * 7,) = 189 coefficients across 27 canonical actuation channels.
    Coefficients represent bounded, smooth control inputs within spec amplitudes.
    """
    n_channels = 27
    coeffs_per_channel = 7
    theta = np.zeros(n_channels * coeffs_per_channel, dtype=np.float64)

    # Gentle, certified torque on lumbar lateral/flexion and thorax axial channels
    # Channel 6: SpineInputX, Channel 7: SpineInputY, Channel 8: TorsoInput
    theta[6 * coeffs_per_channel + 0] = 5.0  # c0
    theta[6 * coeffs_per_channel + 6] = 2.5  # c6
    theta[7 * coeffs_per_channel + 0] = -3.0
    theta[7 * coeffs_per_channel + 6] = -1.5
    theta[8 * coeffs_per_channel + 0] = 8.0
    theta[8 * coeffs_per_channel + 6] = 4.0

    return theta


def load_simscape_golden_baseline() -> SimscapeGoldenBaseline:
    """Load canonical Simscape ground-truth forward-dynamics baseline."""
    if not CANONICAL_BASELINE_PATH.is_file():
        msg = (
            f"Simscape canonical baseline fixture missing at {CANONICAL_BASELINE_PATH}"
        )
        raise FileNotFoundError(msg)

    with np.load(CANONICAL_BASELINE_PATH, allow_pickle=False) as data:
        time = np.asarray(data["time"], dtype=np.float64)
        grip = np.asarray(data["grip"], dtype=np.float64)
        clubhead = np.asarray(data["clubhead"], dtype=np.float64)
        grip_quat = np.asarray(data["grip_quat"], dtype=np.float64)
        club_quat = np.asarray(data["club_quat"], dtype=np.float64)
        theta_star = np.asarray(data["theta_star"], dtype=np.float64)

    return SimscapeGoldenBaseline(
        time=time,
        grip=grip,
        clubhead=clubhead,
        grip_quat=grip_quat,
        club_quat=club_quat,
        theta_star=theta_star,
    )


# --------------------------------------------------------------------------- #
# Metric & Tolerance Evaluation Helpers (DbC)
# --------------------------------------------------------------------------- #


@precondition(
    lambda simulated, reference: bool(simulated.shape == reference.shape),
    "simulated and reference trajectories must have identical shape",
)
@precondition(
    lambda simulated, reference: bool(
        simulated.ndim == 2 and simulated.shape[1] == 3 and simulated.shape[0] > 0
    ),
    "trajectories must be non-empty (N, 3) arrays",
)
@postcondition(
    lambda result: bool(np.isfinite(result) and result >= 0.0),
    "RMSE must be non-negative and finite",
)
def compute_trajectory_rmse(
    simulated: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> float:
    """Compute trajectory Euclidean point RMSE in millimeters.

    RMSE = sqrt( mean_t( ||p_sim(t) - p_ref(t)||^2 ) ) * 1000.0 (mm)
    """
    diff = np.asarray(simulated, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    sq_dist = np.einsum("ij,ij->i", diff, diff)
    mse = float(np.mean(sq_dist))
    return float(np.sqrt(mse) * 1000.0)


@precondition(
    lambda engine_name, sim_grip, ref_grip, sim_clubhead, ref_clubhead, *args, **kwargs: (
        bool(
            len(engine_name) > 0
            and sim_grip.shape == ref_grip.shape
            and sim_clubhead.shape == ref_clubhead.shape
        )
    ),
    "inputs must have valid engine name and matching trajectory shapes",
)
def verify_parity_tolerances(
    engine_name: str,
    sim_grip: NDArray[np.float64],
    ref_grip: NDArray[np.float64],
    sim_clubhead: NDArray[np.float64],
    ref_clubhead: NDArray[np.float64],
    grip_tol_mm: float = GRIP_RMSE_TOL_MM,
    clubhead_tol_mm: float = CLUBHEAD_RMSE_TOL_MM,
    horizon_s: float = BENCHMARK_HORIZON_S,
) -> ParityGateResult:
    """Verify trajectory tolerances for an engine against the Simscape baseline."""
    grip_rmse = compute_trajectory_rmse(sim_grip, ref_grip)
    clubhead_rmse = compute_trajectory_rmse(sim_clubhead, ref_clubhead)

    grip_ok = bool(grip_rmse < grip_tol_mm)
    clubhead_ok = bool(clubhead_rmse < clubhead_tol_mm)
    all_ok = grip_ok and clubhead_ok

    details = (
        f"[{engine_name}] Grip RMSE: {grip_rmse:.3f} mm (limit: {grip_tol_mm:.1f} mm, {'PASS' if grip_ok else 'FAIL'}) | "
        f"Clubhead RMSE: {clubhead_rmse:.3f} mm (limit: {clubhead_tol_mm:.1f} mm, {'PASS' if clubhead_ok else 'FAIL'})"
    )

    logger.info("Parity evaluation: %s", details)

    return ParityGateResult(
        engine=engine_name,
        grip_rmse_mm=grip_rmse,
        clubhead_rmse_mm=clubhead_rmse,
        grip_passed=grip_ok,
        clubhead_passed=clubhead_ok,
        passed=all_ok,
        n_samples=int(sim_grip.shape[0]),
        horizon_s=float(horizon_s),
        details=details,
    )


# --------------------------------------------------------------------------- #
# Engine Forward-Dynamics Execution Handlers
# --------------------------------------------------------------------------- #


def run_mujoco_parity_rollout(
    theta_star: NDArray[np.float64],
    horizon_s: float = BENCHMARK_HORIZON_S,
    dt_s: float = BENCHMARK_TIMESTEP_S,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Execute MuJoCo forward-dynamics rollout and return (grip, clubhead) trajectories."""
    import mujoco  # noqa: PLC0415
    from src.engines.physics_engines.mujoco.python.simulate_with_coefficients import (  # noqa: PLC0415
        SimOptions,
        get_mujoco_canonical_joint_map,
        simulate_with_coefficients,
    )

    if not MUJOCO_GOLFER_XML.exists():
        msg = f"MuJoCo golfer model not found at {MUJOCO_GOLFER_XML}"
        raise FileNotFoundError(msg)

    model = mujoco.MjModel.from_xml_path(str(MUJOCO_GOLFER_XML))
    nu = int(model.nu)
    jmap = get_mujoco_canonical_joint_map(model)

    # Map canonical 27 channels (x7) into native MuJoCo actuator coefficients
    theta_mujoco = np.zeros(nu * 7, dtype=np.float64)
    coeffs_canonical = theta_star.reshape(-1, 7)
    for c_idx, act_idx in enumerate(jmap.engine_dof_indices):
        if 0 <= act_idx < nu:
            sign = jmap.sign_flips[c_idx]
            theta_mujoco[act_idx * 7 : (act_idx + 1) * 7] = (
                sign * coeffs_canonical[c_idx]
            )

    opts = SimOptions(
        xml_path=MUJOCO_GOLFER_XML,
        T_s=horizon_s,
        dt=dt_s,
        output_rate_hz=1000.0,
        compute_energy=True,
    )
    sim_out = simulate_with_coefficients(theta_mujoco, opts)

    return np.asarray(sim_out.grip), np.asarray(sim_out.clubhead)


def run_pinocchio_parity_rollout(
    theta_star: NDArray[np.float64],
    horizon_s: float = BENCHMARK_HORIZON_S,
    dt_s: float = BENCHMARK_TIMESTEP_S,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Execute Pinocchio forward-dynamics rollout and return (grip, clubhead) trajectories."""
    import pinocchio as pin  # noqa: PLC0415
    from src.engines.physics_engines.pinocchio.python.simulate_with_coefficients import (  # noqa: PLC0415
        SimOptions,
        get_pinocchio_canonical_joint_map,
        simulate_with_coefficients,
    )

    if not PINOCCHIO_GOLFER_URDF.exists():
        msg = f"Pinocchio golfer URDF not found at {PINOCCHIO_GOLFER_URDF}"
        raise FileNotFoundError(msg)

    model = pin.buildModelFromUrdf(str(PINOCCHIO_GOLFER_URDF))
    nv = int(model.nv)
    jmap = get_pinocchio_canonical_joint_map(model)

    theta_pin = np.zeros(nv * 7, dtype=np.float64)
    coeffs_canonical = theta_star.reshape(-1, 7)
    for c_idx, dof_idx in enumerate(jmap.engine_dof_indices):
        if 0 <= dof_idx < nv:
            sign = jmap.sign_flips[c_idx]
            theta_pin[dof_idx * 7 : (dof_idx + 1) * 7] = sign * coeffs_canonical[c_idx]

    opts = SimOptions(
        urdf_path=PINOCCHIO_GOLFER_URDF,
        t_final=horizon_s,
        dt=dt_s,
        basis="bernstein",
        compute_energy=True,
    )
    sim_out = simulate_with_coefficients(theta_pin, opts)

    return np.asarray(sim_out.grip), np.asarray(sim_out.clubhead)


def run_drake_parity_rollout(
    theta_star: NDArray[np.float64],
    horizon_s: float = BENCHMARK_HORIZON_S,
    dt_s: float = BENCHMARK_TIMESTEP_S,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Execute Drake forward-dynamics rollout and return (grip, clubhead) trajectories."""
    from pydrake.multibody.plant import MultibodyPlant  # noqa: PLC0415
    from src.engines.physics_engines.drake.python.simulate_with_coefficients import (  # noqa: PLC0415
        SimOptions,
        get_drake_canonical_joint_map,
        simulate_with_coefficients,
    )

    if not DRAKE_GOLFER_URDF.exists():
        msg = f"Drake golfer URDF not found at {DRAKE_GOLFER_URDF}"
        raise FileNotFoundError(msg)

    plant = MultibodyPlant(dt_s)
    from src.engines.physics_engines.drake.python.motion_matching.simulate import (  # noqa: PLC0415
        load_humanoid_into_plant,
    )

    load_humanoid_into_plant(plant, DRAKE_GOLFER_URDF)
    plant.Finalize()
    n_act = int(plant.num_actuators())
    jmap = get_drake_canonical_joint_map(plant)

    theta_drake = np.zeros(n_act * 7, dtype=np.float64)
    coeffs_canonical = theta_star.reshape(-1, 7)
    for c_idx, act_idx in enumerate(jmap.engine_dof_indices):
        if 0 <= act_idx < n_act:
            sign = jmap.sign_flips[c_idx]
            theta_drake[act_idx * 7 : (act_idx + 1) * 7] = (
                sign * coeffs_canonical[c_idx]
            )

    opts = SimOptions(
        urdf_path=DRAKE_GOLFER_URDF,
        simulation_time_s=horizon_s,
        time_step_s=dt_s,
        sample_rate_hz=1000.0,
        basis="bernstein",
    )
    sim_out = simulate_with_coefficients(theta_drake, opts)

    return np.asarray(sim_out.grip), np.asarray(sim_out.clubhead)


# --------------------------------------------------------------------------- #
# Unit & Integration Tests
# --------------------------------------------------------------------------- #


def test_simscape_golden_baseline_integrity() -> None:
    """Simscape golden baseline must be complete, monotonic, and physically valid."""
    baseline = load_simscape_golden_baseline()

    assert baseline.time.ndim == 1
    assert baseline.time.shape[0] == EXPECTED_SAMPLE_COUNT
    assert baseline.time[0] == pytest.approx(0.0)
    assert baseline.time[-1] == pytest.approx(BENCHMARK_HORIZON_S)
    assert np.all(np.diff(baseline.time) > 0.0), (
        "Time stamps must be strictly monotonic"
    )

    # Kinematic trajectories
    assert baseline.grip.shape == (EXPECTED_SAMPLE_COUNT, 3)
    assert baseline.clubhead.shape == (EXPECTED_SAMPLE_COUNT, 3)
    assert np.all(np.isfinite(baseline.grip))
    assert np.all(np.isfinite(baseline.clubhead))

    # Shaft distance consistency: grip to clubhead distance ~ 0.55 m in canonical model
    initial_shaft_length = float(
        np.linalg.norm(baseline.clubhead[0] - baseline.grip[0])
    )
    assert initial_shaft_length == pytest.approx(0.55, abs=0.01)

    # Quaternions
    assert baseline.grip_quat.shape == (EXPECTED_SAMPLE_COUNT, 4)
    assert baseline.club_quat.shape == (EXPECTED_SAMPLE_COUNT, 4)
    grip_quat_norms = np.linalg.norm(baseline.grip_quat, axis=1)
    np.testing.assert_allclose(grip_quat_norms, 1.0, atol=1e-6)


def test_mujoco_parity_vs_simscape() -> None:
    """MuJoCo forward dynamics matches Simscape baseline within parity tolerances."""
    if not is_mujoco_available():
        pytest.skip("MuJoCo runtime is not installed on host")

    baseline = load_simscape_golden_baseline()
    theta_star = get_certified_benchmark_theta()

    sim_grip, sim_clubhead = run_mujoco_parity_rollout(theta_star)

    res = verify_parity_tolerances(
        engine_name="MuJoCo",
        sim_grip=sim_grip,
        ref_grip=baseline.grip,
        sim_clubhead=sim_clubhead,
        ref_clubhead=baseline.clubhead,
    )

    assert res.grip_passed, (
        f"MuJoCo grip RMSE {res.grip_rmse_mm:.3f} mm exceeded {GRIP_RMSE_TOL_MM} mm gate"
    )
    assert res.clubhead_passed, (
        f"MuJoCo clubhead RMSE {res.clubhead_rmse_mm:.3f} mm exceeded {CLUBHEAD_RMSE_TOL_MM} mm gate"
    )
    assert res.passed


def test_pinocchio_parity_vs_simscape() -> None:
    """Pinocchio forward dynamics matches Simscape baseline within parity tolerances."""
    if not is_pinocchio_available():
        pytest.skip("Pinocchio C++ runtime is not installed on host")

    baseline = load_simscape_golden_baseline()
    theta_star = get_certified_benchmark_theta()

    sim_grip, sim_clubhead = run_pinocchio_parity_rollout(theta_star)

    res = verify_parity_tolerances(
        engine_name="Pinocchio",
        sim_grip=sim_grip,
        ref_grip=baseline.grip,
        sim_clubhead=sim_clubhead,
        ref_clubhead=baseline.clubhead,
    )

    assert res.grip_passed, (
        f"Pinocchio grip RMSE {res.grip_rmse_mm:.3f} mm exceeded {GRIP_RMSE_TOL_MM} mm gate"
    )
    assert res.clubhead_passed, (
        f"Pinocchio clubhead RMSE {res.clubhead_rmse_mm:.3f} mm exceeded {CLUBHEAD_RMSE_TOL_MM} mm gate"
    )
    assert res.passed


def test_drake_parity_vs_simscape() -> None:
    """Drake forward dynamics matches Simscape baseline within parity tolerances."""
    if not is_drake_available():
        pytest.skip("Drake C++ runtime (pydrake) is not installed on host")

    baseline = load_simscape_golden_baseline()
    theta_star = get_certified_benchmark_theta()

    sim_grip, sim_clubhead = run_drake_parity_rollout(theta_star)

    res = verify_parity_tolerances(
        engine_name="Drake",
        sim_grip=sim_grip,
        ref_grip=baseline.grip,
        sim_clubhead=sim_clubhead,
        ref_clubhead=baseline.clubhead,
    )

    assert res.grip_passed, (
        f"Drake grip RMSE {res.grip_rmse_mm:.3f} mm exceeded {GRIP_RMSE_TOL_MM} mm gate"
    )
    assert res.clubhead_passed, (
        f"Drake clubhead RMSE {res.clubhead_rmse_mm:.3f} mm exceeded {CLUBHEAD_RMSE_TOL_MM} mm gate"
    )
    assert res.passed


def test_four_engine_cross_engine_matrix() -> None:
    """Four-engine parity summary matrix across all available engines and Simscape."""
    baseline = load_simscape_golden_baseline()
    theta_star = get_certified_benchmark_theta()

    # Track evaluated results
    results: list[ParityGateResult] = []

    # 1. Simscape Oracle self-consistency
    results.append(
        verify_parity_tolerances(
            engine_name="Simscape Oracle",
            sim_grip=baseline.grip,
            ref_grip=baseline.grip,
            sim_clubhead=baseline.clubhead,
            ref_clubhead=baseline.clubhead,
        )
    )

    # 2. MuJoCo
    if is_mujoco_available():
        mj_grip, mj_clubhead = run_mujoco_parity_rollout(theta_star)
        results.append(
            verify_parity_tolerances(
                engine_name="MuJoCo",
                sim_grip=mj_grip,
                ref_grip=baseline.grip,
                sim_clubhead=mj_clubhead,
                ref_clubhead=baseline.clubhead,
            )
        )

    # 3. Pinocchio
    if is_pinocchio_available():
        pin_grip, pin_clubhead = run_pinocchio_parity_rollout(theta_star)
        results.append(
            verify_parity_tolerances(
                engine_name="Pinocchio",
                sim_grip=pin_grip,
                ref_grip=baseline.grip,
                sim_clubhead=pin_clubhead,
                ref_clubhead=baseline.clubhead,
            )
        )

    # 4. Drake
    if is_drake_available():
        drake_grip, drake_clubhead = run_drake_parity_rollout(theta_star)
        results.append(
            verify_parity_tolerances(
                engine_name="Drake",
                sim_grip=drake_grip,
                ref_grip=baseline.grip,
                sim_clubhead=drake_clubhead,
                ref_clubhead=baseline.clubhead,
            )
        )

    # Render Markdown table report
    report_lines = [
        "# Cross-Engine Parity Benchmark Report",
        "",
        "| Engine | Grip RMSE (mm) | Clubhead RMSE (mm) | Grip Gate (< 5 mm) | Clubhead Gate (< 10 mm) | Verdict |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        report_lines.append(
            f"| {r.engine} | {r.grip_rmse_mm:.3f} | {r.clubhead_rmse_mm:.3f} | "
            f"{'PASS' if r.grip_passed else 'FAIL'} | {'PASS' if r.clubhead_passed else 'FAIL'} | "
            f"{'PASS' if r.passed else 'FAIL'} |"
        )
    report_text = "\n".join(report_lines)
    logger.info("Four-Engine Parity Matrix:\n%s", report_text)

    # All evaluated engines must strictly satisfy tolerance gates
    assert all(r.passed for r in results)


def test_parity_tolerance_gate_boundary_conditions() -> None:
    """Tolerance gate strictly passes compliant errors and rejects breaches."""
    baseline = load_simscape_golden_baseline()

    # Base trajectories
    base_grip = baseline.grip.copy()
    base_clubhead = baseline.clubhead.copy()

    # Compliant perturbation: 4.8 mm grip, 9.5 mm clubhead
    grip_shift_4_8mm = np.array([0.0048, 0.0, 0.0])
    clubhead_shift_9_5mm = np.array([0.0, 0.0095, 0.0])

    res_ok = verify_parity_tolerances(
        engine_name="CompliantEngine",
        sim_grip=base_grip + grip_shift_4_8mm,
        ref_grip=base_grip,
        sim_clubhead=base_clubhead + clubhead_shift_9_5mm,
        ref_clubhead=base_clubhead,
    )
    assert res_ok.grip_passed
    assert res_ok.clubhead_passed
    assert res_ok.passed

    # Breaching grip perturbation: 5.2 mm grip (> 5.0 mm limit)
    res_grip_fail = verify_parity_tolerances(
        engine_name="GripBreachEngine",
        sim_grip=base_grip + np.array([0.0052, 0.0, 0.0]),
        ref_grip=base_grip,
        sim_clubhead=base_clubhead,
        ref_clubhead=base_clubhead,
    )
    assert not res_grip_fail.grip_passed
    assert res_grip_fail.clubhead_passed
    assert not res_grip_fail.passed

    # Breaching clubhead perturbation: 10.5 mm clubhead (> 10.0 mm limit)
    res_head_fail = verify_parity_tolerances(
        engine_name="ClubheadBreachEngine",
        sim_grip=base_grip,
        ref_grip=base_grip,
        sim_clubhead=base_clubhead + np.array([0.0, 0.0, 0.0105]),
        ref_clubhead=base_clubhead,
    )
    assert res_head_fail.grip_passed
    assert not res_head_fail.clubhead_passed
    assert not res_head_fail.passed


def test_engine_dynamic_availability_guards() -> None:
    """Engine availability guards execute without exceptions and return booleans."""
    mj = is_mujoco_available()
    pin = is_pinocchio_available()
    drake = is_drake_available()

    assert isinstance(mj, bool)
    assert isinstance(pin, bool)
    assert isinstance(drake, bool)
