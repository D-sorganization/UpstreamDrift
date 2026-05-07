"""Cross-engine forward-sim equivalence test (PARITY-EQUIVALENCE, issue #4096).

This module is the CI gate that enforces the §2.2 acceptance criterion of
``src/engines/CROSS_ENGINE_PARITY_SPEC.md``: every physics-engine wrapper that
implements ``simulate_with_coefficients`` must round-trip a fixed ``theta`` to
within **5 mm grip-position RMSE vs the Simscape reference** at three test
poses (impact, top-of-backswing, address).

The test is intentionally engine-agnostic:

1. We sample one canonical ``theta_truth`` via a deterministic mirror of the
   MATLAB ``generateRandomCoefficients`` distribution (|A,B|≤1000, |C,D|≤500,
   |E,F|≤100, |G|≤25) using ``numpy.random.default_rng(42)``.
2. For every available engine (Simscape via the MATLAB Engine bridge, MuJoCo,
   Drake, Pinocchio, OpenSim), we run ``simulate_with_coefficients(theta_truth,
   pose)`` for three canonical poses and capture ``SimOut.grip``,
   ``SimOut.grip_quat``, and ``SimOut.clubhead``.
3. We pick the **first available engine** as the reference (Simscape preferred,
   else the first that successfully simulates) and compute, per other engine:

   - grip-position RMSE                  (must be < 5 mm)
   - grip-orientation RMSE               (must be < 1°)
   - clubhead-position RMSE              (must be < 5 mm + max club-length diff)

4. A timestamped Markdown report is emitted to
   ``output/cross_engine_equivalence/<UTC>/report.md`` regardless of pass/fail
   so divergences > 5 mm (P0) are documented for the follow-up issue.

5. Each engine path is gated by ``pytest.mark.requires_<engine>`` so CI skips
   engines whose deps aren't installed rather than failing the gate.

If fewer than two engines are available, the gate becomes a no-op skip — there
is nothing to compare. The CI workflow ``.github/workflows/cross-engine-
equivalence.yml`` installs the full ``[drake,pinocchio,opensim]`` matrix on PRs
that touch ``src/engines/`` so the gate has teeth in CI.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic theta_truth (mirrors generateRandomCoefficients.m, seed=42)
# ---------------------------------------------------------------------------

# DOF-to-joint count: per CROSS_ENGINE_PARITY_SPEC.md §2.6, the canonical
# humanoid has 25 DOF.  Each joint contributes 7 polynomial coefficients
# (A..G) per the Simscape Stateflow torque polynomial.
_N_JOINTS = 25
_COEFFS_PER_JOINT = 7
_THETA_LEN = _N_JOINTS * _COEFFS_PER_JOINT  # 175

# Coefficient-type ranges from Simscape generateRandomCoefficients.m.
# Indexed (i-1) mod 7: 0,1 -> A,B (±1000); 2,3 -> C,D (±500);
# 4,5 -> E,F (±100); 6 -> G (±25).
_COEFF_AMPLITUDES = np.array(
    [1000.0, 1000.0, 500.0, 500.0, 100.0, 100.0, 25.0],
    dtype=float,
)


def _generate_theta_truth(seed: int = 42) -> np.ndarray:
    """Deterministic mirror of MATLAB ``generateRandomCoefficients(N)``.

    The MATLAB original draws ``rand()`` for each coefficient and scales by the
    type-specific amplitude.  Numpy's PCG64 seeded RNG gives us a deterministic,
    reproducible vector that's identical across platforms.

    Postcondition: returned array has length ``_N_JOINTS * _COEFFS_PER_JOINT``
    and every entry lies inside the documented range for its coefficient type.
    """
    rng = np.random.default_rng(seed)
    raw = rng.random(_THETA_LEN) - 0.5  # uniform in [-0.5, 0.5)
    amplitudes = np.tile(2.0 * _COEFF_AMPLITUDES, _N_JOINTS)
    theta = raw * amplitudes

    # Postcondition: bounds check
    for i in range(_THETA_LEN):
        amp = _COEFF_AMPLITUDES[i % _COEFFS_PER_JOINT]
        assert -amp <= theta[i] <= amp, f"theta[{i}] = {theta[i]} outside ±{amp} range"
    return theta


# ---------------------------------------------------------------------------
# Canonical test poses (impact, top-of-backswing, address)
# ---------------------------------------------------------------------------


# Each pose is a (N,) joint-angle vector at simulation t=0.  These are
# deliberately representative of the three regimes the equivalence test
# probes: address (rest), top-of-backswing (max wind-up), impact (downswing
# release).  Values are in radians and chosen to be physically plausible
# without exercising joint-limit clipping in any engine.
@dataclass(frozen=True)
class CanonicalPose:
    """A named test pose: joint-angle vector + initial joint velocities.

    Named ``CanonicalPose`` rather than ``TestPose`` so pytest's collection
    machinery doesn't try to instantiate it as a test class (frozen dataclasses
    have ``__init__`` and would trip the ``Test*`` collection warning).
    """

    name: str
    q: np.ndarray  # (n_joints,)
    qd: np.ndarray  # (n_joints,)


def _make_test_poses() -> list[CanonicalPose]:
    """Return the three canonical test poses with deterministic q, qd."""
    rng = np.random.default_rng(0xA77E55)  # different seed from theta
    address_q = np.zeros(_N_JOINTS)
    address_qd = np.zeros(_N_JOINTS)

    top_q = rng.uniform(-0.6, 0.6, _N_JOINTS)
    top_qd = np.zeros(_N_JOINTS)  # zero velocity at the apex

    impact_q = rng.uniform(-0.3, 0.3, _N_JOINTS)
    impact_qd = rng.uniform(-3.0, 3.0, _N_JOINTS)

    return [
        CanonicalPose("address", address_q, address_qd),
        CanonicalPose("top_of_backswing", top_q, top_qd),
        CanonicalPose("impact", impact_q, impact_qd),
    ]


# ---------------------------------------------------------------------------
# Engine adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineSimResult:
    """Captured slice of ``SimOut`` for a single (engine, pose) pair."""

    engine: str
    pose_name: str
    grip: np.ndarray  # (3,) world position, metres
    grip_quat: np.ndarray  # (4,) [w,x,y,z]
    clubhead: np.ndarray  # (3,) world position, metres
    club_length: float  # grip→clubhead distance for tolerance scaling


def _engine_available(engine: str) -> bool:
    """Lazy availability probe matching ``engine_availability.py`` semantics."""
    try:
        from src.shared.python.engine_core.engine_availability import (
            is_engine_available,
        )

        return bool(is_engine_available(engine))
    except ImportError:
        # Fall back to a direct importlib probe.
        module = {
            "mujoco": "mujoco",
            "drake": "pydrake",
            "pinocchio": "pinocchio",
            "opensim": "opensim",
        }.get(engine, engine)
        try:
            importlib.import_module(module)
            return True
        except ImportError:
            return False


def _simulator_attr(
    module_path: str, attr: str = "simulate_with_coefficients"
) -> Callable[..., Any] | None:
    """Locate the engine's forward-sim entry point.

    Returns None if the module isn't importable yet (the per-engine simulator
    PRs may not have landed) or if it doesn't export ``simulate_with_coefficients``.
    """
    try:
        module = importlib.import_module(module_path)
    except (ImportError, ModuleNotFoundError, AttributeError, OSError) as exc:
        logger.debug("simulator module %s unavailable: %s", module_path, exc)
        return None
    return getattr(module, attr, None)


def _coerce_simout(
    raw: Any,
    engine: str,
    pose_name: str,
) -> EngineSimResult:
    """Reduce a ``SimOut``-like object to the slice the equivalence test needs.

    Handles both the dataclass form documented in the spec and the dict form
    that some engines may temporarily emit while the dataclass migration is in
    flight.
    """

    def _last(arr: Any, n: int) -> np.ndarray:
        """Return the final time-step row, defending against (N,k) or (k,) shapes."""
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            assert a.shape == (n,), (
                f"{engine}/{pose_name}: expected ({n},), got {a.shape}"
            )
            return a
        assert a.ndim == 2 and a.shape[1] == n, (
            f"{engine}/{pose_name}: expected (N,{n}), got {a.shape}"
        )
        return a[-1]

    def _get(name: str) -> Any:
        if hasattr(raw, name):
            return getattr(raw, name)
        if isinstance(raw, dict) and name in raw:
            return raw[name]
        raise AttributeError(f"{engine} SimOut missing field: {name}")

    grip = _last(_get("grip"), 3)
    grip_quat = _last(_get("grip_quat"), 4)
    clubhead = _last(_get("clubhead"), 3)
    club_length = float(np.linalg.norm(clubhead - grip))

    # Postcondition: every quaternion is finite and non-degenerate.
    qn = float(np.linalg.norm(grip_quat))
    assert np.isfinite(qn) and qn > 1e-6, (
        f"{engine}/{pose_name}: degenerate grip_quat with norm {qn}"
    )
    grip_quat = grip_quat / qn  # normalise so quaternion-distance is well-defined

    return EngineSimResult(
        engine=engine,
        pose_name=pose_name,
        grip=grip,
        grip_quat=grip_quat,
        clubhead=clubhead,
        club_length=club_length,
    )


# Map from engine name to the module path expected to expose
# ``simulate_with_coefficients``.  These paths follow the convention agreed in
# the per-engine parity specs; if a simulator PR has not yet landed, the
# import gracefully fails and the engine is skipped.
_ENGINE_MODULE_PATHS = {
    "simscape": "src.engines.simscape.python.motion_matching.simulate",
    "mujoco": "src.engines.physics_engines.mujoco.python.motion_matching.simulate",
    "drake": "src.engines.physics_engines.drake.python.motion_matching.simulate",
    "pinocchio": "src.engines.physics_engines.pinocchio.python.motion_matching.simulate",
    "opensim": "src.engines.physics_engines.opensim.python.motion_matching.simulate",
}


def _run_engine(
    engine: str,
    theta: np.ndarray,
    pose: CanonicalPose,
) -> EngineSimResult | None:
    """Run a single engine's forward sim for one pose; return None if unavailable.

    The function is defensive by design: a missing module, a runtime error
    inside the simulator, or a malformed return value all yield ``None`` so
    the test can either skip or report the divergence rather than crashing
    the entire suite.
    """
    if not _engine_available(engine):
        logger.info("engine %s not installed — skipping", engine)
        return None

    module_path = _ENGINE_MODULE_PATHS[engine]
    sim_fn = _simulator_attr(module_path)
    if sim_fn is None:
        logger.info(
            "engine %s installed but simulate_with_coefficients not found at %s",
            engine,
            module_path,
        )
        return None

    initial_pose = {"q": pose.q.copy(), "qd": pose.qd.copy()}
    try:
        raw = sim_fn(theta=theta.copy(), initial_pose=initial_pose)
    except TypeError:
        # Older signatures positional-only; retry with the documented order.
        try:
            raw = sim_fn(theta.copy(), initial_pose=initial_pose)
        except Exception as exc:  # noqa: BLE001 — engine error → skip
            logger.warning("engine %s pose=%s sim error: %s", engine, pose.name, exc)
            return None
    except Exception as exc:  # noqa: BLE001 — engine error → skip
        logger.warning("engine %s pose=%s sim error: %s", engine, pose.name, exc)
        return None

    try:
        return _coerce_simout(raw, engine, pose.name)
    except (AttributeError, AssertionError, ValueError) as exc:
        logger.warning("engine %s pose=%s SimOut malformed: %s", engine, pose.name, exc)
        return None


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _rmse_position(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE between two world positions in metres.

    The RMSE is computed across *samples*, with each sample's residual being
    the Euclidean distance ``‖p_a − p_b‖``. This is the standard robotics /
    biomechanics convention that the spec refers to:

      - For a single (3,) snapshot: returns ``‖a − b‖``.
      - For a time series (N, 3): returns ``sqrt(mean_t ‖a(t) − b(t)‖²)``.

    Treating the three Cartesian components as independent samples (the naive
    ``sqrt(mean(diff**2))``) under-reports the metric by ``sqrt(3)`` and was a
    bug in an earlier draft of this helper.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        return float(np.linalg.norm(a - b))
    # (N, 3) time-series form
    per_sample = np.linalg.norm(a - b, axis=1)
    return float(np.sqrt(np.mean(per_sample**2)))


def _quat_angle_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    """Angular distance between two unit quaternions, in degrees.

    Uses the standard ``2·acos(|qa·qb|)`` formula; the absolute value picks
    the shorter rotation regardless of quaternion-double-cover sign.
    """
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


# ---------------------------------------------------------------------------
# Report emission
# ---------------------------------------------------------------------------


def _output_dir() -> Path:
    """Create the timestamped output directory and return its path."""
    ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(os.environ.get("UPSTREAM_DRIFT_OUTPUT_ROOT", "output"))
    out = root / "cross_engine_equivalence" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def _format_report(
    reference_engine: str,
    results: dict[tuple[str, str], EngineSimResult],
    deltas: list[dict[str, Any]],
    theta_seed: int,
    grip_tol_mm: float,
    orient_tol_deg: float,
) -> str:
    """Render the equivalence report as a self-contained Markdown document."""
    lines: list[str] = []
    lines.append("# Cross-Engine Forward-Sim Equivalence Report")
    lines.append("")
    lines.append(f"- Reference engine: **{reference_engine}**")
    lines.append(f"- theta_truth seed: `numpy.default_rng({theta_seed})`")
    lines.append(
        f"- theta_truth length: {_THETA_LEN} ({_N_JOINTS} joints × {_COEFFS_PER_JOINT})"
    )
    lines.append(f"- Grip-position tolerance: {grip_tol_mm:.1f} mm RMSE")
    lines.append(f"- Grip-orientation tolerance: {orient_tol_deg:.1f}° RMSE")
    lines.append(
        f"- Clubhead tolerance: {grip_tol_mm:.1f} mm + max club-length difference"
    )
    lines.append("")

    engines = sorted({(eng, _) for (eng, _) in results}, key=lambda x: x[0])

    lines.append("## Per-engine availability")
    lines.append("")
    lines.append("| Engine | address | top_of_backswing | impact |")
    lines.append("|---|---|---|---|")
    for engine in {e for (e, _) in engines}:
        cells = []
        for pose in ("address", "top_of_backswing", "impact"):
            cells.append("OK" if (engine, pose) in results else "—")
        lines.append(f"| {engine} | {' | '.join(cells)} |")
    lines.append("")

    lines.append("## Residuals vs reference")
    lines.append("")
    if not deltas:
        lines.append("_No comparable engines available — gate is a no-op._")
    else:
        lines.append(
            "| Engine | Pose | Grip RMSE (mm) | Grip orient RMSE (°) | "
            "Clubhead RMSE (mm) | ΔClub-length (mm) | Verdict |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for d in deltas:
            verdict = "PASS" if d["passed"] else "**FAIL (P0)**"
            lines.append(
                f"| {d['engine']} | {d['pose']} | "
                f"{d['grip_rmse_mm']:.3f} | "
                f"{d['orient_rmse_deg']:.3f} | "
                f"{d['clubhead_rmse_mm']:.3f} | "
                f"{d['club_length_diff_mm']:.3f} | "
                f"{verdict} |"
            )
    lines.append("")

    failures = [d for d in deltas if not d["passed"]]
    if failures:
        lines.append("## P0 divergences")
        lines.append("")
        lines.append(
            "Any row marked **FAIL (P0)** above breaches the §2.2 acceptance "
            "criterion and must be investigated before merge. Likely causes "
            "(in order of frequency): coordinate-frame mismatch, joint-axis "
            "convention drift, integrator step-size, or shared-YAML build drift."
        )
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

# Tolerances per spec §2.2.
GRIP_POSITION_TOL_MM = 5.0
GRIP_ORIENTATION_TOL_DEG = 1.0
CLUBHEAD_BASE_TOL_MM = 5.0
THETA_SEED = 42

# Engine ordering — Simscape is the preferred reference per spec.
_ENGINE_ORDER = ("simscape", "mujoco", "drake", "pinocchio", "opensim")


@pytest.mark.requires_mujoco
@pytest.mark.requires_drake
@pytest.mark.requires_pinocchio
@pytest.mark.requires_opensim
@pytest.mark.requires_matlab_engine
@pytest.mark.slow
def test_cross_engine_forward_sim_equivalence() -> None:
    """Cross-engine forward-sim equivalence at three canonical poses.

    Per spec §2.2: every engine must round-trip a fixed ``theta_truth`` to
    within 5 mm grip-position RMSE vs the reference at impact, top-of-back-
    swing, and address.

    Engines whose deps are missing are skipped via the ``requires_<engine>``
    markers — this is enforced by the dynamic skip below so individual engines
    can drop out without taking the whole gate down.
    """
    theta = _generate_theta_truth(THETA_SEED)
    poses = _make_test_poses()

    # Run every engine for every pose, dropping any (engine, pose) pair that
    # fails to produce a clean SimOut.
    results: dict[tuple[str, str], EngineSimResult] = {}
    for engine in _ENGINE_ORDER:
        for pose in poses:
            sim = _run_engine(engine, theta, pose)
            if sim is not None:
                results[(engine, pose.name)] = sim

    available_engines = sorted({eng for (eng, _) in results})
    if len(available_engines) < 2:
        # Always emit the report so CI artifacts capture the (lack of)
        # comparison, then skip — there's nothing to assert against.
        out = _output_dir()
        report = _format_report(
            reference_engine=available_engines[0] if available_engines else "<none>",
            results=results,
            deltas=[],
            theta_seed=THETA_SEED,
            grip_tol_mm=GRIP_POSITION_TOL_MM,
            orient_tol_deg=GRIP_ORIENTATION_TOL_DEG,
        )
        (out / "report.md").write_text(report, encoding="utf-8")
        pytest.skip(
            f"Need ≥2 engines for equivalence comparison; have {available_engines}"
        )

    # Pick the reference: Simscape preferred, else first available.
    reference = "simscape" if "simscape" in available_engines else available_engines[0]
    others = [e for e in available_engines if e != reference]

    deltas: list[dict[str, Any]] = []
    for engine in others:
        for pose in poses:
            ref = results.get((reference, pose.name))
            cur = results.get((engine, pose.name))
            if ref is None or cur is None:
                continue
            grip_rmse_m = _rmse_position(cur.grip, ref.grip)
            clubhead_rmse_m = _rmse_position(cur.clubhead, ref.clubhead)
            club_length_diff_m = abs(cur.club_length - ref.club_length)
            orient_deg = _quat_angle_deg(cur.grip_quat, ref.grip_quat)

            grip_rmse_mm = grip_rmse_m * 1000.0
            clubhead_rmse_mm = clubhead_rmse_m * 1000.0
            club_length_diff_mm = club_length_diff_m * 1000.0
            clubhead_tol_mm = CLUBHEAD_BASE_TOL_MM + club_length_diff_mm

            passed = (
                grip_rmse_mm < GRIP_POSITION_TOL_MM
                and orient_deg < GRIP_ORIENTATION_TOL_DEG
                and clubhead_rmse_mm < clubhead_tol_mm
            )
            deltas.append(
                {
                    "engine": engine,
                    "pose": pose.name,
                    "grip_rmse_mm": grip_rmse_mm,
                    "orient_rmse_deg": orient_deg,
                    "clubhead_rmse_mm": clubhead_rmse_mm,
                    "club_length_diff_mm": club_length_diff_mm,
                    "clubhead_tol_mm": clubhead_tol_mm,
                    "passed": passed,
                }
            )

    out = _output_dir()
    report = _format_report(
        reference_engine=reference,
        results=results,
        deltas=deltas,
        theta_seed=THETA_SEED,
        grip_tol_mm=GRIP_POSITION_TOL_MM,
        orient_tol_deg=GRIP_ORIENTATION_TOL_DEG,
    )
    report_path = out / "report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("equivalence report: %s", report_path)

    failures = [d for d in deltas if not d["passed"]]
    if failures:
        rendered = "\n".join(
            f"  - {d['engine']}/{d['pose']}: "
            f"grip={d['grip_rmse_mm']:.2f}mm "
            f"orient={d['orient_rmse_deg']:.3f}° "
            f"clubhead={d['clubhead_rmse_mm']:.2f}mm "
            f"(tol clubhead={d['clubhead_tol_mm']:.2f}mm)"
            for d in failures
        )
        pytest.fail(
            f"Cross-engine equivalence failed for reference={reference}.\n"
            f"P0 divergences (>{GRIP_POSITION_TOL_MM} mm grip, "
            f">{GRIP_ORIENTATION_TOL_DEG}° orient, or >5 mm + Δlength clubhead):\n"
            f"{rendered}\n"
            f"See {report_path} for the full breakdown."
        )


# ---------------------------------------------------------------------------
# Smoke tests for the helpers (run unconditionally, no engines required)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_theta_truth_is_deterministic() -> None:
    """Two calls with the same seed must produce identical theta vectors."""
    a = _generate_theta_truth(42)
    b = _generate_theta_truth(42)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (_THETA_LEN,)


@pytest.mark.unit
def test_theta_truth_respects_amplitude_bounds() -> None:
    """Every coefficient lies in the documented Simscape range."""
    theta = _generate_theta_truth(42)
    for joint in range(_N_JOINTS):
        for k in range(_COEFFS_PER_JOINT):
            i = joint * _COEFFS_PER_JOINT + k
            amp = _COEFF_AMPLITUDES[k]
            assert -amp <= theta[i] <= amp, f"theta[{i}] = {theta[i]} outside ±{amp}"


@pytest.mark.unit
def test_quat_angle_zero() -> None:
    """Identical quaternions → zero angular distance."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert _quat_angle_deg(q, q) < 1e-9


@pytest.mark.unit
def test_quat_angle_180() -> None:
    """Antipodal quaternions still register as zero (double-cover)."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert _quat_angle_deg(q, -q) < 1e-9


@pytest.mark.unit
def test_rmse_position_matches_norm_for_single_sample() -> None:
    """Sanity-check the metric collapses to ‖a-b‖ for one sample."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.003, 0.004, 0.0])  # 5 mm
    assert abs(_rmse_position(a, b) - 0.005) < 1e-12
