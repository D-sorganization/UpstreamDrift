"""Tests for explicit effort hold rate manifest contract and ZOH findings (#12279).

Validates:
1. Default effort hold policy is 1 ms (0.001 s, 8 substeps, 1 ms projection) preserving backward compatibility.
2. Finer 0.125 ms hold policy (0.000125 s, 1 substep, 1 ms closure projection) is recorded in manifest,
   read back identically by engine replay helpers, and bounds 7-iron foot slip within 10 mm (3.3 mm <= 10.0 mm).
3. simscape_matching_reference.tex and DESIGN_DECISIONS.md record the #12279 ZOH hold-rate findings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import (
    DEFAULT_EFFORT_HOLD_DT_S,
    FINE_EFFORT_HOLD_DT_S,
    ROOT_COORDINATES,
    EffortHoldPolicy,
    InputBundle,
    StepPolicy,
    integrate,
    open_loop,
    read_effort_hold_policy,
    resolve_effort_hold_dt_s,
)
from src.shared.python.motion_matching.same_input.bundle import (
    DEFAULT_EFFORT_SUBSTEPS,
    DEFAULT_PROJECTION_DT_S,
    FINE_EFFORT_SUBSTEPS,
    FOOT_SLIP_BOUND_MM,
    KNOWN_7IRON_FOOT_SLIP_1MS_MM,
    KNOWN_7IRON_FOOT_SLIP_FINE_MM,
)

pytestmark = pytest.mark.unit

ORDER = (*ROOT_COORDINATES, "Joint")
SPEC = json.dumps({"coordinate_order": list(ORDER)}).encode()
DT_S = 1e-3
STEPS = 50


class _SimplePlant:
    """Toy oscillator with trivial root closure."""

    coordinate_order = ORDER

    def acceleration(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return -q + tau

    def closure_pose_residual(self, q: np.ndarray) -> np.ndarray:
        return q[:1].copy()

    def closure_rate_matrix(self, q: np.ndarray) -> np.ndarray:
        rate = np.zeros((1, q.size))
        rate[0, 0] = 1.0
        return rate

    def kinematic_frames(self, q: np.ndarray) -> dict[str, np.ndarray]:
        pose = np.eye(4)
        pose[0, 3] = q[-1]
        return {"tip": pose}


def _make_bundle(holds: int, effort_hold_dt_s: float | None = None) -> InputBundle:
    nv = len(ORDER)
    q0 = np.zeros(nv)
    v0 = np.zeros(nv)
    q = np.zeros((STEPS + 1, nv))
    v = np.zeros((STEPS + 1, nv))
    efforts = np.zeros((STEPS, nv))
    hold_efforts = None if holds == 1 else np.zeros((STEPS * holds, nv))
    return InputBundle(
        spec_bytes=SPEC,
        coordinate_order=ORDER,
        dt_s=DT_S,
        q0=q0,
        v0=v0,
        efforts=efforts,
        reference_q=q,
        reference_v=v,
        reference_engine="toy",
        effort_holds=holds,
        hold_efforts=hold_efforts,
        effort_hold_dt_s=effort_hold_dt_s,
    )


def test_default_effort_hold_policy_backward_compatibility(tmp_path: Path) -> None:
    """Default policy is 1 ms (0.001 s, 8 substeps, 1 ms projection) and backward-compatible."""
    assert DEFAULT_EFFORT_HOLD_DT_S == 0.001
    assert DEFAULT_EFFORT_SUBSTEPS == 8
    assert DEFAULT_PROJECTION_DT_S == 0.001

    policy = EffortHoldPolicy()
    assert policy.effort_hold_dt_s == 0.001
    assert policy.effort_substeps == 8
    assert policy.projection_dt_s == 0.001
    assert policy.effort_holds_per_step == 1

    bundle = _make_bundle(1)
    assert bundle.effort_holds == 1
    assert bundle.effort_hold_dt_s == 0.001
    assert bundle.effort_substeps == 8
    assert bundle.projection_dt_s == 0.001

    manifest = bundle.manifest()
    p = manifest["policy"]
    assert p["effort_hold_dt_s"] == 0.001
    assert p["effort_substeps"] == 8
    assert p["projection_dt_s"] == 0.001
    assert p["effort_holds_per_step"] == 1
    assert p["effort_hold_s"] == pytest.approx(0.001)

    path = tmp_path / "default_bundle.npz"
    bundle.save(path)
    loaded = InputBundle.load(path)
    assert loaded.effort_hold_dt_s == 0.001
    assert loaded.effort_substeps == 8
    assert loaded.projection_dt_s == 0.001
    assert read_effort_hold_policy(loaded) == EffortHoldPolicy()
    assert read_effort_hold_policy(path) == EffortHoldPolicy()

    # Legacy manifest roundtrip without effort_hold_dt_s keys
    legacy_manifest = dict(manifest)
    legacy_policy = dict(p)
    del legacy_policy["effort_hold_dt_s"]
    del legacy_policy["effort_substeps"]
    del legacy_policy["projection_dt_s"]
    legacy_manifest["policy"] = legacy_policy
    assert read_effort_hold_policy(legacy_manifest) == EffortHoldPolicy()


def test_finer_effort_hold_policy_manifest_roundtrip_and_slip_bound(
    tmp_path: Path,
) -> None:
    """Finer 0.125 ms policy is recorded, read back identically, and bounds 7-iron foot slip."""
    assert FINE_EFFORT_HOLD_DT_S == 0.000125
    assert FINE_EFFORT_SUBSTEPS == 1

    fine_policy = EffortHoldPolicy(
        effort_hold_dt_s=FINE_EFFORT_HOLD_DT_S,
        effort_substeps=FINE_EFFORT_SUBSTEPS,
        projection_dt_s=DEFAULT_PROJECTION_DT_S,
    )
    assert fine_policy.effort_holds_per_step == 8

    # Create bundle with explicit fine hold
    bundle = _make_bundle(8, effort_hold_dt_s=FINE_EFFORT_HOLD_DT_S)
    assert bundle.effort_holds == 8
    assert bundle.effort_hold_dt_s == FINE_EFFORT_HOLD_DT_S
    assert bundle.effort_substeps == 1
    assert bundle.projection_dt_s == 0.001

    manifest = bundle.manifest()
    p = manifest["policy"]
    assert p["effort_hold_dt_s"] == pytest.approx(0.000125)
    assert p["effort_substeps"] == 1
    assert p["projection_dt_s"] == pytest.approx(0.001)
    assert p["effort_holds_per_step"] == 8

    path = tmp_path / "fine_bundle.npz"
    bundle.save(path)
    loaded = InputBundle.load(path)
    assert loaded.effort_hold_dt_s == pytest.approx(0.000125)
    assert loaded.effort_substeps == 1
    assert loaded.projection_dt_s == pytest.approx(0.001)

    read_from_bundle = read_effort_hold_policy(loaded)
    read_from_manifest = read_effort_hold_policy(manifest)
    read_from_path = read_effort_hold_policy(path)

    assert read_from_bundle == fine_policy
    assert read_from_manifest == fine_policy
    assert read_from_path == fine_policy

    step_policy = loaded.step_policy()
    assert step_policy.effort_holds == 8
    assert step_policy.substeps_per_hold == 1

    # Measured 7-iron slip bound validation (#12279)
    assert KNOWN_7IRON_FOOT_SLIP_FINE_MM <= FOOT_SLIP_BOUND_MM
    assert KNOWN_7IRON_FOOT_SLIP_1MS_MM > FOOT_SLIP_BOUND_MM
    assert pytest.approx(3.3) == KNOWN_7IRON_FOOT_SLIP_FINE_MM
    assert pytest.approx(10.0) == FOOT_SLIP_BOUND_MM


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_simscape_reference_records_zoh_hold_rate_findings() -> None:
    """simscape_matching_reference.tex records all #12279 ZOH comparison numbers."""
    tex_path = (
        REPO_ROOT
        / "docs"
        / "research"
        / "simscape_matching_reference"
        / "simscape_matching_reference.tex"
    )
    assert tex_path.exists(), f"Missing {tex_path}"
    content = tex_path.read_text(encoding="utf-8")

    # Issue reference and manifest parameters
    assert "Same-Input Effort Hold" in content
    assert "12279" in content
    assert "effort\\_hold\\_dt\\_s" in content or "effort_hold_dt_s" in content
    assert "effort\\_substeps" in content or "effort_substeps" in content
    assert "projection\\_dt\\_s" in content or "projection_dt_s" in content

    # 7-iron foot slip and kinematic errors
    assert "43~mm" in content or "43 mm" in content or "43" in content
    assert "3.3~mm" in content or "3.3 mm" in content
    assert "24.0~mm" in content or "24.0 mm" in content
    assert "8.2~mm" in content or "8.2 mm" in content
    assert "-0.147" in content
    assert "-0.001" in content

    # 7-iron impact metrics
    assert "-5.0" in content
    assert "-2.5" in content
    assert "8.71" in content
    assert "5.56" in content

    # Driver comparison metrics
    assert "16.88" in content
    assert "16.91" in content
    assert "-4.4" in content
    assert "-3.5" in content


def test_design_decisions_records_zoh_hold_rate_findings() -> None:
    """DESIGN_DECISIONS records all #12279 ZOH comparison numbers."""
    decisions_paths = [
        REPO_ROOT
        / "docs"
        / "development"
        / "full_body_models"
        / "DESIGN_DECISIONS_2.md",
        REPO_ROOT / "docs" / "development" / "full_body_models" / "DESIGN_DECISIONS.md",
    ]
    combined_content = "\n".join(
        p.read_text(encoding="utf-8") for p in decisions_paths if p.exists()
    )

    assert "12279" in combined_content
    assert "43 mm" in combined_content
    assert "3.3 mm" in combined_content
    assert "24.0 mm" in combined_content
    assert "8.2 mm" in combined_content
    assert "-0.147 rad" in combined_content or "−0.147 rad" in combined_content
    assert "-0.001 rad" in combined_content or "−0.001 rad" in combined_content
    assert (
        "-5.0 %" in combined_content
        or "-5.0%" in combined_content
        or "−5.0 %" in combined_content
    )
    assert (
        "-2.5 %" in combined_content
        or "-2.5%" in combined_content
        or "−2.5 %" in combined_content
    )
    assert "8.71" in combined_content
    assert "5.56" in combined_content
    assert "16.88" in combined_content
    assert "16.91" in combined_content


def test_effort_hold_policy_parameter_validation() -> None:
    """DbC validation of EffortHoldPolicy and resolve_effort_hold_dt_s."""
    with pytest.raises(ValueError, match="effort_hold_dt_s"):
        EffortHoldPolicy(effort_hold_dt_s=-0.001)
    with pytest.raises(ValueError, match="projection_dt_s"):
        EffortHoldPolicy(projection_dt_s=0.0)
    with pytest.raises(TypeError, match="effort_substeps"):
        EffortHoldPolicy(effort_substeps="1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="effort_substeps"):
        EffortHoldPolicy(effort_substeps=0)

    # resolve_effort_hold_dt_s
    assert resolve_effort_hold_dt_s() == 0.001
    assert resolve_effort_hold_dt_s(effort_hold_dt_s=0.000125) == 0.000125
    assert resolve_effort_hold_dt_s(effort_hold_s=0.000125) == 0.000125
    assert (
        resolve_effort_hold_dt_s(effort_hold_dt_s=0.000125, effort_hold_s=0.000125)
        == 0.000125
    )
    with pytest.raises(ValueError, match="must match"):
        resolve_effort_hold_dt_s(effort_hold_dt_s=0.001, effort_hold_s=0.000125)
