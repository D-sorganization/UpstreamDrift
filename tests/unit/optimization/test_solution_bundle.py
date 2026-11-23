"""Solution Bundle schema and codec contract (Refs #15493, Slice 2)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.optimization.solution_bundle import (
    SCHEMA_VERSION,
    SolutionBundle,
    SolutionBundleError,
    load_solution_bundle,
    save_solution_bundle,
)

N_STEPS = 6
N_JOINTS = 3
JOINTS = ("pelvis_rz", "thorax_rz", "lead_wrist_flex")


def _bundle() -> SolutionBundle:
    rng = np.random.default_rng(15493)

    def grid() -> np.ndarray:
        return rng.normal(size=(N_STEPS, N_JOINTS))

    arrays = {
        "time": np.linspace(0.0, 0.25, N_STEPS),
        "q_ref": grid(),
        "qd_ref": grid(),
        "tau_ff_ref": grid(),
        "q_rollout": grid(),
        "qd_rollout": grid(),
        "tau_cmd": grid(),
        "tau_applied": grid(),
        "knots": np.array([0.0, 0.0, 0.0, 0.125, 0.25, 0.25, 0.25]),
        "delta_tau_coeffs": rng.normal(size=(4, N_JOINTS)),
        "q0": rng.normal(size=N_JOINTS),
        "qd0": rng.normal(size=N_JOINTS),
        "controller_state0": np.zeros(0),
        "kp": np.full(N_JOINTS, 200.0),
        "kd": np.full(N_JOINTS, 20.0),
    }
    return SolutionBundle(
        solution_id="sol-0001",
        parent_id="match-0000",
        git_sha="a" * 40,
        model_hash="m" * 64,
        asset_hashes={"club": "c" * 64},
        frame="world_z_up",
        joint_order=JOINTS,
        control={"hold": "zoh", "saturation": {"mode": "clip", "limit_nm": 150.0}},
        integrator={"name": "rk4", "dt": 0.05, "rtol": 1e-8},
        environment={"gravity": [0.0, 0.0, -9.81]},
        optimizer_seed=7,
        impact_event={
            "t": 0.2,
            "q": [0.1, 0.2, 0.3],
            "qd": [1.0, 2.0, 3.0],
        },
        flight_metrics={"carry_m": 231.5, "ball_speed_ms": 70.1},
        feasibility_margins={"torque_nm": 12.0},
        assumptions=("rigid shaft",),
        arrays=arrays,
    )


def test_roundtrip_preserves_manifest_and_arrays_exactly(tmp_path: Path) -> None:
    original = _bundle()
    save_solution_bundle(original, tmp_path / "b")
    loaded = load_solution_bundle(tmp_path / "b")

    assert loaded == replace(original, arrays=loaded.arrays)
    assert loaded.schema_version == SCHEMA_VERSION
    assert set(loaded.arrays) == set(original.arrays)
    for name, value in original.arrays.items():
        np.testing.assert_array_equal(loaded.arrays[name], value)
        assert loaded.arrays[name].dtype == value.dtype


def test_manifest_is_strict_json_with_declared_conventions(tmp_path: Path) -> None:
    save_solution_bundle(_bundle(), tmp_path / "b")
    manifest = json.loads((tmp_path / "b" / "manifest.json").read_text("utf-8"))

    assert manifest["units"] == "SI"
    assert manifest["quaternion_convention"] == "wxyz"
    assert manifest["joint_order"] == list(JOINTS)
    assert len(manifest["arrays_sha256"]) == 64


def test_load_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    save_solution_bundle(_bundle(), tmp_path / "b")
    path = tmp_path / "b" / "manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    manifest["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(manifest), "utf-8")

    with pytest.raises(SolutionBundleError, match="unsupported schema_version"):
        load_solution_bundle(tmp_path / "b")


def test_load_rejects_incompatible_model(tmp_path: Path) -> None:
    save_solution_bundle(_bundle(), tmp_path / "b")

    with pytest.raises(SolutionBundleError, match="model_hash"):
        load_solution_bundle(tmp_path / "b", expected_model_hash="x" * 64)
    load_solution_bundle(tmp_path / "b", expected_model_hash="m" * 64)


def test_load_rejects_tampered_arrays(tmp_path: Path) -> None:
    save_solution_bundle(_bundle(), tmp_path / "b")
    npz = tmp_path / "b" / "arrays.npz"
    npz.write_bytes(npz.read_bytes() + b"\x00")

    with pytest.raises(SolutionBundleError, match="arrays_sha256"):
        load_solution_bundle(tmp_path / "b")


def test_load_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(SolutionBundleError, match="manifest.json"):
        load_solution_bundle(tmp_path / "nope")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_save_rejects_nonfinite_arrays(tmp_path: Path, bad: float) -> None:
    bundle = _bundle()
    bundle.arrays["tau_applied"][2, 1] = bad

    with pytest.raises(SolutionBundleError, match="tau_applied.*non-finite"):
        save_solution_bundle(bundle, tmp_path / "b")
    assert not (tmp_path / "b").exists()


def test_save_rejects_nonmonotonic_time(tmp_path: Path) -> None:
    bundle = _bundle()
    bundle.arrays["time"][3] = bundle.arrays["time"][2]

    with pytest.raises(SolutionBundleError, match="strictly increasing"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_rejects_dimension_mismatch(tmp_path: Path) -> None:
    bundle = _bundle()
    bundle.arrays["q_rollout"] = bundle.arrays["q_rollout"][:-1]

    with pytest.raises(SolutionBundleError, match="q_rollout.*shape"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_rejects_missing_array(tmp_path: Path) -> None:
    bundle = _bundle()
    del bundle.arrays["kp"]

    with pytest.raises(SolutionBundleError, match="missing arrays.*kp"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_rejects_decreasing_knots(tmp_path: Path) -> None:
    bundle = _bundle()
    bundle.arrays["knots"] = bundle.arrays["knots"][::-1].copy()

    with pytest.raises(SolutionBundleError, match="knots.*nondecreasing"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_rejects_nonfinite_impact_event(tmp_path: Path) -> None:
    bundle = _bundle()
    bundle = replace(bundle, impact_event={**bundle.impact_event, "t": float("nan")})

    with pytest.raises(SolutionBundleError, match="impact_event"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_rejects_impact_dimension_mismatch(tmp_path: Path) -> None:
    bundle = _bundle()
    bundle = replace(bundle, impact_event={**bundle.impact_event, "q": [0.0]})

    with pytest.raises(SolutionBundleError, match="impact_event.q"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_rejects_duplicate_joint_names(tmp_path: Path) -> None:
    bundle = replace(_bundle(), joint_order=("a", "a", "b"))

    with pytest.raises(SolutionBundleError, match="joint_order"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_refuses_to_overwrite_existing_bundle(tmp_path: Path) -> None:
    save_solution_bundle(_bundle(), tmp_path / "b")

    with pytest.raises(SolutionBundleError, match="already exists"):
        save_solution_bundle(_bundle(), tmp_path / "b")
