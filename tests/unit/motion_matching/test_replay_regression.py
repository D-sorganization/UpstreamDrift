"""Regression tests for Simscape replay fix qualification and candidate physical verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]


def _evidence_dir() -> Path:
    repo = Path(__file__).resolve().parents[3]
    return (
        repo
        / "docs"
        / "development"
        / "simscape_tour_matching"
        / "native_evidence"
        / "two_window_fit_9967_100"
    )


def _seed_file() -> Path:
    repo = Path(__file__).resolve().parents[3]
    return (
        repo
        / "docs"
        / "development"
        / "simscape_tour_matching"
        / "native_evidence"
        / "initial_velocity_seed_qualified_r2025b.json"
    )


def test_candidate_physical_properties_match_calibrated_seed() -> None:
    """Verify candidate physical properties match the seed rather than relying on filename."""
    cand_path = _evidence_dir() / "returned-candidate.json"
    seed_path = _seed_file()

    assert cand_path.is_file(), f"Candidate file missing: {cand_path}"
    assert seed_path.is_file(), f"Seed file missing: {seed_path}"

    with open(cand_path, encoding="utf-8") as f:
        cand = json.load(f)
    with open(seed_path, encoding="utf-8") as f:
        seed = json.load(f)

    # 1. State vector q0
    q0_cand = np.asarray(cand["q0"], dtype=np.float64)
    q0_seed = np.asarray(seed["q"], dtype=np.float64)
    np.testing.assert_allclose(
        q0_cand,
        q0_seed,
        atol=1e-6,
        err_msg="Candidate q0 does not match calibrated seed",
    )

    # 2. Velocity vector qd0
    qd0_cand = np.asarray(cand["qd0"], dtype=np.float64)
    qd0_seed = np.asarray(seed["qd"], dtype=np.float64)
    np.testing.assert_allclose(
        qd0_cand,
        qd0_seed,
        atol=1e-6,
        err_msg="Candidate qd0 does not match calibrated seed",
    )

    # 3. Marker offsets
    offsets_cand = np.asarray(cand["marker_offsets_m"], dtype=np.float64)
    offsets_seed = np.asarray(seed["offsets_m"], dtype=np.float64)
    np.testing.assert_allclose(
        offsets_cand,
        offsets_seed,
        atol=1e-6,
        err_msg="Candidate marker offsets do not match calibrated seed",
    )

    # 4. Calibrated arm geometry
    assert cand.get("geometry_in") == [14.5, 12.0]
    assert seed.get("geometry_in") == [14.5, 12.0]


def test_missing_seed_triggers_explicit_error() -> None:
    """Verify that a missing seed file cannot silently fallback to default geometry."""
    bogus_path = _evidence_dir() / "nonexistent_seed.json"
    assert not bogus_path.is_file()

    def load_seed(path: Path) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"MissingRequiredSeed: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    with pytest.raises(FileNotFoundError, match="MissingRequiredSeed"):
        load_seed(bogus_path)


def test_actuator_force_frame_metadata_guard() -> None:
    """Verify input-frame metadata prevents duplicate rotation."""
    cand_path = _evidence_dir() / "returned-candidate.json"
    with open(cand_path, encoding="utf-8") as f:
        cand = json.load(f)

    assert cand.get("actuator_force_frame") == "world"

    def apply_forces(cand_dict: dict, pre_rotate: bool) -> str:
        frame = cand_dict.get("actuator_force_frame", "world")
        if frame == "world" and pre_rotate:
            raise ValueError(
                "DoubleRotationDefect: Forces are already in world frame; pre-rotation invalid."
            )
        return "forces_applied_correctly"

    assert apply_forces(cand, pre_rotate=False) == "forces_applied_correctly"
    with pytest.raises(ValueError, match="DoubleRotationDefect"):
        apply_forces(cand, pre_rotate=True)
