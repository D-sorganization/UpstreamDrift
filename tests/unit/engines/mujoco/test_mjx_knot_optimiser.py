"""Unit tests for MJX knot optimiser core (#11049)."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

import defusedxml.ElementTree as DET
import mujoco
import numpy as np

from src.engines.physics_engines.mujoco.python.motion_matching.mjx_knot_optimiser import (
    ARMATURE_KG_M2,
    ROOT_VERTICAL_COORDINATE,
    WELD_DAMPING_N_S_M,
    WELD_STIFFNESS_N_M,
    DiagnoseResult,
    KnotOptimisationResult,
    KnotOptimisationSettings,
    MjxPackage,
    diagnose_reference,
    load_mjx_package,
    optimise_reference,
)
from tests.unit.engines.mujoco.mjx_toy_package import write_toy_package

pytestmark = pytest.mark.unit


def test_settings_valid_defaults() -> None:
    """Verify KnotOptimisationSettings default values."""
    s = KnotOptimisationSettings()
    assert s.iterations == 40
    assert s.substeps == 6
    assert s.knot_spacing_s == 0.04
    assert s.learning_rate == 2e-3
    assert s.regularisation == 1e-3
    assert s.horizon_s == 1.65
    assert s.weld_stiffness == WELD_STIFFNESS_N_M
    assert s.weld_damping == WELD_DAMPING_N_S_M


def test_settings_rejected_iterations() -> None:
    """Verify validation rejects negative iterations."""
    with pytest.raises(ValueError, match="iterations"):
        KnotOptimisationSettings(iterations=-1)


def test_settings_rejected_substeps() -> None:
    """Verify validation rejects substeps < 1."""
    with pytest.raises(ValueError, match="substeps"):
        KnotOptimisationSettings(substeps=0)


def test_settings_rejected_knot_spacing_s() -> None:
    """Verify validation rejects non-positive or non-finite knot_spacing_s."""
    with pytest.raises(ValueError, match="knot_spacing_s"):
        KnotOptimisationSettings(knot_spacing_s=0.0)
    with pytest.raises(ValueError, match="knot_spacing_s"):
        KnotOptimisationSettings(knot_spacing_s=-0.04)
    with pytest.raises(ValueError, match="knot_spacing_s"):
        KnotOptimisationSettings(knot_spacing_s=float("nan"))
    with pytest.raises(ValueError, match="knot_spacing_s"):
        KnotOptimisationSettings(knot_spacing_s=float("inf"))


def test_settings_rejected_learning_rate() -> None:
    """Verify validation rejects non-positive or non-finite learning_rate."""
    with pytest.raises(ValueError, match="learning_rate"):
        KnotOptimisationSettings(learning_rate=0.0)
    with pytest.raises(ValueError, match="learning_rate"):
        KnotOptimisationSettings(learning_rate=-2e-3)
    with pytest.raises(ValueError, match="learning_rate"):
        KnotOptimisationSettings(learning_rate=float("nan"))


def test_settings_rejected_regularisation() -> None:
    """Verify validation rejects negative or non-finite regularisation."""
    with pytest.raises(ValueError, match="regularisation"):
        KnotOptimisationSettings(regularisation=-1e-3)
    with pytest.raises(ValueError, match="regularisation"):
        KnotOptimisationSettings(regularisation=float("nan"))
    with pytest.raises(ValueError, match="regularisation"):
        KnotOptimisationSettings(regularisation=float("inf"))


def test_settings_rejected_horizon_s() -> None:
    """Verify validation rejects non-positive or non-finite horizon_s."""
    with pytest.raises(ValueError, match="horizon_s"):
        KnotOptimisationSettings(horizon_s=0.0)
    with pytest.raises(ValueError, match="horizon_s"):
        KnotOptimisationSettings(horizon_s=-1.0)
    with pytest.raises(ValueError, match="horizon_s"):
        KnotOptimisationSettings(horizon_s=float("nan"))


def test_settings_rejected_weld_stiffness() -> None:
    """Verify validation rejects non-positive or non-finite weld_stiffness."""
    with pytest.raises(ValueError, match="weld_stiffness"):
        KnotOptimisationSettings(weld_stiffness=0.0)
    with pytest.raises(ValueError, match="weld_stiffness"):
        KnotOptimisationSettings(weld_stiffness=-100.0)
    with pytest.raises(ValueError, match="weld_stiffness"):
        KnotOptimisationSettings(weld_stiffness=float("inf"))


def test_settings_rejected_weld_damping() -> None:
    """Verify validation rejects non-positive or non-finite weld_damping."""
    with pytest.raises(ValueError, match="weld_damping"):
        KnotOptimisationSettings(weld_damping=0.0)
    with pytest.raises(ValueError, match="weld_damping"):
        KnotOptimisationSettings(weld_damping=-1.0)
    with pytest.raises(ValueError, match="weld_damping"):
        KnotOptimisationSettings(weld_damping=float("nan"))


def test_load_mjx_package_strips_equality_and_floors_armature(tmp_path: Path) -> None:
    """Verify load_mjx_package removes equalities and floors dof_armature."""
    write_toy_package(tmp_path)
    xml_str = (tmp_path / "mjx_package.xml").read_text(encoding="utf-8")
    raw_root = DET.fromstring(xml_str)
    assert len(raw_root.findall("equality")) > 0, "Toy XML must contain equality"

    raw_model = mujoco.MjModel.from_xml_string(xml_str)
    assert raw_model.neq > 0, "Raw model must have equalities"
    assert np.any(raw_model.dof_armature < ARMATURE_KG_M2)

    pkg = load_mjx_package(tmp_path)
    assert isinstance(pkg, MjxPackage)
    assert pkg.model.neq == 0, "Stripped package model must have 0 equalities"
    assert np.all(pkg.model.dof_armature >= ARMATURE_KG_M2), "Armatures must be floored"
    assert ROOT_VERTICAL_COORDINATE in pkg.meta["coordinate_order"]
    assert "q_track" in pkg.arrays


def test_load_mjx_package_missing_file_raises(tmp_path: Path) -> None:
    """Verify load_mjx_package raises FileNotFoundError naming missing file."""
    write_toy_package(tmp_path)
    missing = tmp_path / "mjx_package.xml"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="mjx_package.xml"):
        load_mjx_package(tmp_path)


def test_optimise_reference_iterations_0(tmp_path: Path) -> None:
    """Verify optimise_reference with iterations=0 returns port check replay RMS."""
    write_toy_package(tmp_path)
    pkg = load_mjx_package(tmp_path)
    settings = KnotOptimisationSettings(
        iterations=0,
        substeps=2,
        knot_spacing_s=0.05,
        weld_stiffness=1000.0,
        weld_damping=10.0,
        horizon_s=0.2,
    )
    res = optimise_reference(pkg, settings)
    assert isinstance(res, KnotOptimisationResult)
    assert len(res.history) == 1
    assert res.history[0]["replay_marker_rms_m"] == res.port_check_replay_marker_rms_m
    assert res.best_replay_marker_rms_m == res.port_check_replay_marker_rms_m
    assert res.q_best.shape == pkg.arrays["q_track"].shape
    assert res.delta_best.shape == (res.knots, res.actuated_coordinates)


def test_optimise_reference_iterations_2(tmp_path: Path) -> None:
    """Verify optimise_reference with iterations=2 achieves best <= history[0]."""
    write_toy_package(tmp_path)
    pkg = load_mjx_package(tmp_path)
    settings = KnotOptimisationSettings(
        iterations=2,
        substeps=2,
        knot_spacing_s=0.05,
        weld_stiffness=1000.0,
        weld_damping=10.0,
        horizon_s=0.2,
    )
    res = optimise_reference(pkg, settings)
    assert len(res.history) == 3
    assert res.best_replay_marker_rms_m <= res.history[0]["replay_marker_rms_m"]


def test_optimise_reference_init_delta_wrong_shape_raises(tmp_path: Path) -> None:
    """Verify optimise_reference rejects init_delta of mismatched shape."""
    write_toy_package(tmp_path)
    pkg = load_mjx_package(tmp_path)
    settings = KnotOptimisationSettings(
        iterations=0,
        substeps=2,
        knot_spacing_s=0.05,
        weld_stiffness=1000.0,
        weld_damping=10.0,
        horizon_s=0.2,
    )
    bad_delta = np.zeros((100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="init_delta"):
        optimise_reference(pkg, settings, init_delta=bad_delta)


def test_optimise_reference_on_iteration_callback(tmp_path: Path) -> None:
    """Verify on_iteration is called once per iteration with (k, delta, total, objective, q)."""
    write_toy_package(tmp_path)
    pkg = load_mjx_package(tmp_path)
    settings = KnotOptimisationSettings(
        iterations=2,
        substeps=2,
        knot_spacing_s=0.05,
        weld_stiffness=1000.0,
        weld_damping=10.0,
        horizon_s=0.2,
    )
    calls: list[tuple[int, np.ndarray, float, float, np.ndarray]] = []

    def callback(
        k: int, delta: np.ndarray, total: float, objective: float, q: np.ndarray
    ) -> None:
        calls.append((k, delta, total, objective, q))

    res = optimise_reference(pkg, settings, on_iteration=callback)
    assert len(calls) == 3
    assert [c[0] for c in calls] == [0, 1, 2]
    for _k, delta, total, objective, q in calls:
        assert delta.shape == (res.knots, res.actuated_coordinates)
        assert q.shape == pkg.arrays["q_track"].shape
        assert math.isfinite(total)
        assert math.isfinite(objective)


def test_diagnose_reference(tmp_path: Path) -> None:
    """Verify diagnose_reference returns finite markers and diagnostics on toy package."""
    write_toy_package(tmp_path)
    pkg = load_mjx_package(tmp_path)
    settings = KnotOptimisationSettings(
        substeps=2,
        knot_spacing_s=0.05,
        weld_stiffness=1000.0,
        weld_damping=10.0,
        horizon_s=0.2,
    )
    res = diagnose_reference(pkg, settings)
    assert isinstance(res, DiagnoseResult)
    assert np.isfinite(res.markers).all()
    assert np.isfinite(res.q_sim).all()
    assert np.isfinite(res.peak_qvel).all()
    assert res.first_bad_frame == -1
    assert math.isfinite(res.replay_marker_rms_m)
    assert res.valid.shape == res.markers.shape[:2]


def test_optimise_reference_missing_root_coordinate_raises(tmp_path: Path) -> None:
    """The root vertical coordinate is looked up by name, never guessed."""
    write_toy_package(tmp_path)
    pkg = load_mjx_package(tmp_path)
    meta = dict(pkg.meta)
    meta["coordinate_order"] = ["root_z", "joint_1", "joint_2"]
    renamed = MjxPackage(meta=meta, arrays=pkg.arrays, model=pkg.model)
    with pytest.raises(ValueError, match=ROOT_VERTICAL_COORDINATE):
        optimise_reference(renamed, KnotOptimisationSettings(iterations=0))
