"""Unit tests for the musculoskeletal swing helpers (issue #11617)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.engines.physics_engines.opensim.python import musculoskeletal_swing as ms
from src.engines.physics_engines.opensim.python.musculoskeletal_grf import (
    default_contact_points,
    friction_generators,
    solve_contact_forces,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_solvers import (
    SolveWindow,
    summarize_solution,
)

pytestmark = pytest.mark.unit


def test_muscle_group_and_side() -> None:
    assert ms.muscle_group("glmax2_r") == "hip extensors"
    assert ms.muscle_group("recfem_l") == "quadriceps"
    assert ms.muscle_group("gaslat_r") == "plantarflexors"
    assert ms.muscle_group("mystery_r") == "other"
    assert ms.muscle_side("soleus_l") == "l"
    assert ms.muscle_side("lumbar") == "?"
    with pytest.raises(ValueError):
        ms.muscle_group("")


def test_map_coordinates_state_paths_and_unmapped() -> None:
    labels = [
        "/jointset/hip_r/hip_flexion_r/value",
        "/jointset/hip_r/hip_flexion_r/speed",
        "/jointset/foo/bar/value",
    ]
    mapping = ms.map_coordinates(labels, ["hip_flexion_r", "knee_angle_r"])
    assert mapping.mapped == {"hip_flexion_r": labels[0]}
    assert mapping.unmapped_model == ("knee_angle_r",)
    assert mapping.unmapped_source == (labels[2],)
    assert mapping.coverage(["hip_flexion_r", "knee_angle_r"]) == pytest.approx(0.5)


def test_map_coordinates_rejects_duplicates() -> None:
    with pytest.raises(ms.MappingError):
        ms.map_coordinates(["a/x/value", "b/x/value"], ["x"])


def test_sto_round_trip(tmp_path: Path) -> None:
    t = np.linspace(0.0, 1.0, 20)
    cols = {"a": np.sin(t), "b": t**2}
    path = ms.write_sto(tmp_path / "k.sto", t, cols)
    t2, back = ms.read_states_table(path)
    np.testing.assert_allclose(t2, t, atol=1e-8)
    np.testing.assert_allclose(back["a"], cols["a"], atol=1e-8)
    with pytest.raises(FileNotFoundError):
        ms.read_states_table(tmp_path / "missing.sto")
    with pytest.raises(ValueError):
        ms.write_sto(tmp_path / "bad.sto", t, {"a": t[:-1]})


def test_smooth_kinematics_removes_noise_keeps_signal() -> None:
    t = np.arange(0, 2.0, 1.0 / 360.0)
    clean = np.sin(2 * np.pi * 1.0 * t)
    noisy = clean + 0.05 * np.sin(2 * np.pi * 90.0 * t)
    out = ms.smooth_kinematics(t, {"q": noisy}, 15.0)["q"]
    assert np.max(np.abs(out[30:-30] - clean[30:-30])) < 0.01
    with pytest.raises(ValueError):
        ms.smooth_kinematics(t, {"q": noisy}, 500.0)
    with pytest.raises(ValueError):
        ms.smooth_kinematics(t**2 + t, {"q": noisy}, 15.0)


def test_resolve_base_model_reports_probed_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ms, "REPO_ROOT", tmp_path)
    monkeypatch.delenv(ms.BASE_MODEL_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="Probed"):
        ms.resolve_base_model()
    fake = tmp_path / "m.osim"
    fake.write_text("<x/>")
    assert ms.resolve_base_model(fake) == fake


def _vertical_jacobian() -> np.ndarray:
    """One point whose position is the three root translations (identity)."""
    jac = np.zeros((1, 3, 6))
    jac[0, :, 3:] = np.eye(3)
    return jac


def test_contact_force_supports_weight() -> None:
    required = np.array([0.0, 0.0, 0.0, 0.0, 800.0, 0.0])
    forces, spin, resid = solve_contact_forces(
        required, _vertical_jacobian(), np.array([True])
    )
    assert resid < 1.0
    assert forces[0, 1] == pytest.approx(800.0, rel=1e-2)
    assert abs(forces[0, 0]) < 5.0 and spin[0] == 0.0


def test_contact_force_friction_limit_leaves_residual() -> None:
    required = np.array([0.0, 0.0, 0.0, 800.0, 100.0, 0.0])  # needs mu = 8
    forces, _, resid = solve_contact_forces(
        required, _vertical_jacobian(), np.array([True]), mu=0.5
    )
    assert resid > 100.0
    assert np.all(forces[:, 1] >= 0.0)


def test_contact_force_inactive_points_carry_nothing() -> None:
    required = np.array([0.0, 0.0, 0.0, 0.0, 50.0, 0.0])
    forces, _, resid = solve_contact_forces(
        required, _vertical_jacobian(), np.array([False])
    )
    assert not forces.any()
    assert resid == pytest.approx(50.0)


def test_contact_force_validates_shapes() -> None:
    with pytest.raises(ValueError):
        solve_contact_forces(np.zeros(5), _vertical_jacobian(), np.array([True]))
    with pytest.raises(ValueError):
        friction_generators(0.0)


def test_default_contact_points_cover_both_feet() -> None:
    pts = default_contact_points()
    assert len(pts) == 8
    assert {p.foot for p in pts} == {"r", "l"}
    assert {p.body for p in pts} == {"calcn_r", "toes_r", "calcn_l", "toes_l"}


def test_summarize_solution_flags_reserves() -> None:
    t = np.linspace(0, 1, 11)
    acts = {
        "glmax1_r": np.full(11, 0.2),
        "soleus_r": np.linspace(0, 1, 11),
        "reserve_hip_flexion_r": np.full(11, 3.0),
        "upper_lumbar_extension": np.full(11, 10.0),
    }
    out = summarize_solution(t, acts, muscle_names=["glmax1_r", "soleus_r"])
    assert out["per_muscle"]["soleus_r"]["peak"] == pytest.approx(1.0)
    assert out["per_group"]["plantarflexors"]["n_muscles"] == 1
    assert out["reserve_torques"]["reserve_hip_flexion_r"]["rms"] == pytest.approx(3.0)
    assert out["upper_body_torques"]["upper_lumbar_extension"]["peak"] == 10.0
    assert out["reserve_rms_max"] == pytest.approx(3.0)
    assert out["muscles_saturated_fraction"] == pytest.approx(0.5)


def test_solve_window_validates() -> None:
    SolveWindow(0.0, 1.0)
    with pytest.raises(ValueError):
        SolveWindow(1.0, 1.0)


def test_build_model_has_muscles_club_and_golfer_mass() -> None:
    pytest.importorskip("opensim")
    golf = (
        Path(__file__).resolve().parents[4]
        / "src/engines/physics_engines/opensim/models/golf_humanoid_scaled.osim"
    )
    try:
        ms.resolve_base_model()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    model, info = ms.build_musculoskeletal_model(golf)
    assert info["n_muscles"] == 80
    assert model.getBodySet().hasComponent("Club")
    assert info["total_mass_kg"] == pytest.approx(79.6987, rel=1e-3)
    kinds = set(info["actuator_kinds"].values())
    assert kinds == {"upper", "reserve"}
    assert not model.getCoordinateSet().get("wrist_flex_r").get_locked()
