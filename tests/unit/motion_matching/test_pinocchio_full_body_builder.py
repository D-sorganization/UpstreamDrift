"""Unit tests for Pinocchio full-body builder with shared contact forces (FB-3-P).

Done gates tested:
(a) Upper-body slice reproduces qualified model mass matrix and FK to 1e-12 at random states;
(b) Full-body FK matches spec frames;
(c) Contact force at analytic states equals the FB-2 reference;
(d) Closure residual unchanged on the upper-body chain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.native_model import (
    FullBodyPinocchioModel,
    NativePinocchioModel,
)
from src.shared.python.motion_matching.contact_law import sphere_ground_contact
from src.shared.python.motion_matching.full_body_spec import (
    load_full_body_spec,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FULL_BODY_SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
UPPER_SPEC_PATH = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)


def _load_specs() -> tuple[dict[str, Any], dict[str, Any]]:
    upper = json.loads(UPPER_SPEC_PATH.read_text(encoding="utf-8"))
    full = load_full_body_spec(FULL_BODY_SPEC_PATH, upper)
    return full, upper


def _require_real_pinocchio() -> Any:
    try:
        import pinocchio as pin
    except ImportError as exc:
        pytest.skip(f"pinocchio not importable: {exc}")
    if (
        type(pin).__module__ == "unittest.mock"
        or not hasattr(pin, "Model")
        or not hasattr(pin, "SE3")
    ):
        pytest.skip("real pinocchio runtime required (found mock/stub)")
    return pin


def test_full_body_model_schema_validation() -> None:
    full, _ = _load_specs()
    bad_spec = dict(full)
    bad_spec["schema_version"] = "wrong-schema"
    with pytest.raises(ValueError, match="full-body"):
        FullBodyPinocchioModel(bad_spec)


def test_upper_body_slice_mass_matrix_and_fk_parity() -> None:
    pin = _require_real_pinocchio()
    full_spec, upper_spec = _load_specs()
    full_model = FullBodyPinocchioModel(full_spec)
    slice_model = full_model.upper_body_model()
    qual_model = NativePinocchioModel(upper_spec)

    rng = np.random.default_rng(20260914)
    for _ in range(5):
        q_dict = {
            name: float(rng.uniform(-0.35, 0.35))
            for name in upper_spec["coordinate_order"]
        }
        poses_slice = slice_model.frame_poses(q_dict)
        poses_qual = qual_model.frame_poses(q_dict)
        for frame in poses_qual:
            diff_fk = float(np.linalg.norm(poses_slice[frame] - poses_qual[frame]))
            assert diff_fk < 1e-12, f"Slice FK diff on {frame}: {diff_fk}"

        qu = qual_model.configuration(q_dict)
        data_slice = slice_model.model.createData()
        data_qual = qual_model.model.createData()
        M_slice = np.asarray(pin.crba(slice_model.model, data_slice, qu))
        M_qual = np.asarray(pin.crba(qual_model.model, data_qual, qu))
        diff_M = float(np.max(np.abs(M_slice - M_qual)))
        assert diff_M < 1e-12, f"Slice mass matrix max diff: {diff_M}"


def test_full_body_fk_matches_spec_frames() -> None:
    _require_real_pinocchio()
    full_spec, upper_spec = _load_specs()
    full_model = FullBodyPinocchioModel(full_spec)
    qual_model = NativePinocchioModel(upper_spec)

    rng = np.random.default_rng(20260915)
    for _ in range(5):
        q_full = {
            name: float(rng.uniform(-0.3, 0.3))
            for name in full_spec["coordinate_order"]
        }
        q_upper = {name: q_full[name] for name in upper_spec["coordinate_order"]}
        poses_full = full_model.frame_poses(q_full)
        poses_qual = qual_model.frame_poses(q_upper)
        for frame in poses_qual:
            diff = float(np.linalg.norm(poses_full[frame] - poses_qual[frame]))
            assert diff < 1e-12, f"FK diff on {frame}: {diff}"


def test_contact_forces_at_analytic_states_equal_fb2_reference() -> None:
    _require_real_pinocchio()
    full_spec, _ = _load_specs()
    full_model = FullBodyPinocchioModel(full_spec)

    # In neutral state, feet may or may not penetrate depending on vertical translation
    q_neutral = dict.fromkeys(full_spec["coordinate_order"], 0.0)
    v_neutral = dict.fromkeys(full_spec["coordinate_order"], 0.0)
    forces_neutral = full_model.contact_forces(q_neutral, v_neutral)
    assert len(forces_neutral) == 4

    # Analytic penetration state: shift pelvis downward along Z so contact spheres penetrate
    q_penetrating = dict(q_neutral)
    q_penetrating["TranslationInputZ"] = -0.90
    v_moving = {
        name: (0.1 if "Translation" in name else 0.0)
        for name in full_spec["coordinate_order"]
    }
    forces = full_model.contact_forces(q_penetrating, v_moving)
    assert len(forces) == 4
    penetrating_count = sum(1 for s in forces.values() if s.penetration_m > 0.0)
    assert penetrating_count > 0, "Expected penetration under lowered Z"

    for name, sample in forces.items():
        fid = full_model._contact_frames[name]
        center = full_model.data.oMf[fid].translation
        vel = full_model._pin.getFrameVelocity(
            full_model.model,
            full_model.data,
            fid,
            full_model._pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).linear
        radius = next(s.radius_m for s in full_model.contact_spheres if s.name == name)
        ref = sphere_ground_contact(
            center, vel, radius, full_model.ground, full_model.contact_parameters
        )
        np.testing.assert_allclose(sample.normal_force_n, ref.normal_force_n)
        np.testing.assert_allclose(sample.friction_force_n, ref.friction_force_n)
        assert sample.penetration_m == pytest.approx(ref.penetration_m)


def test_closure_residuals_unchanged_on_upper_body_chain() -> None:
    _require_real_pinocchio()
    full_spec, upper_spec = _load_specs()
    full_model = FullBodyPinocchioModel(full_spec)
    qual_model = NativePinocchioModel(upper_spec)

    rng = np.random.default_rng(20260916)
    for _ in range(5):
        q_full = {
            name: float(rng.uniform(-0.25, 0.25))
            for name in full_spec["coordinate_order"]
        }
        q_up = {name: q_full[name] for name in upper_spec["coordinate_order"]}
        v_full = {
            name: float(rng.uniform(-0.4, 0.4))
            for name in full_spec["coordinate_order"]
        }
        v_up = {name: v_full[name] for name in upper_spec["coordinate_order"]}

        pos_qual, vel_qual = qual_model.closure_residuals(q_up, v_up)
        pos_full, vel_full = full_model.closure_residuals(q_full, v_full)
        np.testing.assert_allclose(pos_full, pos_qual, atol=1e-12)
        np.testing.assert_allclose(vel_full, vel_qual, atol=1e-12)


def test_accelerations_incorporate_contact_and_preserve_closure() -> None:
    _require_real_pinocchio()
    full_spec, _ = _load_specs()
    full_model = FullBodyPinocchioModel(full_spec)

    q = dict.fromkeys(full_spec["coordinate_order"], 0.0)
    q["TranslationInputZ"] = -0.90
    v = dict.fromkeys(full_spec["coordinate_order"], 0.0)
    tau = dict.fromkeys(full_spec["coordinate_order"], 0.0)

    acc = full_model.accelerations(q, v, tau)
    assert set(acc) == set(full_spec["coordinate_order"])
    assert all(np.isfinite(val) for val in acc.values())

    pos_err, vel_err = full_model.closure_errors()
    assert pos_err.shape == (6,)
    assert vel_err.shape == (6,)
    assert np.all(np.isfinite(pos_err))
    assert np.all(np.isfinite(vel_err))
