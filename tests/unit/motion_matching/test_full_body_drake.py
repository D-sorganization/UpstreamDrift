"""Unit tests for Drake full-body URDF export and KKT adapter (FB-3-D, #10067).

TDD tests covering:
1. Full-body URDF export structure and metadata (tested without Drake runtime).
2. Clean skip when pydrake is unavailable.
3. Upper-body slice mass matrix and FK parity (< 1e-12).
4. Full-body FK matching spec frames (< 1e-12).
5. Contact forces matching FB-2 reference at analytic states (< 1e-12).
6. Closure residuals unchanged on the upper-body chain (< 1e-12).
7. Constrained forward accelerations combining contact wrenches and weld closure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import defusedxml.ElementTree as ET

import numpy as np
import pytest

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


def _require_real_drake() -> Any:
    try:
        import pydrake.all as drake_all
    except ImportError as exc:
        pytest.skip(f"pydrake not importable: {exc}")
    if type(drake_all).__module__ == "unittest.mock" or not hasattr(
        drake_all, "MultibodyPlant"
    ):
        pytest.skip("real pydrake runtime required (found mock/stub)")
    return drake_all


def test_export_full_body_urdf_structure_and_metadata() -> None:
    from src.engines.physics_engines.drake.python.full_body_urdf import (
        export_full_body_urdf,
    )

    full_spec, _ = _load_specs()
    spec_bytes = json.dumps(full_spec).encode("utf-8")
    xml, sidecar = export_full_body_urdf(spec_bytes)

    assert sidecar["schema_version"] == 1
    assert sidecar["requires_sidecar"] is True
    assert sidecar["model_sha256"] == hashlib.sha256(spec_bytes).hexdigest()
    assert sidecar["urdf_sha256"] == hashlib.sha256(xml.encode("utf-8")).hexdigest()
    assert len(sidecar["coordinate_order"]) == 41
    assert sidecar["coordinate_order"] == full_spec["coordinate_order"]

    tree = ET.fromstring(xml)
    assert tree.tag == "robot"

    # All 41 scalar joints present
    joints = tree.findall("joint")
    joint_names = [j.get("name") for j in joints]
    for coord in full_spec["coordinate_order"]:
        assert coord in joint_names

    # 4 contact sphere links present
    links = tree.findall("link")
    link_names = [link.get("name") for link in links]
    assert "contact_heel_r" in link_names
    assert "contact_forefoot_r" in link_names
    assert "contact_heel_l" in link_names
    assert "contact_forefoot_l" in link_names


def test_full_body_drake_schema_validation() -> None:
    _require_real_drake()
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )

    full_spec, _ = _load_specs()
    bad_spec = dict(full_spec)
    bad_spec["schema_version"] = "wrong-schema"
    with pytest.raises(ValueError, match="full-body"):
        FullBodyDrakeModel(bad_spec)


def test_upper_body_slice_mass_matrix_and_fk_parity() -> None:
    _require_real_drake()
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )
    from src.shared.python.motion_matching.full_body_spec import (
        FULL_BODY_SCHEMA_VERSION,
    )

    full_spec, upper_spec = _load_specs()
    full_model = FullBodyDrakeModel(full_spec)
    slice_model = full_model.upper_body_model()

    qual_spec = {
        "schema_version": FULL_BODY_SCHEMA_VERSION,
        "gravity_m_s2": upper_spec["gravity_m_s2"],
        "bodies": upper_spec["bodies"],
        "joints": upper_spec["joints"],
        "coordinate_order": upper_spec["coordinate_order"],
        "frames": upper_spec["frames"],
        "closure": upper_spec["closure"],
        "contact": {
            "ground": {"normal_policy": "opposite_gravity", "height_m": 0.0},
            "parameters": full_spec["contact"]["parameters"],
            "spheres": [],
        },
    }
    qual_model = FullBodyDrakeModel(qual_spec)

    # Verify link masses and inertias of upper body bodies
    for b in upper_spec["bodies"]:
        if b["name"] == "world":
            continue
        body_full = full_model.plant.GetBodyByName(
            full_model.metadata["body_links"][b["name"]], full_model._instance
        )
        body_qual = qual_model.plant.GetBodyByName(
            qual_model.metadata["body_links"][b["name"]], qual_model._instance
        )
        assert abs(body_full.default_mass() - body_qual.default_mass()) < 1e-12

    rng = np.random.default_rng(20260914)
    upper_coords = upper_spec["coordinate_order"]
    for _ in range(5):
        q_dict = {name: float(rng.uniform(-0.35, 0.35)) for name in upper_coords}
        poses_full = full_model.frame_poses(
            {**dict.fromkeys(full_spec["coordinate_order"], 0.0), **q_dict}
        )
        poses_slice = slice_model.frame_poses(q_dict)
        poses_qual = qual_model.frame_poses(q_dict)
        for frame in poses_slice:
            diff_fk = float(np.linalg.norm(poses_slice[frame] - poses_qual[frame]))
            assert diff_fk < 1e-12, f"Slice vs Qual FK diff on {frame}: {diff_fk}"
            diff_full_fk = float(np.linalg.norm(poses_full[frame] - poses_slice[frame]))
            assert diff_full_fk < 1e-12, (
                f"Full vs Slice FK diff on {frame}: {diff_full_fk}"
            )

        slice_model.plant.SetPositions(
            slice_model.context, slice_model._vector(q_dict, slice_model._q_indices)
        )
        qual_model.plant.SetPositions(
            qual_model.context, qual_model._vector(q_dict, qual_model._q_indices)
        )
        m_slice = slice_model.plant.CalcMassMatrix(slice_model.context)
        m_qual = qual_model.plant.CalcMassMatrix(qual_model.context)
        np.testing.assert_allclose(m_slice, m_qual, atol=1e-12, rtol=0)


def test_full_body_contact_forces_analytic_state() -> None:
    _require_real_drake()
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )

    full_spec, _ = _load_specs()
    full_model = FullBodyDrakeModel(full_spec)

    coords = full_spec["coordinate_order"]
    q_pen = dict.fromkeys(coords, 0.0)
    q_pen["TranslationInputZ"] = -0.90
    v_moving = {name: (0.1 if "Translation" in name else 0.0) for name in coords}

    forces = full_model.contact_forces(q_pen, v_moving)
    assert len(forces) == 4
    pen_count = sum(1 for s in forces.values() if s.penetration_m > 0.0)
    assert pen_count > 0, "Expected penetration with Z=-0.90"

    for name, sample in forces.items():
        pos, vel, radius = full_model.get_sphere_kinematics(name, q_pen, v_moving)
        ref = sphere_ground_contact(
            pos, vel, radius, full_model.ground_plane, full_model.contact_parameters
        )
        np.testing.assert_allclose(
            sample.normal_force_n, ref.normal_force_n, atol=1e-12
        )
        np.testing.assert_allclose(
            sample.friction_force_n, ref.friction_force_n, atol=1e-12
        )
        assert sample.penetration_m == pytest.approx(ref.penetration_m, abs=1e-12)


def test_closure_residuals_unchanged_on_upper_body() -> None:
    _require_real_drake()
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )

    full_spec, upper_spec = _load_specs()
    full_model = FullBodyDrakeModel(full_spec)
    slice_model = full_model.upper_body_model()

    rng = np.random.default_rng(20260916)
    coords = full_spec["coordinate_order"]
    upper_coords = upper_spec["coordinate_order"]
    for _ in range(5):
        q_full = {name: float(rng.uniform(-0.25, 0.25)) for name in coords}
        q_up = {name: q_full[name] for name in upper_coords}
        v_full = {name: float(rng.uniform(-0.4, 0.4)) for name in coords}
        v_up = {name: v_full[name] for name in upper_coords}

        pos_slice, vel_slice = slice_model.closure_residuals(q_up, v_up)
        pos_full, vel_full = full_model.closure_residuals(q_full, v_full)
        np.testing.assert_allclose(pos_full, pos_slice, atol=1e-12)
        np.testing.assert_allclose(vel_full, vel_slice, atol=1e-12)


def test_drake_accelerations_with_contact_and_closure() -> None:
    _require_real_drake()
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )

    full_spec, _ = _load_specs()
    full_model = FullBodyDrakeModel(full_spec)

    coords = full_spec["coordinate_order"]
    q = dict.fromkeys(coords, 0.0)
    q["TranslationInputZ"] = -0.90
    v = dict.fromkeys(coords, 0.0)
    tau = dict.fromkeys(coords, 0.0)

    acc = full_model.accelerations(q, v, tau)
    assert len(acc) == 41
    assert all(np.isfinite(val) for val in acc.values())

    pos_err, vel_err = full_model.closure_errors()
    assert pos_err.shape == (6,)
    assert vel_err.shape == (6,)
    assert np.all(np.isfinite(pos_err))
    assert np.all(np.isfinite(vel_err))
