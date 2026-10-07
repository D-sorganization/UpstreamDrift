"""Unit tests for the pure parts of the muscle graft (issue #11617, phase 2)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from src.engines.physics_engines.opensim.python import musculoskeletal_graft as g

pytestmark = pytest.mark.unit


def _spec(twist_r: float = -25.0, twist_l: float = 40.0, skew: float = 0.0) -> dict:
    axes = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    joints = []
    for side, twist, y in (("r", twist_r, -0.1), ("l", twist_l, 0.1)):
        p = np.eye(4)
        p[:3, :3] = axes @ g.rotation_z(twist) + (skew if side == "l" else 0.0)
        p[:3, 3] = [0.0, y, -0.05]
        joints.append({"name": f"hip_{side}", "parent_to_base": p.tolist()})
    return {
        "joints": joints,
        "subject": {"hip_zero_twist_deg": {"r": twist_r, "l": twist_l}},
        "_axes": axes,
    }


def test_pelvis_frame_recovers_axes_and_centres() -> None:
    spec = _spec()
    rotation, centres, residual = g.pelvis_frame_from_spec(spec)
    np.testing.assert_allclose(rotation, spec["_axes"], atol=1e-12)
    assert residual < 1e-12
    np.testing.assert_allclose(centres["r"], [0.0, -0.1, -0.05])
    np.testing.assert_allclose(centres["l"], [0.0, 0.1, -0.05])


def test_pelvis_frame_rejects_inconsistent_sides_and_missing_data() -> None:
    with pytest.raises(ValueError, match="disagree"):
        g.pelvis_frame_from_spec(_spec(skew=0.05))
    with pytest.raises(ValueError, match="hip_zero_twist_deg"):
        g.pelvis_frame_from_spec({"joints": [], "subject": {}})


def test_map_pelvis_point_preserves_hip_centre_relation() -> None:
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    base_c, spec_c = np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.5, 0.0])
    np.testing.assert_allclose(
        g.map_pelvis_point(base_c, rotation, base_c, spec_c), spec_c
    )
    out = g.map_pelvis_point(base_c + [1, 0, 0], rotation, base_c, spec_c)
    np.testing.assert_allclose(out, spec_c + [0, 1, 0], atol=1e-12)


def test_map_wrap_orientation_composes_rotations() -> None:
    fixed = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    local = [0.1, -0.2, 0.3]
    mapped = g.map_wrap_orientation(local, fixed)
    expected = fixed @ Rotation.from_euler("XYZ", local).as_matrix()
    np.testing.assert_allclose(
        Rotation.from_euler("XYZ", mapped).as_matrix(), expected, atol=1e-12
    )


def test_leg_scale_factors_and_validation() -> None:
    base = {f"{b}_{s}": [0.0, -0.2, 0.0] for b in g.LEG_BODIES for s in g.SIDES}
    spec = {k: [0.0, -0.2 * 1.1, 0.0] for k in base}
    scales = g.leg_scale_factors(spec, base, talus_scale=0.95)
    assert scales["femur_r"] == pytest.approx(1.1)
    assert scales["talus_l"] == 0.95
    assert scales["patella_r"] == scales["femur_r"]
    assert scales["pelvis"] == pytest.approx(1.1)
    spec["toes_r"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="zero mass centre"):
        g.leg_scale_factors(spec, base, talus_scale=0.95)
    with pytest.raises(ValueError):
        g.leg_scale_factors(spec, base, talus_scale=0.0)


def test_spec_leg_mass_centres_and_side() -> None:
    spec = {
        "bodies": [
            {"name": "femur_r", "solids": [{"mass_kg": 2.0, "com_m": [0, -1, 0]}]},
            {"name": "Hip", "solids": [{"mass_kg": 1.0, "com_m": [1, 1, 1]}]},
            {"name": "talus_l", "solids": [{"mass_kg": 0.0, "com_m": [0, 0, 0]}]},
        ]
    }
    centres = g.spec_leg_mass_centres(spec)
    assert set(centres) == {"femur_r"}
    assert g.side_of("psoas_r") == "r" and g.side_of("Hip") is None


def test_spec_document_rejects_other_schema() -> None:
    with pytest.raises(ValueError):
        g.spec_document(b'{"schema_version": "v0"}')
