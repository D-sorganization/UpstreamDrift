"""Candidate viewing must preserve named coordinates, model identity, and physics (MV-02, #10478)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.native_candidate_viewer import (
    Candidate,
    build_visuals,
    load_candidate,
)
from src.shared.python.body_part_viz.anatomical_visuals import VisualSkinMode

pytestmark = [pytest.mark.unit]


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    spec = tmp_path / "model.json"
    spec.write_text(json.dumps({"coordinate_order": ["a", "b"]}), encoding="utf-8")
    candidate = tmp_path / "candidate.npz"
    np.savez(
        candidate,
        time_s=[0.0, 0.1],
        q=[[1.0, 2.0], [3.0, 4.0]],
        coordinate_order=["b", "a"],
    )
    receipt = {"document_sha256": hashlib.sha256(spec.read_bytes()).hexdigest()}
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return candidate, spec


def test_reorders_by_name_and_keeps_physical_time(files: tuple[Path, Path]) -> None:
    candidate = load_candidate(*files)
    np.testing.assert_array_equal(candidate.q, [[2.0, 1.0], [4.0, 3.0]])
    np.testing.assert_array_equal(candidate.time_s, [0.0, 0.1])


def test_changed_model_is_rejected(files: tuple[Path, Path]) -> None:
    files[1].write_text(
        '{"coordinate_order": ["a", "b"], "changed": true}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Candidate/model hash mismatch"):
        load_candidate(*files)


@pytest.mark.parametrize(
    "names,time,q",
    [
        (["a", "a"], [0.0, 0.1], [[1.0, 2.0], [3.0, 4.0]]),
        (["a", "b"], [0.0, 0.0], [[1.0, 2.0], [3.0, 4.0]]),
        (["a", "b"], [0.0, 0.1], [[1.0, 2.0], [3.0, np.nan]]),
    ],
)
def test_invalid_trajectory_is_rejected(
    files: tuple[Path, Path], names: list[str], time: list[float], q: list[list[float]]
) -> None:
    np.savez(files[0], time_s=time, q=q, coordinate_order=names)
    with pytest.raises(ValueError):
        load_candidate(*files)


@pytest.mark.requires_pinocchio
def test_build_visuals_modes_and_physics_immutability() -> None:
    """Toggling skins must create valid visual geometries while leaving physics unchanged."""
    import sys

    pin_mod = sys.modules.get("pinocchio")
    if pin_mod is not None and type(pin_mod).__module__ == "unittest.mock":
        pytest.skip("Pinocchio is mocked in unit test environment")

    try:
        import pinocchio as pin
        from src.engines.physics_engines.pinocchio.python.native_model import (
            FullBodyPinocchioModel,
        )
    except (ImportError, AttributeError):
        pytest.skip("Pinocchio native bindings unavailable")

    repo_root = Path(__file__).resolve().parents[3]
    spec_path = (
        repo_root
        / "docs/development/full_body_models/evidence/ground_support/anthro_driver/full_body_spec_hipcal_scaled.json"
    )
    if not spec_path.is_file():
        pytest.skip(f"Full body spec not found at {spec_path}")

    spec: dict[str, Any] = json.loads(spec_path.read_text(encoding="utf-8"))
    plant = FullBodyPinocchioModel(spec)

    # Initial physical properties snapshot
    initial_inertias = [i.copy() for i in plant.model.inertias]
    initial_contacts = [
        (c.name, c.body, float(c.radius_m), tuple(c.position_m))
        for c in plant.contact_spheres
    ]
    q0 = pin.neutral(plant.model)

    # Build inertia ellipsoids
    geom_ellipsoids = build_visuals(plant, spec, skin_mode="inertia_ellipsoids")
    assert geom_ellipsoids.ngeoms > 0

    # Build anatomical meshes
    geom_anatomical = build_visuals(plant, spec, skin_mode="anatomical_mesh")
    assert geom_anatomical.ngeoms > 0

    # Assert that plant.model was NOT mutated in any way
    assert len(plant.model.inertias) == len(initial_inertias)
    for i1, i2 in zip(plant.model.inertias, initial_inertias, strict=True):
        assert np.array_equal(i1.inertia, i2.inertia)
        assert np.array_equal(i1.lever, i2.lever)
        assert i1.mass == i2.mass

    # Assert contact spheres are unchanged
    current_contacts = [
        (c.name, c.body, float(c.radius_m), tuple(c.position_m))
        for c in plant.contact_spheres
    ]
    assert len(current_contacts) == len(initial_contacts)
    for c1, c2 in zip(current_contacts, initial_contacts, strict=True):
        assert c1[0] == c2[0] and c1[1] == c2[1] and c1[2] == c2[2]
        assert c1[3] == c2[3]

    # Compute FK and mass matrix CRBA with q0
    pin.forwardKinematics(plant.model, plant.data, q0)
    pin.computeJointJacobians(plant.model, plant.data, q0)
    pin.crba(plant.model, plant.data, q0)
    m0 = plant.data.M.copy()

    # Again ensure plant data and kinematics match
    pin.forwardKinematics(plant.model, plant.data, q0)
    pin.crba(plant.model, plant.data, q0)
    np.testing.assert_array_equal(plant.data.M, m0)
