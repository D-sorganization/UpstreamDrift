"""Measured neutral visual height, independently checked in native MuJoCo."""

import json
import numpy as np
import pytest

from src.shared.python.model_appearance.player_stature import (
    build_stature_calibrated,
    neutral_visual_height,
)

pytestmark = pytest.mark.unit


@pytest.mark.timeout(240)
@pytest.mark.parametrize("stature,mass", [(1.8542, 83.91458845), (1.7272, 68.0388555)])
def test_neutral_height_from_mujoco_meshes(stature, mass):
    """Verify stature, leg distances and mass using the compiled model."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.model_appearance import document_from_dict
    from src.shared.python.motion_matching.anthropometry import segment_parameters

    spec, info = build_stature_calibrated(stature, mass)
    assert abs(neutral_visual_height(spec) - stature) <= 0.001
    assert info["iterations"] > 0
    assert spec["subject"]["mass_kg"] == mass
    total = (
        sum(s["mass_kg"] for b in spec["bodies"] for s in b["solids"])
        - spec["club"]["total_mass_kg"]
    )
    assert total == pytest.approx(mass, abs=1e-9)
    from src.shared.python.motion_matching.anthropometric_candidate import (
        zero_pose_joint_positions,
    )

    positions = zero_pose_joint_positions(spec)
    for side in ("l", "r"):
        for segment, proximal, distal in (
            ("thigh", "hip", "knee"),
            ("shank", "knee", "ankle"),
        ):
            actual = np.linalg.norm(
                positions[f"{distal}_{side}"] - positions[f"{proximal}_{side}"]
            )
            assert actual == pytest.approx(
                segment_parameters(stature, mass, segment).length_m, abs=1e-10
            )
    doc = document_from_dict(
        {"schema_version": "appearance-v1", "head": {"headwear": "none"}}
    )
    xml, _ = export_full_body_mjcf(json.dumps(spec).encode(), appearance=doc)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def height(i):
        """Transform one native mesh to world-frame vertical coordinates."""
        mesh = model.geom_dataid[i]
        vertices = model.mesh_vert[
            model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh]
            + model.mesh_vertnum[mesh]
        ]
        return (vertices @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i])[:, 2]

    head = height(model.geom("visual_head_skin_mesh").id).max()
    soles = [
        height(i).min()
        for i in range(model.ngeom)
        if "sole0" in (model.geom(i).name or "")
    ]
    assert soles
    assert head - min(soles) == pytest.approx(stature, abs=0.001)


@pytest.mark.parametrize(
    "height,mass",
    [(0.0, 80.0), (-1.0, 80.0), (float("nan"), 80.0), (1.8, float("inf"))],
)
def test_invalid_subject_is_rejected(height, mass):
    """Reject nonpositive and nonfinite subject dimensions."""
    with pytest.raises(ValueError, match="positive and finite"):
        build_stature_calibrated(height, mass)


def test_unreachable_target_is_not_clamped(monkeypatch):
    """Reject a target outside the supported scale bracket."""
    from src.shared.python.model_appearance import player_stature

    monkeypatch.setattr(player_stature, "neutral_visual_height", lambda spec: 10.0)
    with pytest.raises(ValueError, match="different signs"):
        build_stature_calibrated(1.8, 80.0)
