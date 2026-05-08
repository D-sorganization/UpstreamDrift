"""Tests for the OpenSim ScaleTool wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    Marker,
    MarkerFrame,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.scaling.opensim_scale import (
    OpenSimScaleBackend,
)

opensim = pytest.importorskip("opensim")


def _rig() -> SkeletonRig:
    return SkeletonRig(
        id="generic",
        joints={
            "pelvis": JointDef(
                name="pelvis",
                parent=None,
                children=["femur"],
                tpose_offset=[0.0, 0.0, 0.0],
                axes=["X"],
            ),
            "femur": JointDef(
                name="femur",
                parent="pelvis",
                children=[],
                tpose_offset=[0.0, -0.4, 0.0],
                axes=["X"],
            ),
        },
        root_joint="pelvis",
    )


def _markers() -> MarkerFrame:
    return MarkerFrame(
        timestamp=0.0,
        markers={
            "femur_prox": Marker(name="femur_prox", x=0.0, y=0.0, z=0.0),
            "femur_dist": Marker(name="femur_dist", x=0.0, y=-0.45, z=0.0),
            "pelvis_l": Marker(name="pelvis_l", x=-0.1, y=0.0, z=0.0),
            "pelvis_r": Marker(name="pelvis_r", x=0.1, y=0.0, z=0.0),
        },
    )


def test_opensim_scale_backend_smoke():
    rig = _rig()
    markers = _markers()
    marker_to_segment = {
        "femur_prox": "femur",
        "femur_dist": "femur",
        "pelvis_l": "pelvis",
        "pelvis_r": "pelvis",
    }
    backend = OpenSimScaleBackend(mass_kg=70.0, height_m=1.75)
    scaled = backend.scale(rig, markers, marker_to_segment)

    assert scaled.id != rig.id
    # postcondition: every non-root segment has a positive length
    for jname, jdef in scaled.joints.items():
        if jdef.parent is None:
            continue
        length = float(np.linalg.norm(jdef.tpose_offset))
        assert length > 0, f"{jname} length must be positive"


def test_opensim_scale_backend_validation():
    with pytest.raises(ValueError):
        OpenSimScaleBackend(mass_kg=-1.0)
    with pytest.raises(ValueError):
        OpenSimScaleBackend(height_m=0)
