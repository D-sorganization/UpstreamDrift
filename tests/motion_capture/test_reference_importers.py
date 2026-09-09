"""Existing model exports and CIR trajectories retain their measurement clocks."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reference.importers import (
    finish_motion_import,
    load_motion_draft,
)
from src.shared.python.motion_pipeline.contracts import (
    Marker,
    MarkerFrame,
    MarkerTrajectory,
)

pytestmark = pytest.mark.unit


def test_cir_mapping_preserves_missing_frames_and_signed_axes(tmp_path: Path) -> None:
    trajectory = MarkerTrajectory(
        id="model",
        frames=[
            MarkerFrame(
                timestamp=1.5, markers={"hip": Marker(name="hip", x=1, y=2, z=3)}
            ),
            MarkerFrame(timestamp=1.8, markers={}),
            MarkerFrame(
                timestamp=2.0, markers={"hip": Marker(name="hip", x=4, y=5, z=6)}
            ),
        ],
    )
    path = tmp_path / "animation.json"
    path.write_text(
        json.dumps(
            {"schema": "marker-trajectory/1.0.0", "trajectory": trajectory.model_dump()}
        )
    )
    draft = load_motion_draft(path)
    asset = finish_motion_import(
        draft,
        title="Model",
        units="m",
        axes=("-Z", "+X", "+Y"),
        joint_names=("pelvis",),
    )
    assert asset.time_s == (1.5, 1.8, 2.0)
    assert asset.points_m == (((-3, 1, 2),), (None,), ((-6, 4, 5),))
    assert len(asset.source.sha256) == 64
    assert asset.source_names == ("hip",)


def test_existing_body_target_export_is_imported_without_resampling(
    tmp_path: Path,
) -> None:
    path = tmp_path / "body.json"
    path.write_text(
        json.dumps(
            {
                "schema": "body_target_json_v1",
                "time_s": [0, 0.025],
                "marker_names": ["pelvis", "wrist", "elbow"],
                "marker_xyz": [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]] * 2,
                "impact_idx": 1,
                "events": [],
                "source": {},
                "coordinate_frame": "z_up_right_handed",
            }
        )
    )
    draft = load_motion_draft(path)
    asset = finish_motion_import(
        draft,
        title="Expert",
        units="m",
        axes=("+X", "+Y", "+Z"),
        joint_names=draft.names,
    )
    assert asset.time_s == (0, 0.025)
    assert asset.points_m[0][1] == (1, 0, 0)
    with pytest.raises(ValueError, match="Canonical"):
        finish_motion_import(
            draft,
            title="Expert",
            units="mm",
            axes=("+X", "+Y", "+Z"),
            joint_names=draft.names,
        )


def test_real_c3d_units_and_occlusions(tmp_path: Path) -> None:
    ezc3d = pytest.importorskip("ezc3d")
    c3d = ezc3d.c3d()
    c3d["parameters"]["POINT"]["RATE"]["value"] = [100]
    c3d["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = ["hip", "hand"]
    points = np.zeros((4, 2, 3))
    points[0, 1, :] = 1000
    points[:3, 1, 1] = np.nan
    c3d["data"]["points"] = points
    path = tmp_path / "sample.c3d"
    c3d.write(str(path))
    draft = load_motion_draft(path)
    asset = finish_motion_import(
        draft,
        title="C3D",
        units="mm",
        axes=("+X", "+Y", "+Z"),
        joint_names=("pelvis", "wrist"),
    )
    assert asset.time_s == (0, 0.01, 0.02)
    assert asset.points_m[0][1] == (1, 0, 0)
    assert asset.points_m[1][1] is None
