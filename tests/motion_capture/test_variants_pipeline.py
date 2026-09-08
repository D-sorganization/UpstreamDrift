"""Variants through the real pipeline on the synthetic lab session (#9792-#9794)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct.model import FitOptions
from src.motion_capture.reconstruct.model.fit2d import (
    ImageSpaceSource,
    fit_session_model_2d,
)
from src.motion_capture.reconstruct.model.golfer import GOLFER_LANDMARK_MAP, GOLFER_SPEC
from src.motion_capture.reconstruct.pipeline import (
    MatchSpec,
    reconstruct_session,
    start_cameras_from,
)
from src.motion_capture.rig import __main__ as rig_cli
from src.motion_capture.variants import get_variant, list_variants, variant_dir
from tests.motion_capture.reconstruct.test_pipeline import _session

pytestmark = pytest.mark.unit


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.timeout(300)
def test_variants_provenance_and_image_space_fit(tmp_path: Path) -> None:
    session, cameras = _session(tmp_path)
    full = reconstruct_session(
        session,
        start_cameras=start_cameras_from(cameras),
        scale_anchor=("neck", 0.5),
        match=MatchSpec(camera_source=f"cameras:{cameras.name}"),
    )
    summary = _read(session / "reconstruct" / "session_reconstruction.json")
    assert summary["schema_version"] == "session-reconstruction/1.0.0"
    assert summary["variant"] == "" and summary["observation_set"] == "observations"
    prov = summary["provenance"]
    assert prov["parameters"]["views"] == list(full.views)
    assert prov["parameters"]["camera_source"] == f"cameras:{cameras.name}"
    assert "observations/face_on.json" in {i["path"] for i in prov["inputs"]}
    assert _read(session / "reconstruct" / "reconstruction.json")["provenance"]
    assert _read(session / "reconstruct" / "clean_report.json")["schema_version"]
    default_joints = np.load(session / "reconstruct" / "joints_3d_m.npy")

    pair = reconstruct_session(
        session,
        start_cameras=start_cameras_from(cameras),
        scale_anchor=("neck", 0.5),
        match=MatchSpec(views=("face_on", "down_line"), variant="pair_fd"),
    )
    root = variant_dir(session, "pair_fd")
    assert pair.views == ("face_on", "down_line")
    assert (root / "reconstruct" / "joints_3d_m.npy").is_file()
    assert _read(root / "reconstruct" / "session_reconstruction.json")["views"] == [
        "face_on",
        "down_line",
    ]
    np.testing.assert_array_equal(
        np.load(session / "reconstruct" / "joints_3d_m.npy"), default_joints
    )
    names = [v.name for v in list_variants(session)]
    assert names == ["", "pair_fd"]
    with pytest.raises(Exception, match="no camera record"):
        reconstruct_session(
            session,
            start_cameras=start_cameras_from(cameras),
            scale_anchor=("neck", 0.5),
            match=MatchSpec(views=("face_on", "nope"), variant="bad"),
        )
    with pytest.raises(Exception, match="at least two"):
        reconstruct_session(
            session,
            start_cameras=start_cameras_from(cameras),
            scale_anchor=("neck", 0.5),
            match=MatchSpec(views=("face_on",), variant="bad"),
        )

    # Triangulated model fit and export on the pair variant through the CLI.
    argv = ["--session", str(session), "--variant", "pair_fd"]
    assert (
        rig_cli.main(
            ["fit-model", *argv, "--model", "golfer", "--max-iterations", "12"]
        )
        == 0
    )
    angles = _read(root / "model" / "joint_angles.json")
    assert angles["provenance"]["parameters"]["source"] == {"kind": "triangulate"}
    assert (
        "variants/pair_fd/reconstruct/session_reconstruction.json"
        in angles["provenance"]["derived_from"]
    )
    assert rig_cli.main(["export", *argv]) == 0
    assert (root / "reconstruct" / "reconstruction.trc").is_file()
    assert not (session / "model").exists()

    # Lineage walks from the model fit down to the observation files.
    assert (
        rig_cli.main(
            [
                "lineage",
                "--session",
                str(session),
                "--path",
                "variants/pair_fd/model/joint_angles.json",
            ]
        )
        == 0
    )

    # Image-space fit from one view, cameras from the default variant.
    fit, out_dir = fit_session_model_2d(
        session,
        GOLFER_SPEC,
        GOLFER_LANDMARK_MAP,
        ImageSpaceSource(("face_on",), "", "observations", "cam_face"),
        options=FitOptions(max_iterations=12),
    )
    assert fit.rms_px is not None and np.isfinite(fit.rms_px)
    assert out_dir == variant_dir(session, "cam_face") / "model"
    angles = _read(out_dir / "joint_angles.json")
    assert angles["provenance"]["parameters"]["source"]["kind"] == "image_space"
    assert angles["rms_px"] == pytest.approx(fit.rms_px)
    record = get_variant(session, "cam_face")
    assert record is not None and record.source == {
        "kind": "image_space",
        "cameras_from": "",
    }
    assert record.views == ("face_on",)
    # The same through the CLI (builder + parser agree).
    assert (
        rig_cli.main(
            [
                "fit-model",
                "--session",
                str(session),
                "--variant",
                "cam_down",
                "--from-views",
                "down_line",
                "--cameras-from",
                "",
                "--max-iterations",
                "8",
            ]
        )
        == 0
    )
    assert (variant_dir(session, "cam_down") / "model" / "joint_angles.json").is_file()
