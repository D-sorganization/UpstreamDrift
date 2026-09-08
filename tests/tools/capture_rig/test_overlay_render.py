"""Model-on-video overlay rendering and the ``rig overlay`` clip (#9795)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from src.motion_capture.reconstruct.cameras import (
    PinholeCamera,
    intrinsics_from_fov,
    look_at,
)
from src.motion_capture.reconstruct.overlay3d import PALETTE, Track, skeleton_edges
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES
from src.motion_capture.rig import __main__ as rig_cli
from src.motion_capture.variants import register_variant
from src.tools.capture_rig import commands
from src.tools.capture_rig.overlay_render import (
    OverlaySpec,
    export_overlay,
    render_frame,
)
from tests.tools.capture_rig.test_core import SIZE, _bundle

pytestmark = [pytest.mark.unit]


def _camera() -> PinholeCamera:
    k = intrinsics_from_fov(SIZE[0], SIZE[1], 60.0)
    position = np.array([0.0, 1.0, 3.0])
    return PinholeCamera(
        "cam_a", k, look_at(position, np.array([0.0, 1.0, 0.0])), position, SIZE
    )


def _reconstructed_bundle(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """The video bundle plus a fabricated 12-frame reconstruction seen by cam_a."""
    root = _bundle(tmp_path)
    camera = _camera()
    joints = np.zeros((12, len(JOINT_NAMES), 3))
    rng = np.random.default_rng(0)
    joints[:] = np.array([0.0, 1.0, 0.0]) + rng.uniform(
        -0.3, 0.3, (12, len(JOINT_NAMES), 3)
    )
    recon = root / "reconstruct"
    recon.mkdir()
    np.save(recon / "joints_3d_m.npy", joints)
    (recon / "reconstruction.json").write_text(
        json.dumps({"cameras": [camera.to_calibration().to_dict()]}), encoding="utf-8"
    )
    (recon / "session_reconstruction.json").write_text(
        json.dumps({"views": ["cam_a"], "fps": 10.0}), encoding="utf-8"
    )
    register_variant(root, "", views=("cam_a",))
    return root, joints


def test_render_frame_draws_every_track_and_a_legend() -> None:
    image = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    px = np.zeros((1, len(JOINT_NAMES), 2))
    px[0, :, 0] = np.linspace(20, 120, len(JOINT_NAMES))
    px[0, :, 1] = 100.0
    track = Track(
        "v",
        "joints",
        "v joints",
        PALETTE[3],
        px,
        np.ones((1, len(JOINT_NAMES)), bool),
        skeleton_edges(),
        False,
        JOINT_NAMES,
    )
    out = render_frame(image, [track], 0)
    assert out.shape == image.shape
    assert tuple(int(c) for c in out[100, 20]) == PALETTE[3]
    assert out[:40, SIZE[0] - 300 :].max() > 0, "legend drawn top-right"
    assert render_frame(image, [track], 5, legend=False).max() == 0, "past the end"


def test_export_overlay_writes_a_clip_and_a_sidecar(tmp_path: Path) -> None:
    root, _ = _reconstructed_bundle(tmp_path)
    spec = OverlaySpec.build(root, "cam_a", [""])
    assert [t.kind for t in spec.tracks] == ["joints"] and spec.frames == 12
    out = tmp_path / "overlay.mp4"
    sidecar = export_overlay(root, "cam_a", [""], out, start=2, stop=7)
    assert out.is_file() and out.stat().st_size > 0
    assert sidecar["frames"] == 6 and sidecar["tracks"][0]["kind"] == "joints"
    assert sidecar["tracks"][0]["held_out"] is False
    written = json.loads(out.with_suffix(".json").read_text("utf-8"))
    assert written["schema_version"] == "overlay-clip/1.0.0"
    assert written["provenance"]["parameters"]["view"] == "cam_a"
    with pytest.raises(Exception, match="playable"):
        export_overlay(root, "cam_b", [""], tmp_path / "x.mp4")
    with pytest.raises(Exception, match="speed"):
        export_overlay(root, "cam_a", [""], tmp_path / "x.mp4", speed=0)


def test_overlay_cli_and_command_builder(tmp_path: Path) -> None:
    root, _ = _reconstructed_bundle(tmp_path)
    out = tmp_path / "cli.mp4"
    argv = commands.overlay_command(root, "cam_a", out, variants=("",), speed=0.5)
    assert (
        argv[3] == "overlay"
        and "--variant" in argv
        and argv[-3:] == ["0.5", "--observations", "observations"]
    )
    assert rig_cli.main(argv[3:]) == 0
    assert out.is_file()
    with pytest.raises(Exception, match="view"):
        commands.overlay_command(root, " ", out)
