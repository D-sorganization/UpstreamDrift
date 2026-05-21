"""Real C3D body-target video export coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("ezc3d")

from src.shared.python.motion_matching import AlignOptions
from src.shared.python.motion_matching.body_skeleton import default_body_segments
from src.shared.python.motion_matching.diagnostics.body_target_video import (
    save_c3d_body_video,
)
from src.shared.python.motion_matching.load_body_target import load_body_target_c3d
from src.shared.python.motion_matching.loaders.c3d_body import (
    default_anatomical_marker_set,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER_C3D = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"


@pytest.mark.integration
@pytest.mark.headless_safe
def test_driver_c3d_skeleton_video_export_is_nonblank(tmp_path: Path) -> None:
    if not DRIVER_C3D.exists():
        pytest.skip(f"C3D fixture not present: {DRIVER_C3D}")

    opts = AlignOptions(
        sample_rate_hz=120.0,
        simulation_time_s=0.3,
        time_alignment="impact",
        impact_target_t_s=0.25,
    )
    target = load_body_target_c3d(DRIVER_C3D, opts)
    assert target.marker_xyz.shape == (
        37,
        len(default_anatomical_marker_set()),
        3,
    )
    assert default_body_segments(target.marker_names)

    out = tmp_path / "tour_average_driver_preview.mp4"
    result = save_c3d_body_video(
        DRIVER_C3D,
        out,
        opts=opts,
        frame_indices=(0, 6, 12, 18, 24, 30, 36),
        fps=12.0,
        width=480,
        height=360,
        title="Tour average driver",
    )

    assert result.frame_count == 7
    assert out.exists()
    assert out.stat().st_size > 5_000

    cap = cv2.VideoCapture(str(out))
    try:
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert reported_frames >= 1
        ok, frame = cap.read()
        assert ok
        assert frame.std() > 0
    finally:
        cap.release()
