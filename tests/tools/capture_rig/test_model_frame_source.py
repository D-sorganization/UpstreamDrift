"""Model playback is a virtual view using the real comparison compositor."""

import numpy as np
import pytest

from src.tools.capture_rig.model_frame_source import ModelFrameSource
from tests.motion_capture.test_reference_registration import sample_motion

pytestmark = pytest.mark.unit


def test_default_display_rate_retains_uniform_source_samples() -> None:
    motion = sample_motion()
    source = ModelFrameSource.from_motion(motion)
    assert source.fps == 2
    assert source.frame_count == len(motion.time_s)
    assert [source.time_at(i) for i in range(source.frame_count)] == list(motion.time_s)
    source.close()


def test_model_frames_have_a_fixed_virtual_camera_and_source_clock() -> None:
    motion = sample_motion()
    source = ModelFrameSource.from_motion(motion, fps=20, size=(640, 480))
    assert (source.width, source.height, source.frame_count, source.fps) == (
        640,
        480,
        41,
        20,
    )
    assert source.time_at(10) == 0.5
    camera = source.recipe.camera
    first, top = source.read(0), source.read(10)
    assert first.shape == (480, 640, 3) and first.dtype == np.uint8
    assert np.any(first != top)
    assert source.recipe.camera == camera
    assert "virtual" in camera.provenance.lower()
    assert not source.recipe.registration.is_calibrated
    first[:] = 0
    assert np.any(source.read(0))
    assert source.read(41) is None
    source.close()
    with pytest.raises(ValueError, match="closed"):
        source.read(0)


@pytest.mark.parametrize(
    "fps,size",
    [(0, (640, 480)), (0.5, (640, 480)), (float("nan"), (640, 480)), (30, (0, 480))],
)
def test_invalid_model_display_inputs_fail(fps: float, size: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        ModelFrameSource.from_motion(sample_motion(), fps=fps, size=size)


def test_sparse_source_resamples_at_encodable_rate_and_bounds_display_budget() -> None:
    motion = sample_motion().changed(time_s=(0, 10, 20, 30, 40))
    source = ModelFrameSource.from_motion(motion)
    assert source.fps == 1
    assert source.time_at(40) == 40
    source.close()
    oversized = motion.changed(time_s=(0, 100000, 200000, 300000, 400000))
    with pytest.raises(ValueError, match="display frames"):
        ModelFrameSource.from_motion(oversized, fps=360)


def test_shared_appearance_and_handedness_change_virtual_frames() -> None:
    source = ModelFrameSource.from_motion(sample_motion(), fps=20, size=(640, 480))
    recipe = source.recipe
    mirrored = recipe.model_validate(
        recipe.model_dump()
        | {
            "registration": recipe.registration.model_dump() | {"mirror_lateral": True},
            "appearance": recipe.appearance.model_dump() | {"draw_ellipsoids": True},
        }
    )
    other = ModelFrameSource(mirrored)
    assert np.any(source.read(10) != other.read(10))
    assert other.recipe.asset == source.recipe.asset
    with pytest.raises(ValueError):
        source.time_at(-1)
    source.close()
    other.close()
