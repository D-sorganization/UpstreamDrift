"""Single-frame forward kinematics of a same-input plant (#11744).

The LIFT-4 bushing plant differentiates one frame (the hand body) by finite
differences many times per RK4 stage.  ``VectorPlant.kinematic_frame`` returns
that frame alone; it must equal ``kinematic_frames(q)[name]`` bit for bit, so
the speed-up cannot change any committed trajectory.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.lifting.bundles import BACK_SQUAT_BUNDLE
from src.shared.python.motion_matching.same_input import InputBundle, VectorPlant

pytestmark = pytest.mark.unit

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def bundle() -> InputBundle:
    return InputBundle.load(BACK_SQUAT_BUNDLE)


def _states(bundle: InputBundle) -> list[np.ndarray]:
    rng = np.random.default_rng(11744)
    q0 = np.asarray(bundle.q0, dtype=float)
    return [q0, *(q0 + 0.05 * rng.standard_normal(q0.size) for _ in range(3))]


def test_mujoco_kinematic_frame_equals_kinematic_frames_exactly(
    bundle: InputBundle,
) -> None:
    plant = VectorPlant("mujoco", bundle.spec_bytes)
    for q in _states(bundle):
        frames = plant.kinematic_frames(q)
        for name, pose in frames.items():
            assert np.array_equal(plant.kinematic_frame(q, name), pose), name


def test_kinematic_frame_falls_back_without_a_native_single_frame(
    bundle: InputBundle,
) -> None:
    plant = VectorPlant("mujoco", bundle.spec_bytes)
    q = _states(bundle)[1]
    expected = plant.kinematic_frames(q)["Grip"]

    class _NoSingleFrame:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def frame_poses(self, coordinates: dict[str, float]) -> dict:
            return self._inner.frame_poses(coordinates)  # type: ignore[attr-defined]

    plant._adapter = _NoSingleFrame(plant._adapter)  # noqa: SLF001 - fallback path
    assert np.array_equal(plant.kinematic_frame(q, "Grip"), expected)


def test_kinematic_frame_rejects_unknown_frames(bundle: InputBundle) -> None:
    plant = VectorPlant("mujoco", bundle.spec_bytes)
    with pytest.raises(KeyError, match="no_such_frame"):
        plant.kinematic_frame(np.asarray(bundle.q0, dtype=float), "no_such_frame")
    with pytest.raises(TypeError, match="frame name"):
        plant.kinematic_frame(np.asarray(bundle.q0, dtype=float), 3)  # type: ignore[arg-type]
