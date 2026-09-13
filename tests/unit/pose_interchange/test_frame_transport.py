"""Frame transport checks with independently specified offset signs."""

import numpy as np
import pytest
from src.shared.python.pose_interchange.frame_transport import FixedFrameTransport

pytestmark = pytest.mark.unit


def test_offset_twist_and_dual_wrench_preserve_power() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [1, 0, 0]
    frame = FixedFrameTransport("source", "target", transform)
    twist = np.array([0, 0, 2, 0, 0, 0.0])
    np.testing.assert_allclose(frame.twist(twist), [0, 0, 2, 0, -2, 0])
    wrench = np.array([1, 2, 3, 4, 5, 6.0])
    mapped = frame.wrench(wrench)
    np.testing.assert_allclose(mapped, [1, -4, 8, 4, 5, 6])
    assert mapped @ frame.twist(twist) == pytest.approx(wrench @ twist)
    np.testing.assert_allclose(frame.inverse().twist(frame.twist(twist)), twist)
    np.testing.assert_allclose(frame.inverse().wrench(mapped), wrench)


def test_pose_composition_and_input_ownership() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    frame = FixedFrameTransport("a", "b", transform)
    transform[:3, 3] = 0
    np.testing.assert_allclose(frame.pose(np.eye(4))[:3, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        frame.target_from_source[0, 0] = 5


@pytest.mark.parametrize(
    "bad", [np.zeros((4, 4)), np.diag([1, 1, -1, 1]), np.full((4, 4), np.nan)]
)
def test_rejects_nonrigid_transform(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        FixedFrameTransport("a", "b", bad)
