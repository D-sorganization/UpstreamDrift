"""Fixed body attachments must follow rigid transforms with exact identities."""

import numpy as np
import pytest

from src.shared.python.motion_matching.marker_projection import project_markers

pytestmark = pytest.mark.unit


def test_rotation_translation_and_attachment_order() -> None:
    frame = np.array([[0.0, -1, 0, 3], [1, 0, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
    result = project_markers({"club": frame}, ["club", "club"], [[1, 0, 0], [0, 2, 0]])
    np.testing.assert_allclose(result, [[3, 5, 5], [1, 4, 5]])


@pytest.mark.parametrize("offsets", [[[0, 0]], [[0, 0, np.nan]], []])
def test_invalid_offsets_rejected(offsets: list) -> None:
    with pytest.raises(ValueError):
        project_markers({"club": np.eye(4)}, ["club"], offsets)


def test_missing_body_rejected() -> None:
    with pytest.raises(ValueError, match="body"):
        project_markers({}, ["club"], [[0, 0, 0]])


def test_invalid_transform_rejected() -> None:
    with pytest.raises(ValueError, match="transform"):
        project_markers({"club": np.zeros((4, 4))}, ["club"], [[0, 0, 0]])
