import unittest
import numpy as np
from pathlib import Path
from src.motion_capture.c3d_simscape_preview import (
    canonicalize_rotation,
    check_matlab_availability,
    match_motion,
)


class TestC3DSimscapePreview(unittest.TestCase):
    def test_canonicalize_rotation(self) -> None:
        # Test valid quaternion
        q = np.array([2.0, 0.0, 0.0, 0.0])
        q_norm = canonicalize_rotation(q)
        np.testing.assert_array_almost_equal(q_norm, np.array([1.0, 0.0, 0.0, 0.0]))

        # Test invalid shape
        with self.assertRaises(ValueError):
            canonicalize_rotation(np.array([1.0, 0.0, 0.0]))

    def test_match_motion_no_matlab(self) -> None:
        # Should return fallback diagnostics without failing
        markers = {"TEST": np.zeros((3, 10))}
        res = match_motion(markers, matlab_available=False)
        self.assertFalse(res["matched"])
        self.assertIn("rotations", res)
