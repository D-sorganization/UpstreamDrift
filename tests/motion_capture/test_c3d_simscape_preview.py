import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.motion_capture.c3d_simscape_preview import (
    canonicalize_rotation,
    load_c3d_markers,
    match_motion,
    run_pipeline,
)
from tests.unit.sidekick.lab.bio._synthetic import _synthetic_c3d_dict
from tests.unit.sidekick.lab.bio.test_c3d_reader import _patched_reader

pytestmark = pytest.mark.unit


class TestC3DSimscapePreview(unittest.TestCase):
    def test_canonicalize_rotation(self) -> None:
        # Test valid quaternion
        q = np.array([2.0, 0.0, 0.0, 0.0])
        q_norm = canonicalize_rotation(q)
        np.testing.assert_array_almost_equal(q_norm, np.array([1.0, 0.0, 0.0, 0.0]))

        # Test invalid shape
        with self.assertRaises(ValueError):
            canonicalize_rotation(np.array([1.0, 0.0, 0.0]))

    def test_canonicalize_rotation_upcasts_integer_inputs(self) -> None:
        q = np.array([200, 200, 200, 200], dtype=np.int16)

        q_norm = canonicalize_rotation(q)

        np.testing.assert_allclose(q_norm, np.full(4, 0.5))
        self.assertTrue(np.issubdtype(q_norm.dtype, np.floating))

    def test_match_motion_no_matlab(self) -> None:
        # Should return fallback diagnostics without failing
        markers = {"TEST": np.zeros((3, 10))}
        res = match_motion(markers, matlab_available=False)
        self.assertFalse(res["matched"])
        self.assertIn("rotations", res)
        self.assertIn("reason", res)

    def test_match_motion_with_matlab_does_not_claim_success(self) -> None:
        markers = {"TEST": np.zeros((3, 10))}
        res = match_motion(markers, matlab_available=True)
        self.assertFalse(res["matched"])
        self.assertIn("reason", res)

    def test_load_c3d_markers_corrupt_file_raises(self) -> None:
        bad_path = Path(self._testMethodName + "_corrupt.c3d")
        bad_path.write_bytes(b"not-a-c3d")
        try:
            with self.assertRaises(ValueError):
                load_c3d_markers(str(bad_path))
        finally:
            bad_path.unlink(missing_ok=True)

    def test_load_c3d_markers_converts_mm_to_metres(self) -> None:
        pts = np.zeros((4, 1, 1), dtype=float)
        pts[0, 0, 0] = 500.0
        c3d_path = Path(self._testMethodName + "_mm.c3d")
        synthetic = _synthetic_c3d_dict(
            n_frames=1,
            n_markers=1,
            marker_names=["T"],
            point_data=pts,
            units="mm",
        )

        def reader_factory() -> type:
            class _Reader:
                def __init__(self, path: Path | str) -> None:
                    self._inner = _patched_reader(synthetic, Path(path))

                def get_metadata(self):
                    return self._inner.get_metadata()

                def points_dataframe(self, **kwargs):
                    return self._inner.points_dataframe(**kwargs)

            return _Reader

        c3d_path.write_bytes(b"\x00")
        try:
            with patch(
                "src.motion_capture.c3d_simscape_preview.c3d_reader_class",
                side_effect=reader_factory,
            ):
                result = load_c3d_markers(str(c3d_path))

            self.assertEqual(result.units, "m")
            self.assertEqual(result.source_units, "mm")
            self.assertAlmostEqual(result.markers["T"][0, 0], 0.5)
        finally:
            c3d_path.unlink(missing_ok=True)

    def test_run_pipeline_corrupt_c3d_raises(self) -> None:
        bad_path = Path(self._testMethodName + "_bad.c3d")
        bad_path.write_bytes(b"not-a-c3d")
        out_path = Path(self._testMethodName + "_out.txt")
        try:
            with self.assertRaises(ValueError):
                run_pipeline(str(bad_path), str(out_path))
        finally:
            bad_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
