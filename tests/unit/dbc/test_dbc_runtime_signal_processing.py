"""Runtime DbC tests for signal processing module contracts.

Tests the require()/ensure() contracts added to:
- compute_coherence (preconditions: fs>0, non-empty, equal length)
- compute_spectral_arc_length (postcondition: SAL <= 0)
"""

from __future__ import annotations

import unittest

import numpy as np


class TestComputeCoherenceContracts(unittest.TestCase):
    """Verify require() contracts on compute_coherence."""

    def _coherence(self, x: np.ndarray, y: np.ndarray, fs: float) -> object:
        from src.shared.python.signal_toolkit.signal_processing import (
            compute_coherence,
        )

        return compute_coherence(x, y, fs)

    def test_valid_coherence(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)
        y = rng.standard_normal(256)
        freqs, coh = self._coherence(x, y, 100.0)  # type: ignore[misc]
        self.assertEqual(len(freqs), len(coh))
        self.assertTrue(np.all(coh >= 0))

    def test_zero_fs_raises(self) -> None:
        x = np.ones(100)
        with self.assertRaises((ValueError, Exception)):
            self._coherence(x, x, 0.0)

    def test_negative_fs_raises(self) -> None:
        x = np.ones(100)
        with self.assertRaises((ValueError, Exception)):
            self._coherence(x, x, -10.0)

    def test_empty_x_raises(self) -> None:
        with self.assertRaises((ValueError, Exception)):
            self._coherence(np.array([]), np.array([1.0]), 100.0)

    def test_empty_y_raises(self) -> None:
        with self.assertRaises((ValueError, Exception)):
            self._coherence(np.array([1.0]), np.array([]), 100.0)

    def test_mismatched_length_raises(self) -> None:
        with self.assertRaises((ValueError, Exception)):
            self._coherence(np.ones(100), np.ones(50), 100.0)


class TestSpectralArcLengthContracts(unittest.TestCase):
    """Verify ensure() postcondition on SAL."""

    def _sal(self, data: np.ndarray, fs: float) -> float:
        from src.shared.python.signal_toolkit.signal_processing import (
            compute_spectral_arc_length,
        )

        return compute_spectral_arc_length(data, fs)

    def test_sal_is_non_positive(self) -> None:
        """SAL is always <= 0 by definition (negated arc length)."""
        rng = np.random.default_rng(42)
        data = np.sin(2 * np.pi * 5.0 * np.arange(1000) / 500.0) + rng.normal(
            0, 0.1, 1000
        )
        result = self._sal(data, 500.0)
        self.assertLessEqual(result, 0.0)

    def test_sal_zero_for_empty(self) -> None:
        result = self._sal(np.array([]), 100.0)
        self.assertEqual(result, 0.0)

    def test_sal_smoother_signal_closer_to_zero(self) -> None:
        """A smoother signal should have SAL closer to 0 than a noisy one."""
        t = np.linspace(0, 1, 500)
        smooth = np.sin(2 * np.pi * 5 * t)
        noisy = smooth + np.random.default_rng(42).normal(0, 2, 500)
        sal_smooth = self._sal(smooth, 500.0)
        sal_noisy = self._sal(noisy, 500.0)
        # SAL is negative; smoother → closer to 0 → greater (less negative)
        self.assertGreater(sal_smooth, sal_noisy)

    def test_sal_fs_positive_raises_on_zero(self) -> None:
        with self.assertRaises((ValueError, Exception)):
            self._sal(np.ones(100), 0.0)

    def test_sal_fs_negative_raises(self) -> None:
        with self.assertRaises((ValueError, Exception)):
            self._sal(np.ones(100), -100.0)


class TestComputePSDContracts(unittest.TestCase):
    """Test existing @precondition decorators still work."""

    def test_psd_negative_fs_raises(self) -> None:
        from src.shared.python.signal_toolkit.signal_processing import compute_psd

        with self.assertRaises((ValueError, Exception)):
            compute_psd(np.ones(100), fs=-10.0)

    def test_psd_empty_data_raises(self) -> None:
        from src.shared.python.signal_toolkit.signal_processing import compute_psd

        with self.assertRaises((ValueError, Exception)):
            compute_psd(np.array([]), fs=100.0)

    def test_psd_valid_returns_tuple(self) -> None:
        from src.shared.python.signal_toolkit.signal_processing import compute_psd

        freqs, psd = compute_psd(np.random.default_rng(42).standard_normal(256), 100.0)
        self.assertEqual(len(freqs), len(psd))
        self.assertTrue(np.all(psd >= 0))


if __name__ == "__main__":
    unittest.main()
