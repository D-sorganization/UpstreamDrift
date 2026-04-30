"""DbC runtime contract tests for Wave 6 modules.

Tests contracts in:
- biomechanics.hill_muscle (activation range, force postcondition)
- biomechanics.muscle_analysis (data shape, n_synergies, VAF postcondition)
- biomechanics.swing_comparison (segment_indices, DTW postconditions)
- signal_toolkit.fitting (non-empty signal, RMSE postcondition)
- signal_toolkit.io (time/values matching, non-empty)
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.shared.python.core.contracts import PreconditionError

# ── Helper factories ───────────────────────────────────────────────


def _make_signal(
    duration: float = 1.0,
    fs: float = 100.0,
) -> Any:
    """Create a simple sinusoidal test signal."""
    from src.shared.python.signal_toolkit.core import Signal

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    values = np.sin(2 * np.pi * t)
    return Signal(time=t, values=values, name="test_sin", units="V")


# ── HillMuscleModel contracts ──────────────────────────────────────


class TestHillMusclePreconditions(unittest.TestCase):
    """Test require() contracts on HillMuscleModel.compute_force."""

    def _make_model_and_state(self, activation: float = 0.5) -> tuple[Any, Any]:
        from src.shared.python.biomechanics.hill_muscle import (
            HillMuscleModel,
            MuscleParameters,
            MuscleState,
        )

        params = MuscleParameters(
            F_max=1000.0,
            l_opt=0.1,
            l_slack=0.2,
            v_max=10.0,
            pennation_angle=0.0,
            damping=0.01,
        )
        model = HillMuscleModel(params)
        state = MuscleState(activation=activation, l_CE=0.1, v_CE=0.0)
        return model, state

    def test_activation_negative_rejected(self) -> None:
        model, state = self._make_model_and_state(activation=-0.1)
        with self.assertRaises(PreconditionError):
            model.compute_force(state)

    def test_activation_above_one_rejected(self) -> None:
        model, state = self._make_model_and_state(activation=1.5)
        with self.assertRaises(PreconditionError):
            model.compute_force(state)

    def test_activation_zero_accepted(self) -> None:
        model, state = self._make_model_and_state(activation=0.0)
        force = model.compute_force(state)
        self.assertGreaterEqual(force, 0.0)

    def test_activation_one_accepted(self) -> None:
        model, state = self._make_model_and_state(activation=1.0)
        force = model.compute_force(state)
        self.assertGreaterEqual(force, 0.0)

    def test_valid_activation_returns_nonnegative(self) -> None:
        model, state = self._make_model_and_state(activation=0.5)
        force = model.compute_force(state)
        self.assertGreaterEqual(force, 0.0)


class TestHillMusclePostconditions(unittest.TestCase):
    """Test ensure() contracts on HillMuscleModel."""

    def test_force_always_nonneg(self) -> None:
        """Force postcondition: result >= 0."""
        from src.shared.python.biomechanics.hill_muscle import (
            HillMuscleModel,
            MuscleParameters,
            MuscleState,
        )

        params = MuscleParameters(
            F_max=1000.0,
            l_opt=0.1,
            l_slack=0.2,
            v_max=10.0,
            pennation_angle=0.0,
            damping=0.01,
        )
        model = HillMuscleModel(params)

        for act in [0.0, 0.25, 0.5, 0.75, 1.0]:
            state = MuscleState(activation=act, l_CE=0.1, v_CE=0.0)
            force = model.compute_force(state)
            self.assertGreaterEqual(force, 0.0, f"activation={act}")


# ── MuscleSynergyAnalyzer contracts ────────────────────────────────


class TestMuscleSynergyPreconditions(unittest.TestCase):
    """Test require() contracts on MuscleSynergyAnalyzer."""

    def test_1d_data_rejected(self) -> None:
        from src.shared.python.biomechanics.muscle_analysis import (
            MuscleSynergyAnalyzer,
        )

        with self.assertRaises(PreconditionError):
            MuscleSynergyAnalyzer(np.array([1, 2, 3]))

    def test_empty_data_rejected(self) -> None:
        from src.shared.python.biomechanics.muscle_analysis import (
            MuscleSynergyAnalyzer,
        )

        with self.assertRaises(PreconditionError):
            MuscleSynergyAnalyzer(np.empty((0, 3)))

    def test_valid_2d_data_accepted(self) -> None:
        from src.shared.python.biomechanics.muscle_analysis import (
            MuscleSynergyAnalyzer,
        )

        data = np.random.rand(50, 4)
        analyzer = MuscleSynergyAnalyzer(data)
        self.assertEqual(analyzer.n_muscles, 4)
        self.assertEqual(analyzer.n_samples, 50)

    def test_n_synergies_zero_rejected(self) -> None:
        from src.shared.python.biomechanics.muscle_analysis import (
            MuscleSynergyAnalyzer,
        )

        data = np.random.rand(50, 4)
        analyzer = MuscleSynergyAnalyzer(data)
        with self.assertRaises(PreconditionError):
            analyzer.extract_synergies(n_synergies=0)

    def test_n_synergies_exceeds_muscles_rejected(self) -> None:
        from src.shared.python.biomechanics.muscle_analysis import (
            MuscleSynergyAnalyzer,
        )

        data = np.random.rand(50, 4)
        analyzer = MuscleSynergyAnalyzer(data)
        with self.assertRaises(PreconditionError):
            analyzer.extract_synergies(n_synergies=5)


class TestMuscleSynergyPostconditions(unittest.TestCase):
    """Test ensure() for VAF in [0, 1]."""

    def test_vaf_in_range(self) -> None:
        try:
            from src.shared.python.biomechanics.muscle_analysis import (
                MuscleSynergyAnalyzer,
            )
        except ImportError:
            self.skipTest("sklearn not available")

        # Create structured data with clear synergies
        np.random.seed(42)
        W = np.random.rand(4, 2)
        H = np.random.rand(2, 100)
        data = np.abs(W @ H).T  # Shape: (100, 4), non-negative
        analyzer = MuscleSynergyAnalyzer(data)

        try:
            result = analyzer.extract_synergies(n_synergies=2)
            self.assertGreaterEqual(result.vaf, 0.0)
            self.assertLessEqual(result.vaf, 1.0 + 1e-6)
        except ImportError:
            self.skipTest("sklearn not available")


# ── SwingComparator contracts ──────────────────────────────────────


class TestSwingComparatorPreconditions(unittest.TestCase):
    """Test require() contracts on SwingComparator."""

    def test_compare_peak_speeds_empty_indices_rejected(self) -> None:
        from unittest.mock import MagicMock

        from src.shared.python.biomechanics.swing_comparison import SwingComparator

        # Mock the analyzers
        ref = MagicMock()
        stu = MagicMock()
        ref.joint_velocities = np.random.rand(100, 5)
        stu.joint_velocities = np.random.rand(100, 5)

        comparator = SwingComparator.__new__(SwingComparator)
        comparator.ref = ref
        comparator.student = stu

        with self.assertRaises(PreconditionError):
            comparator.compare_peak_speeds({})


class TestSwingComparatorPostconditions(unittest.TestCase):
    """Test ensure() postconditions on DTW similarity."""

    def test_dtw_similarity_in_range(self) -> None:
        """DTW similarity_score should be in [0, 100]."""
        from unittest.mock import MagicMock, patch

        from src.shared.python.biomechanics.swing_comparison import SwingComparator

        ref = MagicMock()
        stu = MagicMock()
        ref.joint_velocities = np.random.rand(100, 5)
        stu.joint_velocities = np.random.rand(100, 5)

        comparator = SwingComparator.__new__(SwingComparator)
        comparator.ref = ref
        comparator.student = stu

        # Patch DTW computation to return known values
        with (
            patch(
                "src.shared.python.signal_toolkit.signal_processing.compute_dtw_path",
                return_value=(5.0, [(i, i) for i in range(100)]),
            ),
            patch(
                "src.shared.python.data_io.common_utils.normalize_z_score",
                side_effect=lambda x, eps: x,
            ),
        ):
            result = comparator.compute_kinematic_similarity(0)
            if result is not None:
                self.assertGreaterEqual(result.similarity_score, 0.0)
                self.assertLessEqual(result.similarity_score, 100.0)
                self.assertGreaterEqual(result.distance, 0.0)


# ── Fitting contracts ──────────────────────────────────────────────


class TestSinusoidFitterPreconditions(unittest.TestCase):
    """Test require() for non-empty signal in SinusoidFitter."""

    def test_empty_signal_rejected(self) -> None:
        from src.shared.python.signal_toolkit.core import Signal
        from src.shared.python.signal_toolkit.fitting import SinusoidFitter

        empty_sig = Signal(
            time=np.array([]),
            values=np.array([]),
            name="empty",
        )
        fitter = SinusoidFitter()
        with self.assertRaises(PreconditionError):
            fitter.fit(empty_sig)

    def test_valid_signal_accepted(self) -> None:
        from src.shared.python.signal_toolkit.fitting import SinusoidFitter

        sig = _make_signal(duration=1.0, fs=100.0)
        fitter = SinusoidFitter()
        result = fitter.fit(sig)
        self.assertGreaterEqual(result.rmse, 0.0)


class TestExponentialFitterPreconditions(unittest.TestCase):
    """Test require() for non-empty signal in ExponentialFitter."""

    def test_empty_signal_rejected(self) -> None:
        from src.shared.python.signal_toolkit.core import Signal
        from src.shared.python.signal_toolkit.fitting import ExponentialFitter

        empty_sig = Signal(
            time=np.array([]),
            values=np.array([]),
            name="empty",
        )
        fitter = ExponentialFitter()
        with self.assertRaises(PreconditionError):
            fitter.fit_decay(empty_sig)

    def test_valid_decay_accepted(self) -> None:
        from src.shared.python.signal_toolkit.core import Signal
        from src.shared.python.signal_toolkit.fitting import ExponentialFitter

        t = np.linspace(0, 5, 200)
        values = 10.0 * np.exp(-0.5 * t) + 1.0
        sig = Signal(time=t, values=values, name="decay")
        fitter = ExponentialFitter()
        result = fitter.fit_decay(sig)
        self.assertGreaterEqual(result.rmse, 0.0)


class TestLinearFitterPreconditions(unittest.TestCase):
    """Test require() for non-empty signal in LinearFitter."""

    def test_empty_signal_rejected(self) -> None:
        from src.shared.python.signal_toolkit.core import Signal
        from src.shared.python.signal_toolkit.fitting import LinearFitter

        empty_sig = Signal(
            time=np.array([]),
            values=np.array([]),
            name="empty",
        )
        fitter = LinearFitter()
        with self.assertRaises(PreconditionError):
            fitter.fit(empty_sig)

    def test_valid_linear_accepted(self) -> None:
        from src.shared.python.signal_toolkit.core import Signal
        from src.shared.python.signal_toolkit.fitting import LinearFitter

        t = np.linspace(0, 10, 100)
        values = 2.5 * t + 1.0
        sig = Signal(time=t, values=values, name="linear")
        fitter = LinearFitter()
        result = fitter.fit(sig)
        self.assertGreaterEqual(result.rmse, 0.0)
        self.assertAlmostEqual(result.r_squared, 1.0, places=5)


class TestPolynomialFitterPreconditions(unittest.TestCase):
    """Test require() for non-empty signal and order >= 0."""

    def test_empty_signal_rejected(self) -> None:
        from src.shared.python.signal_toolkit.core import Signal
        from src.shared.python.signal_toolkit.fitting import PolynomialFitter

        empty_sig = Signal(
            time=np.array([]),
            values=np.array([]),
            name="empty",
        )
        fitter = PolynomialFitter()
        with self.assertRaises(PreconditionError):
            fitter.fit(empty_sig)

    def test_negative_order_rejected(self) -> None:
        from src.shared.python.signal_toolkit.fitting import PolynomialFitter

        sig = _make_signal()
        fitter = PolynomialFitter()
        with self.assertRaises(PreconditionError):
            fitter.fit(sig, order=-1)

    def test_valid_polynomial_accepted(self) -> None:
        from src.shared.python.signal_toolkit.core import Signal
        from src.shared.python.signal_toolkit.fitting import PolynomialFitter

        t = np.linspace(0, 5, 100)
        values = 3 * t**2 - 2 * t + 1
        sig = Signal(time=t, values=values, name="quadratic")
        fitter = PolynomialFitter()
        result = fitter.fit(sig, order=2)
        self.assertGreaterEqual(result.rmse, 0.0)


class TestFittingPostconditions(unittest.TestCase):
    """Test ensure() postconditions for RMSE >= 0."""

    def test_sinusoid_rmse_nonneg(self) -> None:
        from src.shared.python.signal_toolkit.fitting import SinusoidFitter

        sig = _make_signal(duration=1.0, fs=100.0)
        result = SinusoidFitter().fit(sig)
        self.assertGreaterEqual(result.rmse, 0.0)


# ── Signal I/O contracts ──────────────────────────────────────────


class TestSignalImporterPreconditions(unittest.TestCase):
    """Test require() contracts on SignalImporter.from_numpy."""

    def test_empty_time_rejected(self) -> None:
        from src.shared.python.signal_toolkit.io import SignalImporter

        with self.assertRaises(PreconditionError):
            SignalImporter.from_numpy(
                time=np.array([]),
                values=np.array([]),
            )

    def test_mismatched_lengths_rejected(self) -> None:
        from src.shared.python.signal_toolkit.io import SignalImporter

        with self.assertRaises(PreconditionError):
            SignalImporter.from_numpy(
                time=np.array([0.0, 1.0, 2.0]),
                values=np.array([1.0, 2.0]),
            )

    def test_valid_arrays_accepted(self) -> None:
        from src.shared.python.signal_toolkit.io import SignalImporter

        t = np.linspace(0, 1, 100)
        v = np.sin(2 * np.pi * t)
        sig = SignalImporter.from_numpy(time=t, values=v, name="test")
        self.assertEqual(len(sig.time), 100)
        self.assertEqual(len(sig.values), 100)

    def test_matching_single_element_accepted(self) -> None:
        from src.shared.python.signal_toolkit.io import SignalImporter

        sig = SignalImporter.from_numpy(
            time=np.array([0.0]),
            values=np.array([1.0]),
        )
        self.assertEqual(len(sig.time), 1)


if __name__ == "__main__":
    unittest.main()
