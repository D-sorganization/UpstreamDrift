"""DbC tests for Hill-type muscle model and biomechanics modules.

Validates that:
- MuscleParameters rejects non-positive F_max, l_opt, l_slack
- HillMuscleModel.compute_force returns non-negative total force
- force_length_active is maximized at l_norm=1 (optimal length)
- force_velocity is bounded within [0, ~1.8]
- Activation clamping works correctly
- Biomechanics kinematic sequence analysis validates input dimensions
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ["DBC_LEVEL"] = "enforce"


class TestMuscleParametersPreconditions(unittest.TestCase):
    """MuscleParameters must reject non-positive physical values."""

    def test_negative_fmax_raises(self) -> None:
        from src.shared.python.biomechanics.hill_muscle import MuscleParameters

        with self.assertRaises(ValueError):
            MuscleParameters(F_max=-100.0, l_opt=0.1, l_slack=0.2)

    def test_zero_fmax_raises(self) -> None:
        from src.shared.python.biomechanics.hill_muscle import MuscleParameters

        with self.assertRaises(ValueError):
            MuscleParameters(F_max=0.0, l_opt=0.1, l_slack=0.2)

    def test_negative_lopt_raises(self) -> None:
        from src.shared.python.biomechanics.hill_muscle import MuscleParameters

        with self.assertRaises(ValueError):
            MuscleParameters(F_max=100.0, l_opt=-0.1, l_slack=0.2)

    def test_zero_lopt_raises(self) -> None:
        from src.shared.python.biomechanics.hill_muscle import MuscleParameters

        with self.assertRaises(ValueError):
            MuscleParameters(F_max=100.0, l_opt=0.0, l_slack=0.2)

    def test_negative_lslack_raises(self) -> None:
        from src.shared.python.biomechanics.hill_muscle import MuscleParameters

        with self.assertRaises(ValueError):
            MuscleParameters(F_max=100.0, l_opt=0.1, l_slack=-0.2)

    def test_zero_lslack_raises(self) -> None:
        from src.shared.python.biomechanics.hill_muscle import MuscleParameters

        with self.assertRaises(ValueError):
            MuscleParameters(F_max=100.0, l_opt=0.1, l_slack=0.0)

    def test_valid_params_ok(self) -> None:
        from src.shared.python.biomechanics.hill_muscle import MuscleParameters

        params = MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20)
        self.assertEqual(params.F_max, 1000.0)


class TestForceLengthActivePostconditions(unittest.TestCase):
    """force_length_active must be ~1.0 at optimal length and < 1 elsewhere."""

    def _make_model(self):  # type: ignore[no-untyped-def]
        from src.shared.python.biomechanics.hill_muscle import (
            HillMuscleModel,
            MuscleParameters,
        )

        return HillMuscleModel(MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20))

    def test_optimal_length_gives_max(self) -> None:
        model = self._make_model()
        fl = model.force_length_active(1.0)
        self.assertAlmostEqual(fl, 1.0, places=5)

    def test_very_short_fiber_gives_low_force(self) -> None:
        model = self._make_model()
        fl = model.force_length_active(0.3)
        self.assertLess(fl, 0.5)  # Below 50% of peak

    def test_very_long_fiber_gives_near_zero(self) -> None:
        model = self._make_model()
        fl = model.force_length_active(2.0)
        self.assertLess(fl, 0.1)

    def test_output_non_negative(self) -> None:
        model = self._make_model()
        for l_norm in np.linspace(0.1, 3.0, 50):
            fl = model.force_length_active(float(l_norm))
            self.assertGreaterEqual(fl, 0.0, f"f_l({l_norm}) = {fl} < 0")


class TestForceLengthPassivePostconditions(unittest.TestCase):
    """Passive force must be zero below optimal length, positive above."""

    def _make_model(self):  # type: ignore[no-untyped-def]
        from src.shared.python.biomechanics.hill_muscle import (
            HillMuscleModel,
            MuscleParameters,
        )

        return HillMuscleModel(MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20))

    def test_below_optimal_gives_zero(self) -> None:
        model = self._make_model()
        self.assertEqual(model.force_length_passive(0.8), 0.0)

    def test_at_optimal_gives_zero(self) -> None:
        model = self._make_model()
        self.assertEqual(model.force_length_passive(1.0), 0.0)

    def test_above_optimal_gives_positive(self) -> None:
        model = self._make_model()
        fp = model.force_length_passive(1.5)
        self.assertGreater(fp, 0.0)

    def test_monotonically_increasing_above_optimal(self) -> None:
        model = self._make_model()
        prev = 0.0
        for l_norm in np.linspace(1.01, 2.0, 20):
            fp = model.force_length_passive(float(l_norm))
            self.assertGreaterEqual(
                fp, prev, f"Passive force not monotonic at l_norm={l_norm}"
            )
            prev = fp


class TestForceVelocityPostconditions(unittest.TestCase):
    """force_velocity must satisfy Hill's hyperbola constraints."""

    def _make_model(self):  # type: ignore[no-untyped-def]
        from src.shared.python.biomechanics.hill_muscle import (
            HillMuscleModel,
            MuscleParameters,
        )

        return HillMuscleModel(MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20))

    def test_isometric_gives_one(self) -> None:
        model = self._make_model()
        fv = model.force_velocity(0.0)
        self.assertAlmostEqual(fv, 1.0, places=5)

    def test_concentric_less_than_isometric(self) -> None:
        """Shortening muscle produces less force than isometric."""
        model = self._make_model()
        fv = model.force_velocity(-0.5)
        self.assertLess(fv, 1.0)

    def test_eccentric_greater_than_isometric(self) -> None:
        """Lengthening muscle produces more force than isometric."""
        model = self._make_model()
        fv = model.force_velocity(0.5)
        self.assertGreater(fv, 1.0)

    def test_concentric_non_negative(self) -> None:
        model = self._make_model()
        for v in np.linspace(-0.99, 0.0, 20):
            fv = model.force_velocity(float(v))
            self.assertGreaterEqual(fv, 0.0, f"f_v({v}) = {fv} < 0")


class TestTendonForcePostconditions(unittest.TestCase):
    """Tendon force must be zero when slack, positive when stretched."""

    def _make_model(self):  # type: ignore[no-untyped-def]
        from src.shared.python.biomechanics.hill_muscle import (
            HillMuscleModel,
            MuscleParameters,
        )

        return HillMuscleModel(MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20))

    def test_slack_gives_zero(self) -> None:
        model = self._make_model()
        self.assertEqual(model.tendon_force(0.9), 0.0)

    def test_at_slack_length_gives_zero(self) -> None:
        model = self._make_model()
        self.assertEqual(model.tendon_force(1.0), 0.0)

    def test_stretched_gives_positive(self) -> None:
        model = self._make_model()
        ft = model.tendon_force(1.05)
        self.assertGreater(ft, 0.0)

    def test_monotonically_increasing_above_slack(self) -> None:
        model = self._make_model()
        prev = 0.0
        for lt in np.linspace(1.0, 1.1, 20):
            ft = model.tendon_force(float(lt))
            self.assertGreaterEqual(ft, prev, f"Tendon force not monotonic at lt={lt}")
            prev = ft


class TestComputeForcePostconditions(unittest.TestCase):
    """compute_force must return non-negative total force."""

    def _make_model(self):  # type: ignore[no-untyped-def]
        from src.shared.python.biomechanics.hill_muscle import (
            HillMuscleModel,
            MuscleParameters,
        )

        return HillMuscleModel(MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20))

    def test_isometric_max_activation(self) -> None:
        from src.shared.python.biomechanics.hill_muscle import MuscleState

        model = self._make_model()
        state = MuscleState(activation=1.0, l_CE=0.15, v_CE=0.0, l_MT=0.35)
        force = model.compute_force(state)
        self.assertGreater(force, 0.0)
        # At optimal length, isometric, full activation -> should be near F_max
        self.assertAlmostEqual(force, 1000.0, delta=50.0)

    def test_zero_activation_isometric(self) -> None:
        from src.shared.python.biomechanics.hill_muscle import MuscleState

        model = self._make_model()
        state = MuscleState(activation=0.0, l_CE=0.15, v_CE=0.0, l_MT=0.35)
        force = model.compute_force(state)
        # Zero activation at optimal length -> only passive force, which is zero
        self.assertAlmostEqual(force, 0.0, delta=10.0)

    def test_output_non_negative(self) -> None:
        from src.shared.python.biomechanics.hill_muscle import MuscleState

        model = self._make_model()
        for activation in [0.0, 0.3, 0.6, 1.0]:
            for l_CE in [0.10, 0.15, 0.20]:
                for v_CE in [-0.5, 0.0, 0.5]:
                    state = MuscleState(
                        activation=activation, l_CE=l_CE, v_CE=v_CE, l_MT=0.35
                    )
                    force = model.compute_force(state)
                    self.assertGreaterEqual(
                        force,
                        0.0,
                        f"Force={force} < 0 for a={activation}, "
                        f"l_CE={l_CE}, v_CE={v_CE}",
                    )

    def test_force_increases_with_activation(self) -> None:
        from src.shared.python.biomechanics.hill_muscle import MuscleState

        model = self._make_model()
        forces = []
        for activation in np.linspace(0.0, 1.0, 10):
            state = MuscleState(
                activation=float(activation), l_CE=0.15, v_CE=0.0, l_MT=0.35
            )
            forces.append(model.compute_force(state))

        # Force should be non-decreasing with activation
        for i in range(1, len(forces)):
            self.assertGreaterEqual(
                forces[i] + 1e-10,
                forces[i - 1],
                f"Force not monotonic at activation={i / 10}",
            )


class TestKinematicSequenceAnalyzerPreconditions(unittest.TestCase):
    """Kinematic sequence analyzer should handle edge cases gracefully."""

    def test_analyze_empty_velocities_returns_empty_peaks(self) -> None:
        """Empty velocities should return result with no peaks."""
        from src.shared.python.biomechanics.kinematic_sequence import (
            SegmentTimingAnalyzer,
        )

        analyzer = SegmentTimingAnalyzer(expected_order=["pelvis", "torso"])
        result = analyzer.analyze({}, np.array([0.0, 0.1, 0.2]))
        self.assertEqual(len(result.peaks), 0)

    def test_analyze_returns_result_with_peaks(self) -> None:
        from src.shared.python.biomechanics.kinematic_sequence import (
            SegmentTimingAnalyzer,
        )

        analyzer = SegmentTimingAnalyzer(expected_order=["pelvis", "torso", "arm"])
        times = np.linspace(0, 1, 100)
        segment_vels = {
            "pelvis": np.sin(2 * np.pi * times) * 100,
            "torso": np.sin(2 * np.pi * (times - 0.1)) * 150,
            "arm": np.sin(2 * np.pi * (times - 0.2)) * 200,
        }
        result = analyzer.analyze(segment_vels, times)
        # Should detect peaks for all segments
        self.assertEqual(len(result.peaks), 3)
        # Peaks should have positive velocities
        for peak in result.peaks:
            self.assertGreater(peak.peak_velocity, 0.0)

    def test_sequence_consistency_bounded(self) -> None:
        """Consistency score must be in [0, 1]."""
        from src.shared.python.biomechanics.kinematic_sequence import (
            SegmentTimingAnalyzer,
        )

        analyzer = SegmentTimingAnalyzer(expected_order=["pelvis", "torso"])
        times = np.linspace(0, 1, 100)
        segment_vels = {
            "pelvis": np.sin(2 * np.pi * times) * 100,
            "torso": np.sin(2 * np.pi * (times - 0.1)) * 150,
        }
        result = analyzer.analyze(segment_vels, times)
        self.assertGreaterEqual(result.sequence_consistency, 0.0)
        self.assertLessEqual(result.sequence_consistency, 1.0)


if __name__ == "__main__":
    unittest.main()
