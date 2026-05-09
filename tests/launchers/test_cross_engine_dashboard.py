"""Tests for the Cross-Engine Perturbation Comparison Dashboard.

Covers:
- CrossEngineSimConfig default values and validation
- _StubEngine protocol compliance
- _run_headless() headless comparison path
- Qt window assembly helpers and GUI bootstrap
- main() CLI interface
- _build_arg_parser() argument handling
- CV summary dict structure
- _update_charts / chart data keys (via duck-type check on cv_summary)

All tests are unit-level and headless-safe.  PyQt6-dependent tests are
skipped if PyQt6 is not installed.
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import patch

import numpy as np
import pytest
from src.launchers.cross_engine_dashboard import (
    CrossEngineSimConfig,
    _build_arg_parser,
    _run_headless,
    _StubEngine,
)

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# CrossEngineSimConfig
# ---------------------------------------------------------------------------


class TestCrossEngineSimConfig:
    """Tests for CrossEngineSimConfig construction and validation."""

    def test_cross_engine_dashboard_defaults(self) -> None:
        """Default configuration must have sensible values."""
        cfg = CrossEngineSimConfig()
        assert cfg.t_end == pytest.approx(1.5)
        assert cfg.dt == pytest.approx(0.01)
        assert cfg.noise_amplitude == pytest.approx(0.1)
        assert cfg.n_trials == 10
        assert cfg.seed == 42

    def test_cross_engine_dashboard_custom_values(self) -> None:
        """Custom values must be stored correctly."""
        cfg = CrossEngineSimConfig(
            t_end=2.0, dt=0.005, noise_amplitude=0.5, n_trials=5, seed=7
        )
        assert cfg.t_end == pytest.approx(2.0)
        assert cfg.dt == pytest.approx(0.005)
        assert cfg.noise_amplitude == pytest.approx(0.5)
        assert cfg.n_trials == 5
        assert cfg.seed == 7

    def test_invalid_dt_zero(self) -> None:
        """dt=0 must raise ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            CrossEngineSimConfig(dt=0.0)

    def test_invalid_dt_negative(self) -> None:
        """Negative dt must raise ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            CrossEngineSimConfig(dt=-0.1)

    def test_invalid_t_end_zero(self) -> None:
        """t_end=0 must raise ValueError."""
        with pytest.raises(ValueError, match="t_end must be positive"):
            CrossEngineSimConfig(t_end=0.0)

    def test_t_end_must_exceed_dt(self) -> None:
        """t_end <= dt must raise ValueError."""
        with pytest.raises(ValueError, match="t_end .* must be greater than dt"):
            CrossEngineSimConfig(t_end=0.005, dt=0.01)

    def test_invalid_noise_amplitude_negative(self) -> None:
        """Negative noise_amplitude must raise ValueError."""
        with pytest.raises(ValueError, match="noise_amplitude must be non-negative"):
            CrossEngineSimConfig(noise_amplitude=-0.1)

    def test_invalid_n_trials_zero(self) -> None:
        """n_trials=0 must raise ValueError."""
        with pytest.raises(ValueError, match="n_trials must be positive"):
            CrossEngineSimConfig(n_trials=0)


# ---------------------------------------------------------------------------
# _StubEngine
# ---------------------------------------------------------------------------


class TestStubEngine:
    """Tests for the _StubEngine stub implementation."""

    def test_cross_engine_dashboard_instantiation(self) -> None:
        """_StubEngine can be created with a non-empty name."""
        eng = _StubEngine("test_engine")
        assert eng is not None

    def test_empty_name_raises(self) -> None:
        """Empty name must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            _StubEngine("")

    def test_reset_zeroes_state(self) -> None:
        """After reset, positions and velocities must be zero."""
        eng = _StubEngine("eng", n_dof=3)
        eng._v = np.ones(3)
        eng._q = np.ones(3)
        eng.reset()
        q, v = eng.get_state()
        assert np.allclose(q, 0.0)
        assert np.allclose(v, 0.0)

    def test_get_state_shapes(self) -> None:
        """get_state() must return two arrays of length n_dof."""
        eng = _StubEngine("eng", n_dof=2)
        q, v = eng.get_state()
        assert q.shape == (2,)
        assert v.shape == (2,)

    def test_step_advances_state(self) -> None:
        """After set_control and step, velocity must be non-zero."""
        eng = _StubEngine("eng", n_dof=2)
        eng.set_control(np.array([1.0, 1.0]))
        eng.step(dt=0.01)
        _, v = eng.get_state()
        assert np.any(v != 0.0)

    def test_cross_engine_dashboard_protocol_compliance(self) -> None:
        """_StubEngine must satisfy the SteppableEngine protocol."""
        from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
            SteppableEngine,
        )

        eng = _StubEngine("eng")
        assert isinstance(eng, SteppableEngine)


# ---------------------------------------------------------------------------
# _run_headless
# ---------------------------------------------------------------------------


class TestRunHeadless:
    """Tests for the headless comparison runner."""

    def test_single_stub_engine(self) -> None:
        """Headless run with a single stub engine must return valid CV keys."""
        config = CrossEngineSimConfig(t_end=0.5, dt=0.01, n_trials=2)
        cv_summary = _run_headless(["pendulum_stub"], config)
        expected_keys = {
            "cv_total_energy_final",
            "cv_end_effector_speed_final",
            "cv_peak_end_effector_speed",
        }
        assert set(cv_summary.keys()) == expected_keys

    def test_two_stub_engines(self) -> None:
        """Headless run with two stub engines must return finite CV values."""
        config = CrossEngineSimConfig(t_end=0.3, dt=0.01, n_trials=2)
        cv_summary = _run_headless(["pendulum_stub", "pendulum_stub_2"], config)
        for val in cv_summary.values():
            assert np.isfinite(val)

    def test_empty_engine_list_raises(self) -> None:
        """Empty engine list must raise ValueError."""
        config = CrossEngineSimConfig()
        with pytest.raises(ValueError, match="At least one engine"):
            _run_headless([], config)

    def test_cv_values_are_non_negative(self) -> None:
        """All CV values must be >= 0."""
        config = CrossEngineSimConfig(t_end=0.5, dt=0.01, n_trials=3)
        cv_summary = _run_headless(["pendulum_stub"], config)
        for val in cv_summary.values():
            assert val >= 0.0


# ---------------------------------------------------------------------------
# _build_arg_parser
# ---------------------------------------------------------------------------


class TestArgParser:
    """Tests for the CLI argument parser."""

    def test_cross_engine_dashboard_defaults(self) -> None:
        """Parser defaults must match CrossEngineSimConfig defaults."""
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.no_gui is False
        assert args.engines == "pendulum_stub"
        assert args.n_trials == 10
        assert args.amplitude == pytest.approx(0.1)
        assert args.t_end == pytest.approx(1.5)
        assert args.dt == pytest.approx(0.01)

    def test_no_gui_flag(self) -> None:
        """--no-gui flag must be parsed correctly."""
        parser = _build_arg_parser()
        args = parser.parse_args(["--no-gui"])
        assert args.no_gui is True

    def test_engines_argument(self) -> None:
        """--engines must accept comma-separated engine names."""
        parser = _build_arg_parser()
        args = parser.parse_args(["--engines", "mujoco,pinocchio"])
        assert args.engines == "mujoco,pinocchio"

    def test_n_trials_argument(self) -> None:
        """--n-trials must be parsed as an integer."""
        parser = _build_arg_parser()
        args = parser.parse_args(["--n-trials", "20"])
        assert args.n_trials == 20


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() entry point."""

    def test_main_no_gui(self, caplog: pytest.LogCaptureFixture) -> None:
        """main() with --no-gui must run without error and log CV summary."""
        from src.launchers.cross_engine_dashboard import main

        with caplog.at_level(logging.INFO):
            main(
                [
                    "--no-gui",
                    "--engines",
                    "pendulum_stub",
                    "--n-trials",
                    "2",
                    "--t-end",
                    "0.5",
                    "--dt",
                    "0.01",
                ]
            )

        assert any("CV Summary" in record.message for record in caplog.records)

    def test_main_no_gui_invalid_engines(self) -> None:
        """main() with invalid engine that falls back to stub must not crash."""
        from src.launchers.cross_engine_dashboard import main

        # 'nonexistent_engine' triggers _build_engine fallback to stub
        main(
            [
                "--no-gui",
                "--engines",
                "nonexistent_engine",
                "--n-trials",
                "2",
                "--t-end",
                "0.5",
                "--dt",
                "0.01",
            ]
        )

    def test_main_empty_engines_exits(self) -> None:
        """main() with empty --engines must call sys.exit(1)."""
        from src.launchers.cross_engine_dashboard import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--no-gui", "--engines", ""])
        assert exc_info.value.code == 1

    def test_main_gui_mode_no_qt(self) -> None:
        """main() GUI mode falls back to headless if PyQt6 is absent."""
        from src.launchers.cross_engine_dashboard import main

        with patch.dict(
            sys.modules,
            {"PyQt6": None, "PyQt6.QtWidgets": None, "PyQt6.QtCore": None},
        ):
            # Should fall back to headless without raising
            main(
                [
                    "--engines",
                    "pendulum_stub",
                    "--n-trials",
                    "2",
                    "--t-end",
                    "0.5",
                    "--dt",
                    "0.01",
                ]
            )
