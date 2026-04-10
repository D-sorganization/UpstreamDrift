"""Tests for pendulum GUI panel builder helpers."""

from __future__ import annotations

import math
import sys
from types import ModuleType

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def _install_fake_simulation_modules() -> None:
    class _FakeResultBase:
        pass

    for module_name in (
        "src.shared.python.pendulum_simulator.simulation",
        "src.shared.python.pendulum_simulator.simulation_triple",
        "src.shared.python.pendulum_simulator.simulation_golfer",
    ):
        module = ModuleType(module_name)
        module.make_polynomial_torque = lambda *args, **kwargs: (args, kwargs)
        module.run_simulation = lambda **kwargs: kwargs
        if module_name.endswith(".simulation"):
            module.SimulationResult = _FakeResultBase
        elif module_name.endswith(".simulation_triple"):
            module.TripleSimulationResult = _FakeResultBase
        else:
            module.GolferSimulationResult = _FakeResultBase
        sys.modules[module_name] = module


def _install_fake_perturbation_modules() -> None:
    package = ModuleType("src.shared.python.pendulum_simulator.perturbation")
    package.__path__ = []  # type: ignore[attr-defined]
    sys.modules["src.shared.python.pendulum_simulator.perturbation"] = package

    config_module = ModuleType(
        "src.shared.python.pendulum_simulator.perturbation.config"
    )

    class _FakePerturbationConfig:
        pass

    class _FakePerturbationSummary:
        pass

    config_module.PerturbationConfig = _FakePerturbationConfig
    config_module.PerturbationSummary = _FakePerturbationSummary
    sys.modules["src.shared.python.pendulum_simulator.perturbation.config"] = (
        config_module
    )

    analyzer_module = ModuleType(
        "src.shared.python.pendulum_simulator.pendulum_perturbation_analyzer"
    )

    class _FakeAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_base_torque_profile(self, *args, **kwargs) -> None:
            pass

        def run_batch(self, *args, **kwargs):
            return _FakePerturbationSummary()

    analyzer_module.PendulumPerturbationAnalyzer = _FakeAnalyzer
    sys.modules[
        "src.shared.python.pendulum_simulator.pendulum_perturbation_analyzer"
    ] = analyzer_module

    perturbation_analysis_module = ModuleType(
        "src.shared.python.pendulum_simulator.perturbation_analysis"
    )
    perturbation_analysis_module.variability_summary = lambda *args, **kwargs: None
    sys.modules["src.shared.python.pendulum_simulator.perturbation_analysis"] = (
        perturbation_analysis_module
    )


_install_fake_simulation_modules()
_install_fake_perturbation_modules()

from src.shared.python.pendulum_simulator.gui import panel_builders  # noqa: E402


class _FakeResult:
    def __init__(self, positions, velocities, times) -> None:
        self._positions = positions
        self._velocities = velocities
        self.t = np.asarray(times)
        self.n_steps = len(times)

    def positions_at(self, idx: int):
        return self._positions[idx]

    def joint_velocities_at(self, idx: int):
        return self._velocities[idx]


class _FakePendulum:
    def __init__(self) -> None:
        self.tilt = None
        self.azimuth = None

    def set_tilt_angle(self, value) -> None:
        self.tilt = value

    def set_view_azimuth(self, value) -> None:
        self.azimuth = value


class _FakeControls:
    PRESETS = {
        "Preset": (None, None, None, None, "1.0, 2.0", "3.0"),
    }

    def __init__(self, params: dict) -> None:
        self._params = params

    def get_params(self) -> dict:
        return self._params


class _FakeSimulationPanel:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.__dict__.update(kwargs)
        self.perturbation_panel = None
        self._settings_key = None

    def set_perturbation_panel(self, panel) -> None:
        self.perturbation_panel = panel


class _FakePerturbationPanel:
    def __init__(self) -> None:
        self.coeffs_source = None
        self.preset_names_source = None
        self.preset_coeffs = None
        self.simulate_fn = None
        self.extract_fn = None

    def set_coeffs_source(self, callback) -> None:
        self.coeffs_source = callback

    def set_preset_source(self, names_callback, coeffs_callback) -> None:
        self.preset_names_source = names_callback
        self.preset_coeffs = coeffs_callback

    def set_simulation_callbacks(self, simulate_fn, extract_fn) -> None:
        self.simulate_fn = simulate_fn
        self.extract_fn = extract_fn


class _FakeOptimizationWidget:
    def __init__(self, model_name: str, n_torque_params: int) -> None:
        self.model_name = model_name
        self.n_torque_params = n_torque_params


def test_helper_parsers_and_motion_extraction() -> None:
    assert panel_builders._parse_coefficients("1.0, 2.5, , 3") == [1.0, 2.5, 3.0]
    assert panel_builders._parse_coefficients("") == [0.0]
    assert panel_builders._chunk_coefficients(np.array([1, 2, 3, 4, 5]), 2) == [
        [1, 2],
        [3, 4, 5],
    ]

    direct = _FakeResult(
        positions=[{"tip": (1.0, 2.0)}],
        velocities=[{"tip": (3.0, 4.0)}],
        times=[0.0],
    )
    speed, pos = panel_builders._extract_tip_motion(direct, "tip", velocity_key="tip")
    assert speed == pytest.approx(5.0)
    assert pos.tolist() == [1.0, 2.0]

    derived = _FakeResult(
        positions=[{"tip": (0.0, 0.0)}, {"tip": (3.0, 4.0)}],
        velocities=[{}, {}],
        times=[0.0, 2.0],
    )
    speed, pos = panel_builders._extract_tip_motion(derived, "tip")
    assert speed == pytest.approx(2.5)
    assert pos.tolist() == [3.0, 4.0]


def test_build_double_panel_wires_helper_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pendulum = _FakePendulum()
    fake_controls = _FakeControls(
        {
            "m1": 1.0,
            "m2": 2.0,
            "L1": 3.0,
            "L2": 4.0,
            "theta1_rad": 0.1,
            "phi_rad": 0.2,
            "dtheta1": 0.3,
            "dphi": 0.4,
            "shoulder_coeffs": [0.1, 0.2],
            "wrist_coeffs": [0.3, 0.4],
            "t_end": 1.5,
            "tilt_deg": 15.0,
            "azimuth_deg": 30.0,
            "gravity_on": True,
            "enable_limits": False,
            "enable_clamp": False,
        }
    )
    fake_perturb = _FakePerturbationPanel()

    monkeypatch.setattr(panel_builders, "ControlsWidget", lambda: fake_controls)
    monkeypatch.setattr(panel_builders, "PendulumWidget", lambda: fake_pendulum)
    monkeypatch.setattr(panel_builders, "MatrixWidget", lambda: object())
    monkeypatch.setattr(panel_builders, "TorqueHistoryWidget", lambda: object())
    monkeypatch.setattr(panel_builders, "OptimizationWidget", _FakeOptimizationWidget)
    monkeypatch.setattr(panel_builders, "PerturbationPanel", lambda: fake_perturb)
    monkeypatch.setattr(panel_builders, "SimulationPanel", _FakeSimulationPanel)

    panel = panel_builders.build_double_panel(object())

    assert panel._settings_key == "splitter_double"
    assert panel.pendulum is fake_pendulum

    params = panel.kwargs["params_builder"](fake_controls.get_params())
    assert fake_pendulum.tilt == pytest.approx(math.radians(15.0))
    assert fake_pendulum.azimuth == pytest.approx(math.radians(30.0))
    assert params.g == pytest.approx(
        panel_builders.GRAVITY_MSS * math.cos(math.radians(15.0))
    )

    assert fake_perturb.coeffs_source() == [[0.1, 0.2], [0.3, 0.4]]
    assert fake_perturb.preset_names_source() == ["Preset"]
    assert fake_perturb.preset_coeffs("Preset") == [[1.0, 2.0], [3.0]]
