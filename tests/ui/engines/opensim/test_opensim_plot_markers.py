"""OpenSim Run Simulation plots whichever markers the model has (#12945)."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("PyQt6")

_GUI = "src.engines.physics_engines.opensim.python.opensim_gui"
_RESOLVER = "src.engines.physics_engines.opensim.python.opensim_golf.marker_tracks"
STEPS = 5


def _gui_module():
    try:
        return importlib.import_module(_GUI)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"OpenSim GUI module not importable: {exc}")


def _track(offset: float = 0.0) -> np.ndarray:
    return np.column_stack(
        [np.linspace(0, 1, STEPS) + offset, np.zeros(STEPS), np.arange(STEPS)]
    )


def _result(markers: dict[str, np.ndarray]):
    core = importlib.import_module(
        "src.engines.physics_engines.opensim.python.opensim_golf.core"
    )
    return core.SimulationResult(
        time=np.linspace(0, 0.4, STEPS),
        states=np.zeros((STEPS, 4)),
        muscle_forces=np.zeros((STEPS, 0)),
        control_signals=np.zeros((STEPS, 0)),
        joint_torques=np.zeros((STEPS, 2)),
        marker_positions=markers,
    )


@pytest.fixture
def widget(qapp):
    mod = _gui_module()
    w = mod.MainWidget(model_path=None)
    try:
        yield w
    finally:
        w.close()
        w.deleteLater()


@pytest.mark.unit
def test_resolver_prefers_hand_and_clubhead() -> None:
    mod = importlib.import_module(_RESOLVER)
    names = mod.resolve_trajectory_markers(
        {"Zed": _track(), "ClubHead": _track(), "Hand": _track()}
    )
    assert names == ("Hand", "ClubHead")


@pytest.mark.unit
def test_resolver_falls_back_to_the_models_own_markers() -> None:
    mod = importlib.import_module(_RESOLVER)
    names = mod.resolve_trajectory_markers(
        {"HeadTop": _track(), "BackLeft": _track(), "Hip": _track()}
    )
    assert names == ("BackLeft", "HeadTop")


@pytest.mark.unit
def test_resolver_mixes_preferred_and_available() -> None:
    mod = importlib.import_module(_RESOLVER)
    names = mod.resolve_trajectory_markers({"ClubHead": _track(), "A": _track()})
    assert names == ("ClubHead", "A")


@pytest.mark.unit
def test_resolver_returns_nothing_without_usable_markers() -> None:
    mod = importlib.import_module(_RESOLVER)
    assert mod.resolve_trajectory_markers({}) == ()
    assert mod.resolve_trajectory_markers({"Bad": np.zeros(3)}) == ()


@pytest.mark.unit
def test_resolver_rejects_a_non_mapping() -> None:
    mod = importlib.import_module(_RESOLVER)
    with pytest.raises(TypeError):
        mod.resolve_trajectory_markers(["Hand"])  # type: ignore[arg-type]


@pytest.mark.unit
def test_plot_results_uses_the_models_markers(widget) -> None:
    widget.result = _result({"HeadTop": _track(), "BackLeft": _track(1.0)})
    widget.plot_results()  # used to raise KeyError: 'Hand'
    labels = [t.get_label() for t in widget.fig.axes[-1].get_lines()]
    assert labels == ["BackLeft path", "HeadTop path"]


@pytest.mark.unit
def test_plot_results_skips_the_trajectory_with_a_message_when_no_markers(
    widget,
) -> None:
    widget.result = _result({})
    widget.plot_results()
    texts = [t.get_text() for t in widget.fig.axes[-1].texts]
    assert any("no markers" in t.lower() for t in texts)
    assert len(widget.fig.axes) == 3  # joint angles and torques still plotted


class _FakeModel:
    def __init__(self, result) -> None:
        self._result = result

    def run_simulation(self):
        return self._result


@pytest.mark.unit
def test_run_simulation_completes_on_a_model_without_hand_marker(widget) -> None:
    widget.model = _FakeModel(_result({"HeadTop": _track()}))
    widget.run_simulation()
    assert widget.lbl_details.text().startswith("Simulation Complete")
    assert widget.btn_run.isEnabled()


@pytest.mark.unit
def test_status_is_reset_even_when_plotting_raises_unexpectedly(
    widget, monkeypatch
) -> None:
    widget.model = _FakeModel(_result({}))

    def boom() -> None:
        raise KeyError("Hand")

    monkeypatch.setattr(widget, "plot_results", boom)
    with pytest.raises(KeyError):
        widget.run_simulation()
    assert widget.lbl_details.text() != "Running..."
    assert widget.btn_run.isEnabled()
