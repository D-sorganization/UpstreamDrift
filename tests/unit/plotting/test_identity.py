"""Tests for src.shared.python.plotting.identity (Issue #8828).

Covers:
    (a) A rendered figure's footer contains engine+model identity when
        provided, and stays silent when nothing is known.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from src.shared.python.plotting.energy import plot_energy_overview
from src.shared.python.plotting.identity import PlotIdentity, apply_identity_footer
from src.shared.python.plotting.kinematics import plot_joint_positions


class _EngineStub:
    """Stand-in for a PhysicsEngine exposing engine_type/model_name."""

    def __init__(self, engine_type: Any = None, model_name: str | None = None) -> None:
        self.engine_type = engine_type
        self.model_name = model_name


class _MockRecorder:
    def __init__(self, engine: Any = None) -> None:
        self.engine = engine

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        return np.linspace(0, 1, 10), np.zeros((10, 3))

    def get_induced_acceleration_series(
        self, source: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.linspace(0, 1, 10), np.zeros((10, 3))

    def set_analysis_config(self, config: dict[str, Any]) -> None:
        pass


def _figure_text_strings(fig) -> list[str]:
    return [t.get_text() for t in fig.texts]


@pytest.mark.unit
class TestPlotIdentity:
    def test_label_combines_populated_fields(self) -> None:
        identity = PlotIdentity(engine="mujoco", model="golfer_v3", run_id="run-42")
        label = identity.label()
        assert "Engine: mujoco" in label
        assert "Model: golfer_v3" in label
        assert "Run: run-42" in label

    def test_label_none_when_empty(self) -> None:
        assert PlotIdentity().label() is None

    def test_label_omits_unknown_fields(self) -> None:
        identity = PlotIdentity(engine="drake")
        label = identity.label()
        assert label == "Engine: drake"

    def test_is_empty(self) -> None:
        assert PlotIdentity().is_empty()
        assert not PlotIdentity(engine="mujoco").is_empty()

    def test_as_metadata_dict_only_includes_known_fields(self) -> None:
        identity = PlotIdentity(engine="mujoco")
        meta = identity.as_metadata_dict()
        assert meta == {"engine": "mujoco"}

    def test_from_recorder_reads_engine_type_and_model_name(self) -> None:
        recorder = _MockRecorder(
            engine=_EngineStub(engine_type="mujoco", model_name="golfer")
        )
        identity = PlotIdentity.from_recorder(recorder)
        assert identity.engine == "mujoco"
        assert identity.model == "golfer"

    def test_from_recorder_uses_enum_value_not_repr(self) -> None:
        import enum

        class EngineType(enum.Enum):
            MUJOCO = "mujoco"

        recorder = _MockRecorder(engine=_EngineStub(engine_type=EngineType.MUJOCO))
        identity = PlotIdentity.from_recorder(recorder)
        assert identity.engine == "mujoco"

    def test_from_recorder_does_not_fabricate_missing_fields(self) -> None:
        recorder = _MockRecorder(engine=None)
        identity = PlotIdentity.from_recorder(recorder)
        assert identity.engine is None
        assert identity.model is None
        assert identity.run_id is None

    def test_from_recorder_accepts_explicit_run_id(self) -> None:
        recorder = _MockRecorder(engine=None)
        identity = PlotIdentity.from_recorder(recorder, run_id="run-7")
        assert identity.run_id == "run-7"


@pytest.mark.unit
class TestApplyIdentityFooter:
    def test_noop_for_none_identity(self) -> None:
        import matplotlib.pyplot as plt

        fig, _ax = plt.subplots()
        apply_identity_footer(fig, None)
        assert _figure_text_strings(fig) == []
        plt.close(fig)

    def test_noop_for_empty_identity(self) -> None:
        import matplotlib.pyplot as plt

        fig, _ax = plt.subplots()
        apply_identity_footer(fig, PlotIdentity())
        assert _figure_text_strings(fig) == []
        plt.close(fig)

    def test_renders_populated_identity(self) -> None:
        import matplotlib.pyplot as plt

        fig, _ax = plt.subplots()
        apply_identity_footer(fig, PlotIdentity(engine="drake", model="pendulum"))
        texts = _figure_text_strings(fig)
        assert len(texts) == 1
        assert "drake" in texts[0]
        assert "pendulum" in texts[0]
        plt.close(fig)


@pytest.mark.unit
class TestPlotFunctionsThreadIdentity:
    """(a) plot_* figures embed identity in the footer when known."""

    def test_plot_joint_positions_with_explicit_identity(self) -> None:
        recorder = _MockRecorder(engine=None)
        identity = PlotIdentity(engine="mujoco", model="golfer_v3")
        fig, _ax = plot_joint_positions(recorder, identity=identity)
        texts = _figure_text_strings(fig)
        assert any("mujoco" in t and "golfer_v3" in t for t in texts)

    def test_plot_joint_positions_derives_identity_from_recorder(self) -> None:
        recorder = _MockRecorder(
            engine=_EngineStub(engine_type="drake", model_name="humanoid")
        )
        fig, _ax = plot_joint_positions(recorder)
        texts = _figure_text_strings(fig)
        assert any("drake" in t and "humanoid" in t for t in texts)

    def test_plot_joint_positions_no_footer_when_nothing_known(self) -> None:
        recorder = _MockRecorder(engine=None)
        fig, _ax = plot_joint_positions(recorder)
        assert _figure_text_strings(fig) == []

    def test_plot_energy_overview_with_explicit_identity(self) -> None:
        recorder = _MockRecorder(engine=None)
        identity = PlotIdentity(engine="pinocchio", model="swing_model", run_id="r1")
        fig, _ax = plot_energy_overview(recorder, identity=identity)
        texts = _figure_text_strings(fig)
        assert any("pinocchio" in t and "swing_model" in t and "r1" in t for t in texts)

    def test_plot_energy_overview_derives_identity_from_recorder(self) -> None:
        recorder = _MockRecorder(
            engine=_EngineStub(engine_type="mujoco", model_name="golfer")
        )
        fig, _ax = plot_energy_overview(recorder)
        texts = _figure_text_strings(fig)
        assert any("mujoco" in t and "golfer" in t for t in texts)
