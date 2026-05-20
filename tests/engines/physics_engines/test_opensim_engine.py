"""Tests for OpenSimPhysicsEngine wrapper without a real OpenSim install.

We exercise the unavailable-engine behaviour (graceful errors), argument
validation, and methods that don't need a loaded model.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
    OpenSimPhysicsEngine,
)
from src.shared.python.core.contracts.exceptions import PreconditionError


@pytest.fixture
def engine() -> OpenSimPhysicsEngine:
    return OpenSimPhysicsEngine()


class TestConstruction:
    def test_default_state(self, engine: OpenSimPhysicsEngine) -> None:
        assert engine._model is None
        assert engine._state is None
        assert engine.is_initialized is False

    def test_model_name_no_model(self, engine: OpenSimPhysicsEngine) -> None:
        assert engine.model_name == "OpenSim_NoModel"

    def test_get_state_returns_empty_arrays(self, engine: OpenSimPhysicsEngine) -> None:
        q, v = engine.get_state()
        assert q.size == 0 and v.size == 0

    def test_get_time_zero(self, engine: OpenSimPhysicsEngine) -> None:
        assert engine.get_time() == 0.0


class TestLoadUnavailable:
    def test_load_from_path_without_opensim_raises(
        self, engine: OpenSimPhysicsEngine
    ) -> None:
        with pytest.raises(ImportError):
            engine.load_from_path("/tmp/fake.osim")

    def test_load_from_path_nonexistent_when_opensim_present(
        self, engine: OpenSimPhysicsEngine, tmp_path
    ) -> None:
        # Pretend OpenSim is available so FileNotFound raises before opensim use.
        with patch(
            "src.engines.physics_engines.opensim.python.opensim_physics_engine.opensim",
            MagicMock(),
        ):
            missing = tmp_path / "missing.osim"
            with pytest.raises(FileNotFoundError):
                engine.load_from_path(str(missing))

    def test_reload_raises_runtime_error(self, engine: OpenSimPhysicsEngine) -> None:
        # Force is_initialized True by injecting fakes.
        engine._model = MagicMock()
        engine._state = MagicMock()
        with pytest.raises(RuntimeError, match="already has a loaded model"):
            engine.load_from_path("/anything")

    def test_load_from_string_without_opensim_raises(
        self, engine: OpenSimPhysicsEngine
    ) -> None:
        with pytest.raises(ImportError):
            engine.load_from_string("<xml/>", extension="osim")

    def test_load_from_string_creates_tempfile(
        self, engine: OpenSimPhysicsEngine
    ) -> None:
        # Patch opensim and load_from_path to verify temp file plumbing.
        fake_opensim = MagicMock()
        captured: dict = {}

        def fake_load_from_path(path: str) -> None:
            captured["path"] = path
            assert os.path.exists(path)

        with (
            patch(
                "src.engines.physics_engines.opensim.python."
                "opensim_physics_engine.opensim",
                fake_opensim,
            ),
            patch.object(engine, "load_from_path", side_effect=fake_load_from_path),
        ):
            engine.load_from_string("<xml/>", extension="osim")
        # Temp file should be cleaned up.
        assert not os.path.exists(captured["path"])


class TestArgValidation:
    def test_set_state_none_raises(self, engine: OpenSimPhysicsEngine) -> None:
        with pytest.raises(ValueError):
            engine.set_state(None, np.zeros(2))  # type: ignore[arg-type]

    def test_set_state_uninit_noop(self, engine: OpenSimPhysicsEngine) -> None:
        # Without _model, set_state silently returns.
        engine.set_state(np.zeros(2), np.zeros(2))

    def test_set_control_none_raises(self, engine: OpenSimPhysicsEngine) -> None:
        with pytest.raises(ValueError):
            engine.set_control(None)  # type: ignore[arg-type]

    def test_set_control_uninit_noop(self, engine: OpenSimPhysicsEngine) -> None:
        engine.set_control(np.zeros(2))


class TestUninitPreconditions:
    """Methods guarded by @precondition is_initialized raise when not loaded."""

    @pytest.mark.parametrize(
        "method,args",
        [
            ("reset", ()),
            ("step", ()),
            ("forward", ()),
            ("compute_mass_matrix", ()),
            ("compute_bias_forces", ()),
        ],
    )
    def test_precondition_blocks_uninit_call(
        self, engine: OpenSimPhysicsEngine, method: str, args: tuple
    ) -> None:
        with pytest.raises(PreconditionError):
            getattr(engine, method)(*args)
