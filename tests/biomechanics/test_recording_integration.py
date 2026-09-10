"""Public recorder boundary retains calibrated geometry beside dynamics."""

import numpy as np
import pytest

from src.shared.python.biomechanics.model_bindings import ModelBinding, SegmentBinding
from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.engine_core.mock_engine import MockPhysicsEngine

pytestmark = pytest.mark.unit


class Geometry:
    def get_link_transforms(self):
        return {"pelvis": np.eye(4)}


def test_recorded_geometry_survives_stop_and_reset_clears_it():
    engine = MockPhysicsEngine()
    recorder = GenericPhysicsRecorder(engine)
    binding = ModelBinding("synthetic", "Z-up", {"pelvis": SegmentBinding("pelvis")})
    recorder.configure_biomechanics(binding, Geometry())
    recorder.start()
    recorder.record_step()
    engine.step(0.01)
    recorder.record_step()
    recorder.stop()
    payload = recorder.get_biomechanics_payload()
    assert len(payload["times"]) == 2
    recorder.reset()
    assert recorder.get_biomechanics_payload() is None


def test_unconfigured_recording_has_no_fabricated_geometry():
    recorder = GenericPhysicsRecorder(MockPhysicsEngine())
    assert recorder.get_biomechanics_payload() is None
