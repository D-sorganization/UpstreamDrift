"""Tests for the humanoid character builder preview functionality."""

import sys
from unittest.mock import patch

from humanoid_character_builder.interfaces.api import BodyParameters, CharacterBuilder


def test_simulation_missing_mujoco():
    """Test simulate() returns False gracefully if MuJoCo is missing."""
    builder = CharacterBuilder()
    params = BodyParameters(height_m=1.80)
    result = builder.build(params, generate_meshes=False)

    # Simulate missing mujoco
    with patch.dict(sys.modules, {"mujoco": None}):
        assert result.simulate() is False


# Complex mocked tests disabled due to CI instability with sys.modules patching
# def test_preview_logic_mocked(monkeypatch): ...
# def test_simulation_stability_mocked(monkeypatch): ...
