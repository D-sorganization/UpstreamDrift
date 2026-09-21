"""Tests for model_generation and humanoid_character_builder facades (#8641).

Asserts that all public symbols exported by model_generation.humanoid,
model_generation.mesh, and humanoid_character_builder CLI resolve to live,
importable objects.
"""

from __future__ import annotations

import pytest

import src.shared.python.humanoid_character_builder.__main__ as hcb_main
import src.shared.python.model_generation.humanoid as humanoid_facade
import src.shared.python.model_generation.mesh as mesh_facade

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_model_generation_humanoid_facade_symbols_resolve() -> None:
    """Every export advertised in model_generation.humanoid.__all__ must resolve."""
    assert len(humanoid_facade.__all__) == 34
    missing = [
        name for name in humanoid_facade.__all__ if not hasattr(humanoid_facade, name)
    ]
    assert not missing, f"Unresolvable exports in model_generation.humanoid: {missing}"


def test_model_generation_mesh_facade_symbols_resolve() -> None:
    """Every export advertised in model_generation.mesh.__all__ must resolve."""
    assert len(mesh_facade.__all__) == 14
    missing = [name for name in mesh_facade.__all__ if not hasattr(mesh_facade, name)]
    assert not missing, f"Unresolvable exports in model_generation.mesh: {missing}"


def test_humanoid_character_builder_main_cli_imports() -> None:
    """CLI main entry point must bind CharacterBuilder and parser cleanly."""
    assert hasattr(hcb_main, "CharacterBuilder")
    assert callable(hcb_main.main)
