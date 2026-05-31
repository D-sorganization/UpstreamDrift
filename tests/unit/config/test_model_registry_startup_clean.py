"""Startup cleanliness for the built-in model registry (#6882).

The biomechanics/PINN built-in entries describe non-file-backed metadata and
historically failed strict ``path`` validation, emitting "Invalid model entry"
errors on every ``EngineManager`` / ``ModelRegistry`` construction. These tests
assert the four cited entries load cleanly and remain discoverable.
"""

from __future__ import annotations

import pytest

from src.shared.python.config.model_registry import ModelRegistry

_METADATA_ENTRY_IDS = (
    "biomech_gait",
    "biomech_sit_to_stand",
    "pinn_pure_rigid",
    "pinn_hybrid",
)


@pytest.mark.unit
def test_builtin_metadata_entries_load_without_errors() -> None:
    """The four metadata entries must not produce registry load errors."""
    registry = ModelRegistry()

    offending = [
        err
        for err in registry.load_errors
        if any(model_id in err for model_id in _METADATA_ENTRY_IDS)
    ]
    assert offending == [], f"unexpected load errors: {offending}"


@pytest.mark.unit
def test_builtin_metadata_entries_are_registered() -> None:
    """The metadata entries remain present in the runtime registry."""
    registry = ModelRegistry()
    for model_id in _METADATA_ENTRY_IDS:
        assert model_id in registry.models, f"{model_id} missing from registry"


@pytest.mark.unit
def test_load_errors_attribute_exists_and_is_list() -> None:
    """DbC: the registry exposes a deterministic ``load_errors`` list."""
    registry = ModelRegistry()
    assert isinstance(registry.load_errors, list)
