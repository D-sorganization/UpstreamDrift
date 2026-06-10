"""Smoke tests for the seeded UX YAML configs (epic #5968)."""

from __future__ import annotations

import pytest

from src.shared.python.ux import (
    DEFAULT_ERROR_CATALOG_PATH,
    DEFAULT_FIELD_METADATA_PATH,
    load_error_catalog,
    load_registry,
)

pytestmark = pytest.mark.unit


def test_field_metadata_yaml_loads():
    registry = load_registry(DEFAULT_FIELD_METADATA_PATH)
    # Seeded fields the Phase 0 PR documents in docs/ux/field_metadata.md.
    must_exist = {
        "simulation.duration",
        "simulation.timestep",
        "simulation.live_analysis",
        "simulation.gpu_acceleration",
        "simulation.engine",
        "actuator.control_type",
        "pose_studio.show_radians",
    }
    present = {f.id for f in registry}
    missing = must_exist - present
    assert not missing, f"seed YAML is missing required ids: {missing}"


def test_error_messages_yaml_loads():
    catalog = load_error_catalog(DEFAULT_ERROR_CATALOG_PATH)
    must_exist = {
        "invalid_timestep",
        "invalid_duration",
        "engine_unavailable",
        "model_not_found",
        "simulation_timeout",
        "gpu_unavailable",
    }
    present = {e.code for e in catalog}
    missing = must_exist - present
    assert not missing, f"seed YAML is missing required codes: {missing}"


def test_every_error_field_id_exists_in_registry():
    """Cross-config consistency: every error's field_id must point at a
    real field id (or be None).  Prevents copy from referencing dead
    fields after a refactor.
    """
    catalog = load_error_catalog(DEFAULT_ERROR_CATALOG_PATH)
    registry = load_registry(DEFAULT_FIELD_METADATA_PATH)
    for err in catalog:
        if err.field_id is None:
            continue
        assert err.field_id in registry, (
            f"error {err.code!r} references unknown field {err.field_id!r}"
        )
