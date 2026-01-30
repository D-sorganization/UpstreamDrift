"""Tests for marker-to-model mapping (Guideline A2 - Mandatory)."""

from __future__ import annotations

import pytest

from src.shared.python.engine_availability import (
    MUJOCO_AVAILABLE,
    skip_if_unavailable,
)

pytestmark = skip_if_unavailable("mujoco")

if MUJOCO_AVAILABLE:
    import mujoco


@pytest.fixture
def simple_model() -> mujoco.MjModel:
    """Simple model for testing."""
    xml = """
    <mujoco>
        <worldbody>
            <body name="torso" pos="0 0 1">
                <geom type="box" size="0.2 0.1 0.3"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)
