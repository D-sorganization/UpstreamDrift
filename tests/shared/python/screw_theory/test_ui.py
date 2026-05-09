"""Tests for ScrewVisualizationTab UI component.

Validates that the shared Screw Theory visualization tab can be instantiated
and provides the expected interface: is_active() and get_target_body().
"""

from __future__ import annotations

import pytest
from src.shared.python.engine_core.engine_availability import (
    skip_if_unavailable,
)

pytestmark = pytest.mark.unit
