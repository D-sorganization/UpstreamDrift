"""Shared fixtures for calc_backend tests."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> Any:
    import sys
    from unittest.mock import MagicMock

    sys.modules["rotation_converter"] = MagicMock()
    sys.modules["rotation_converter.reference_frame_operations"] = MagicMock()
    sys.modules["rotation_converter.converter"] = MagicMock()

    from calc_backend.app import app

    return TestClient(app)
