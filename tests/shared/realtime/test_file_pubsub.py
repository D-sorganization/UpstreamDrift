"""Contract: legacy file_pubsub transport removed (#8869)."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


def test_file_pubsub_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.shared.python.realtime.file_pubsub")
