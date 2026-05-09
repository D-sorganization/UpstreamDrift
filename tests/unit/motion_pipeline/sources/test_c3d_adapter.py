"""Tests for the C3D adapter (ezc3d optional dependency)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources import c3d_adapter as _mod
from src.shared.python.motion_pipeline.sources.c3d_adapter import C3DAdapter

_HAS_EZC3D = _mod._HAS_EZC3D
