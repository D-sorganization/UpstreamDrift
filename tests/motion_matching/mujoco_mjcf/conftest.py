"""Test fixtures for the MuJoCo MJCF motion-matching package.

Skips the whole module if ``mujoco`` is not installed so the test suite
remains green on minimal CI configurations.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")
