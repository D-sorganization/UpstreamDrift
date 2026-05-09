"""
Heavy Integration Contracts — Meshcat Visualization
=====================================================
Tests are marked @pytest.mark.live_simulation and run only in the heavy
integration lane.

Contract: Meshcat can create a visualizer, add geometry, and render
without crashing in a headless environment.
"""

from __future__ import annotations

import numpy as np
import pytest


pytestmark = pytest.mark.live_simulation
