"""
Heavy Integration Contracts — C3D Motion Capture Data
=====================================================
Tests are marked @pytest.mark.live_simulation and run only in the heavy
integration lane.

Contract: The c3d library can read/write C3D motion capture files and
integrates with the project's data_io pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.live_simulation
