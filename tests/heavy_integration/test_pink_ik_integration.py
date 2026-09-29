"""
Heavy Integration Contracts — Pink IK Solver
=============================================
Tests are marked @pytest.mark.live_simulation and run only in the heavy
integration lane (weekly CI or local Docker).

Contract: Pink inverse kinematics solver can build tasks and solve IK
for a simple Pinocchio model.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live_simulation
