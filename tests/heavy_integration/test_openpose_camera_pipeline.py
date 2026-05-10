"""Heavy integration tests for OpenPose / camera pipeline (fixes #1989).

Tests OpenPose module importability and estimator instantiation with a
mocked camera source. All tests skip gracefully when pyopenpose or the
project's pose estimation module is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.live_simulation
