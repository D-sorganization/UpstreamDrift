"""Tests for the LOD-driven delegating accessors added under issue #4139.

These tests pin down the new `ClubTrajectory` and `TrajectoryIKResult`
properties so future changes can't silently regress the LOD refactor.
"""

from __future__ import annotations

import numpy as np
import pytest
