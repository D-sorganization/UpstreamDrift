"""Tests for Induced Acceleration Analysis across all physics engines."""

from __future__ import annotations

import numpy as np
import pytest

# --- PINOCCHIO ---


# --- DRAKE ---


# --- MUJOCO ---

# Test basic property: sum should roughly match qacc if we computed it?
# But we are computing acceleration from scratch.
# Should sum to qacc if qacc was consistent with q,v,tau.
# Here we didn't run forward dynamics to set qacc.
