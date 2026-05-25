"""Tests for dtack backend factory pattern.

Tests the BackendFactory.create() method and BackendType enum
without requiring actual physics engine dependencies (uses mocks
for backends that require pinocchio/mujoco/pink).
"""

from __future__ import annotations
