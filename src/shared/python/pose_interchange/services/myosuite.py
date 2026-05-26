"""MyoSuite mock service implementation."""

from __future__ import annotations

from src.shared.python.pose_interchange.live_kinematics import LiveKinematicsService
from src.shared.python.pose_interchange.services._mock import MockKinematicsService


def create_myosuite_service() -> LiveKinematicsService:
    """Create a MyoSuite kinematics service (mocked for now)."""
    return MockKinematicsService("myosuite")
