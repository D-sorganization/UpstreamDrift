from unittest.mock import patch

from src.shared.python.pose_interchange.services.pinocchio import (
    create_pinocchio_service,
    PinocchioKinematicsService,
)
from src.shared.python.pose_interchange.services._mock import MockKinematicsService


def test_create_pinocchio_service_without_wheel():
    with patch(
        "src.shared.python.pose_interchange.services.pinocchio._pinocchio_is_importable",
        return_value=False,
    ):
        svc = create_pinocchio_service()
        assert isinstance(svc, MockKinematicsService)
        assert svc.engine_name == "pinocchio"


def test_create_pinocchio_service_with_wheel():
    with patch(
        "src.shared.python.pose_interchange.services.pinocchio._pinocchio_is_importable",
        return_value=True,
    ):
        svc = create_pinocchio_service()
        assert isinstance(svc, PinocchioKinematicsService)
        assert svc.engine_name == "pinocchio"
