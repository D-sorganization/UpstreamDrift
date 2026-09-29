"""Comprehensive tests for src.robotics.core.exceptions.

Targets the previously-uncovered exception constructors and details handling.
"""

from __future__ import annotations

import pytest

from src.robotics.core.exceptions import (
    ContactError,
    ControlError,
    KinematicsError,
    LocomotionError,
    RoboticsError,
    SolverError,
)


class TestRoboticsError:
    def test_basic_message(self) -> None:
        err = RoboticsError("boom")
        assert err.message == "boom"
        assert err.details == {}
        assert str(err) == "boom"

    def test_with_details(self) -> None:
        err = RoboticsError("boom", details={"a": 1, "b": 2})
        s = str(err)
        assert "boom" in s
        assert "a=1" in s and "b=2" in s

    def test_inheritance(self) -> None:
        assert issubclass(ContactError, RoboticsError)
        assert issubclass(ControlError, RoboticsError)
        assert issubclass(SolverError, RoboticsError)
        assert issubclass(LocomotionError, RoboticsError)
        assert issubclass(KinematicsError, RoboticsError)


class TestContactError:
    def test_minimal(self) -> None:
        err = ContactError("contact failed")
        assert err.contact_id is None
        assert err.body_names is None
        assert err.details == {}

    def test_with_id(self) -> None:
        err = ContactError("c", contact_id=7)
        assert err.contact_id == 7
        assert err.details["contact_id"] == 7

    def test_with_body_names(self) -> None:
        err = ContactError("c", body_names=("foot_l", "ground"))
        assert err.body_names == ("foot_l", "ground")
        assert err.details["body_a"] == "foot_l"
        assert err.details["body_b"] == "ground"

    def test_raise_and_catch_as_base(self) -> None:
        with pytest.raises(RoboticsError):
            raise ContactError("x", contact_id=1)


class TestControlError:
    def test_with_joints(self) -> None:
        err = ControlError("c", joint_indices=[0, 1, 2])
        assert err.joint_indices == [0, 1, 2]
        assert err.details["joint_indices"] == [0, 1, 2]

    def test_with_values(self) -> None:
        err = ControlError("c", control_values=[0.1, 0.2])
        assert err.control_values == [0.1, 0.2]

    def test_existing_details_merged(self) -> None:
        err = ControlError("c", joint_indices=[1], details={"extra": "v"})
        assert err.details["extra"] == "v"
        assert err.details["joint_indices"] == [1]


class TestSolverError:
    def test_full_args(self) -> None:
        err = SolverError("fail", solver_name="scipy", status_code=2, iterations=50)
        assert err.solver_name == "scipy"
        assert err.status_code == 2
        assert err.iterations == 50
        assert err.details["solver"] == "scipy"
        assert err.details["status_code"] == 2
        assert err.details["iterations"] == 50

    def test_minimal(self) -> None:
        err = SolverError("fail")
        assert err.solver_name is None
        assert err.details == {}


class TestLocomotionError:
    def test_full(self) -> None:
        err = LocomotionError("l", gait_phase="swing", support_state="single")
        assert err.gait_phase == "swing"
        assert err.support_state == "single"
        assert err.details["gait_phase"] == "swing"
        assert err.details["support_state"] == "single"


class TestKinematicsError:
    def test_full(self) -> None:
        err = KinematicsError("k", body_name="hand", configuration=[0.0, 0.1, 0.2])
        assert err.body_name == "hand"
        assert err.configuration == [0.0, 0.1, 0.2]
        assert err.details["body"] == "hand"
        assert err.details["config_size"] == 3
