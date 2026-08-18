"""Package-wide exception hierarchy for BunkerShot3D (issue #8608, W1).

Before this issue the package raised a mix of bare ``ValueError``,
``RuntimeError``, ``NotImplementedError`` and ``FileNotFoundError``, so a caller
could not catch "BunkerShot3D rejected this" without catching everything.

The rule enforced here is deliberately strong: *every* exception class defined
anywhere under ``bunkershot3d`` must descend from
:class:`~bunkershot3d.exceptions.BunkerShot3DError`, which itself descends from
the platform root ``core.error_utils.GolfSuiteError`` (CLAUDE.md error-handling
table). The walk is automatic, so a new module cannot quietly introduce an
orphan exception.

The pre-existing standard-library bases are *kept* in the MRO — callers who
catch ``ValueError`` or ``NotImplementedError`` today keep working.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import bunkershot3d
from bunkershot3d.exceptions import (
    BackendNotImplementedError,
    BunkerShot3DError,
    BunkerShot3DStateError,
    BunkerShot3DValueError,
    ConfigurationInvalidError,
    UnitConventionError,
    UnitConversionError,
)
from src.shared.python.core.error_utils import GolfSuiteError

pytestmark = pytest.mark.unit


#: Optional third-party backends whose import failure is not a package defect.
_OPTIONAL_IMPORT_ERRORS = (ImportError, OSError)


def _package_exception_classes() -> dict[str, type[BaseException]]:
    """Every exception class *defined* in a ``bunkershot3d`` module."""
    found: dict[str, type[BaseException]] = {}
    for info in pkgutil.walk_packages(bunkershot3d.__path__, prefix="bunkershot3d."):
        try:
            module = importlib.import_module(info.name)
        except _OPTIONAL_IMPORT_ERRORS:  # pragma: no cover - optional backends
            continue
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseException):
                continue
            if not obj.__module__.startswith("bunkershot3d."):
                continue
            found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return found


class TestRoot:
    def test_root_descends_from_the_platform_root(self) -> None:
        assert issubclass(BunkerShot3DError, GolfSuiteError)

    def test_value_flavour_is_still_a_value_error(self) -> None:
        assert issubclass(BunkerShot3DValueError, BunkerShot3DError)
        assert issubclass(BunkerShot3DValueError, ValueError)

    def test_state_flavour_is_still_a_runtime_error(self) -> None:
        assert issubclass(BunkerShot3DStateError, BunkerShot3DError)
        assert issubclass(BunkerShot3DStateError, RuntimeError)

    @pytest.mark.parametrize(
        "cls",
        [ConfigurationInvalidError, UnitConversionError, UnitConventionError],
    )
    def test_new_errors_are_value_errors(self, cls: type[BaseException]) -> None:
        assert issubclass(cls, BunkerShot3DValueError)


class TestBackwardCompatibility:
    """Existing catch sites must keep working."""

    def test_backend_not_implemented_is_still_a_not_implemented_error(self) -> None:
        assert issubclass(BackendNotImplementedError, NotImplementedError)
        assert issubclass(BackendNotImplementedError, BunkerShot3DError)

    def test_backend_not_implemented_keeps_its_two_argument_signature(self) -> None:
        error = BackendNotImplementedError("liggghts", feature="no clubhead")
        assert error.backend == "liggghts"
        assert error.feature == "no clubhead"
        assert "liggghts" in str(error)
        assert "no clubhead" in str(error)

    def test_trajectory_unavailable_is_still_a_file_not_found_error(self) -> None:
        from bunkershot3d.backends.prescribed_motion import TrajectoryUnavailableError

        assert issubclass(TrajectoryUnavailableError, FileNotFoundError)
        assert issubclass(TrajectoryUnavailableError, BunkerShot3DError)

    @pytest.mark.parametrize(
        "dotted",
        [
            "bunkershot3d.backends.stability.TimestepStabilityError",
            "bunkershot3d.backends.stability.ContactStiffnessError",
            "bunkershot3d.calibration.optimizer.InertParameterError",
            "bunkershot3d.geometry.mesh.MeshValidationError",
            "bunkershot3d.sand.exceptions.SandModelError",
        ],
    )
    def test_existing_value_errors_stay_value_errors(self, dotted: str) -> None:
        module_name, _, class_name = dotted.rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        assert issubclass(cls, ValueError)
        assert issubclass(cls, BunkerShot3DError)

    def test_step_budget_exceeded_stays_a_runtime_error(self) -> None:
        from bunkershot3d.backends.stability import StepBudgetExceededError

        assert issubclass(StepBudgetExceededError, RuntimeError)
        assert issubclass(StepBudgetExceededError, BunkerShot3DError)


class TestHierarchyIsComplete:
    def test_the_walk_finds_the_known_classes(self) -> None:
        """Guard the guard: an empty walk would make the next test vacuous."""
        names = _package_exception_classes()
        assert len(names) >= 10, sorted(names)

    def test_every_package_exception_descends_from_the_root(self) -> None:
        orphans = sorted(
            dotted
            for dotted, cls in _package_exception_classes().items()
            if not issubclass(cls, BunkerShot3DError)
        )
        assert not orphans, (
            "these exceptions do not descend from BunkerShot3DError: "
            f"{orphans}. Root them on bunkershot3d.exceptions so callers can "
            "catch 'BunkerShot3D rejected this' in one clause."
        )
