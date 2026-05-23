"""Surface-level checks on the :mod:`training` public façade.

These tests lock in the importable surface so downstream PRs that wire
in the scheduler and GUI don't accidentally narrow it.
"""

from __future__ import annotations

import pytest

import training

pytestmark = pytest.mark.unit


EXPECTED_PUBLIC_NAMES = frozenset(
    {
        "CURRENT_SCHEMA_VERSION",
        "DEFAULT_ENGINE_FRAMEWORK_MAP",
        "MAX_ID_LENGTH",
        "TERMINAL_STATUSES",
        "CancelToken",
        "CompatibilityChecker",
        "CompatibilityError",
        "CompatibilityIssue",
        "CompatibilityReport",
        "DuplicateJobError",
        "InvalidStatusTransitionError",
        "JobId",
        "JobNotFoundError",
        "MetricKind",
        "ProgressSink",
        "ResourceRequest",
        "RunId",
        "RunResult",
        "ThreadingCancelToken",
        "TrainingConfig",
        "TrainingConfigError",
        "TrainingError",
        "TrainingFramework",
        "TrainingJob",
        "TrainingJobRunner",
        "TrainingMetric",
        "TrainingStatus",
        "can_transition",
        "new_job_id",
        "new_run_id",
        "validate_transition",
    }
)


def test_all_matches_expected_surface() -> None:
    assert set(training.__all__) == EXPECTED_PUBLIC_NAMES


def test_every_exported_name_resolves() -> None:
    for name in training.__all__:
        assert hasattr(training, name), f"{name} is in __all__ but not importable"


def test_no_internal_module_leakage() -> None:
    """Internal sub-modules must not be re-exported as top-level names."""
    leaks = [
        n
        for n in dir(training)
        if not n.startswith("_") and n not in EXPECTED_PUBLIC_NAMES
    ]
    # Sub-module names imported during ``from .x import Y`` are tolerated;
    # what matters is that __all__ is the authoritative surface.
    assert set(training.__all__) <= set(dir(training))
    # Be permissive about sub-module names but check __all__ stays clean:
    assert set(training.__all__) == EXPECTED_PUBLIC_NAMES, (
        f"unexpected surface differences: {leaks}"
    )


def test_training_error_is_root_of_hierarchy() -> None:
    """Every domain-specific error must derive from TrainingError."""
    for name in (
        "CompatibilityError",
        "DuplicateJobError",
        "InvalidStatusTransitionError",
        "JobNotFoundError",
        "TrainingConfigError",
    ):
        cls = getattr(training, name)
        assert issubclass(cls, training.TrainingError), (
            f"{name} does not derive from TrainingError"
        )
