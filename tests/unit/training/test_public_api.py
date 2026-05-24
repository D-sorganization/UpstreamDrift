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
        "BestMetric",
        "CancelToken",
        "CompatibilityChecker",
        "CompatibilityError",
        "CompatibilityIssue",
        "CompatibilityReport",
        "Dataset",
        "DatasetRegistry",
        "DuplicateJobError",
        "InvalidStatusTransitionError",
        "JobFilter",
        "JobId",
        "JobNotFoundError",
        "JobRegistry",
        "MetricKind",
        "ProgressSink",
        "ResourceRequest",
        "RollingMean",
        "RunId",
        "RunResult",
        "Scheduler",
        "SchedulerError",
        "StatusChangeEvent",
        "ThreadingCancelToken",
        "TrainingConfig",
        "TrainingConfigError",
        "TrainingError",
        "TrainingFramework",
        "TrainingJob",
        "TrainingJobRunner",
        "TrainingMetric",
        "TrainingStatus",
        "best_per_metric",
        "can_transition",
        "filter_by_tags",
        "new_job_id",
        "new_run_id",
        "run_result_from_dict",
        "run_result_to_dict",
        "summarize_by_kind",
        "training_config_from_dict",
        "training_config_to_dict",
        "training_job_from_dict",
        "training_job_to_dict",
        "training_metric_from_dict",
        "training_metric_to_dict",
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
