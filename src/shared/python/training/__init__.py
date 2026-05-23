"""Training-controller contracts and domain types (PR1).

Public façade for the training subsystem. PR1 ships pure-Python
contracts: identifiers, status, metrics, resources, configuration,
compatibility checking, job records, and Protocols. Implementations
(subprocess scheduler, GUI, framework adapters, resource monitor) land
in later PRs.

Stability: the surface re-exported via ``__all__`` is the supported
import surface. Do not import internal module paths directly from
outside this package.
"""

from __future__ import annotations

from .compatibility import (
    DEFAULT_ENGINE_FRAMEWORK_MAP,
    CompatibilityChecker,
    CompatibilityIssue,
    CompatibilityReport,
)
from .config import (
    CURRENT_SCHEMA_VERSION,
    TrainingConfig,
    TrainingFramework,
)
from .contracts import (
    CancelToken,
    ProgressSink,
    ThreadingCancelToken,
    TrainingJobRunner,
)
from .datasets import Dataset, DatasetRegistry
from .errors import (
    CompatibilityError,
    DuplicateJobError,
    InvalidStatusTransitionError,
    JobNotFoundError,
    TrainingConfigError,
    TrainingError,
)
from .identifiers import (
    MAX_ID_LENGTH,
    JobId,
    RunId,
    new_job_id,
    new_run_id,
)
from .job import RunResult, TrainingJob
from .metrics import MetricKind, TrainingMetric
from .persistence import (
    run_result_from_dict,
    run_result_to_dict,
    training_config_from_dict,
    training_config_to_dict,
    training_job_from_dict,
    training_job_to_dict,
    training_metric_from_dict,
    training_metric_to_dict,
)
from .registry import JobFilter, JobRegistry
from .resources import ResourceRequest
from .scheduler import Scheduler, SchedulerError, StatusChangeEvent
from .status import (
    TERMINAL_STATUSES,
    TrainingStatus,
    can_transition,
    validate_transition,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "DEFAULT_ENGINE_FRAMEWORK_MAP",
    "MAX_ID_LENGTH",
    "TERMINAL_STATUSES",
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
    "can_transition",
    "new_job_id",
    "new_run_id",
    "run_result_from_dict",
    "run_result_to_dict",
    "training_config_from_dict",
    "training_config_to_dict",
    "training_job_from_dict",
    "training_job_to_dict",
    "training_metric_from_dict",
    "training_metric_to_dict",
    "validate_transition",
]
