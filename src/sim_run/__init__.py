"""Headless simulation run service, store, and schema contracts (Epic #14085 / #14086 / #14087)."""

from __future__ import annotations

from .cli import main as run_cli
from .manifest import (
    RunManifest,
    RunManifestError,
    export_run_manifest_schema,
    validate_run_manifest,
)
from .mcp import SecurityPolicy, SimulationMcpServer, run_stdio_loop
from .models import (
    Provenance,
    ResultEnvelope,
    RunRecord,
    RunRequest,
    RunStatus,
    SCHEMA_VERSION_V1,
    ValidationStatus,
    export_schema,
    export_schemas,
)
from .store import (
    ALLOWED_TRANSITIONS,
    InvalidStateTransitionError,
    RunNotFoundError,
    RunStoreError,
    SimulationRunStore,
)
from .validation import (
    validate_numerical_integrity,
    validate_provenance,
    validate_result,
)
from .worker import (
    LinuxLimits,
    SimulationWorkerPool,
    apply_linux_rlimits,
    get_repo_sha,
    reap_subprocess,
    run_simulation_job,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "InvalidStateTransitionError",
    "LinuxLimits",
    "Provenance",
    "ResultEnvelope",
    "RunManifest",
    "RunManifestError",
    "RunNotFoundError",
    "RunRecord",
    "RunRequest",
    "RunStatus",
    "RunStoreError",
    "SCHEMA_VERSION_V1",
    "SecurityPolicy",
    "SimulationMcpServer",
    "SimulationRunStore",
    "SimulationWorkerPool",
    "ValidationStatus",
    "apply_linux_rlimits",
    "export_run_manifest_schema",
    "export_schema",
    "export_schemas",
    "get_repo_sha",
    "reap_subprocess",
    "run_cli",
    "run_simulation_job",
    "run_stdio_loop",
    "validate_numerical_integrity",
    "validate_provenance",
    "validate_result",
    "validate_run_manifest",
]
