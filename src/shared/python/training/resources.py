"""Resource-request declaration for training jobs.

A :class:`ResourceRequest` is a *hint* — what a job declares it needs.
The scheduler (built in PR2) uses these declarations to gate admission
and avoid over-subscribing the host. This module owns the declaration
shape and its DbC validation; it does **not** perform any resource
accounting.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import TrainingConfigError

__all__ = ["ResourceRequest"]


MIN_MEMORY_MB = 64
"""Floor for declared memory — anything smaller is almost certainly a bug."""


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """Immutable description of the host resources a job expects to use.

    Attributes:
        cpu_cores: Logical CPU cores the job will use. Must be >= 1.
        gpu_count: Number of GPUs required. ``0`` means CPU-only.
        memory_mb: Resident-set memory ceiling in MiB. Must be >=
            :data:`MIN_MEMORY_MB`.
        gpu_memory_mb: Per-GPU memory ceiling in MiB, or ``None`` for
            "no explicit limit" (use whatever the device exposes).

    Invariants (enforced in :meth:`__post_init__`):
        - ``cpu_cores >= 1``
        - ``gpu_count >= 0``
        - ``memory_mb >= MIN_MEMORY_MB``
        - ``gpu_memory_mb`` is either ``None`` or a positive int
        - ``gpu_memory_mb`` is ``None`` when ``gpu_count == 0``
    """

    cpu_cores: int = 1
    gpu_count: int = 0
    memory_mb: int = 1024
    gpu_memory_mb: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cpu_cores, int) or self.cpu_cores < 1:
            raise TrainingConfigError(
                f"cpu_cores must be a positive int (got {self.cpu_cores!r})"
            )
        if not isinstance(self.gpu_count, int) or self.gpu_count < 0:
            raise TrainingConfigError(
                f"gpu_count must be a non-negative int (got {self.gpu_count!r})"
            )
        if not isinstance(self.memory_mb, int) or self.memory_mb < MIN_MEMORY_MB:
            raise TrainingConfigError(
                f"memory_mb must be >= {MIN_MEMORY_MB} (got {self.memory_mb!r})"
            )
        if self.gpu_memory_mb is not None:
            if not isinstance(self.gpu_memory_mb, int) or self.gpu_memory_mb <= 0:
                raise TrainingConfigError(
                    "gpu_memory_mb must be a positive int or None "
                    f"(got {self.gpu_memory_mb!r})"
                )
            if self.gpu_count == 0:
                raise TrainingConfigError(
                    "gpu_memory_mb must be None when gpu_count == 0"
                )

    @property
    def requires_gpu(self) -> bool:
        """``True`` when the job needs at least one GPU."""

        return self.gpu_count > 0
