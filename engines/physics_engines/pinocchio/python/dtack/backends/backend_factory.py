# mypy: disable-error-code="no-any-unimported"
"""Factory for creating backend instances."""

from __future__ import annotations

import typing
from enum import Enum

if typing.TYPE_CHECKING:
    from pathlib import Path

from dtack.backends.mujoco_backend import MuJoCoBackend
from dtack.backends.pink_backend import PINKBackend
from dtack.backends.pinocchio_backend import PinocchioBackend


class BackendType(str, Enum):
    """Supported backend types."""

    PINOCCHIO = "pinocchio"
    MUJOCO = "mujoco"
    PINK = "pink"


class BackendFactory:
    """Factory for creating backend instances from canonical model specification."""

    @staticmethod
    def create(
        backend_type: BackendType | str,
        model_path: Path | str,
    ) -> PinocchioBackend | MuJoCoBackend | PINKBackend:
        """Create a backend instance.

        Args:
            backend_type: Type of backend to create
            model_path: Path to canonical model specification or backend-specific file

        Returns:
            Backend instance

        Raises:
            ValueError: If backend type is not supported
        """
        backend_str = str(backend_type).lower()

        if backend_str == BackendType.PINOCCHIO:
            return PinocchioBackend(model_path)
        if backend_str == BackendType.MUJOCO:
            return MuJoCoBackend(model_path)
        if backend_str == BackendType.PINK:
            return PINKBackend(model_path)

        msg = f"Unsupported backend type: {backend_type}"
        raise ValueError(msg)
