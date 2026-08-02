"""Shared Python-version compatibility helpers for shared modules."""

from __future__ import annotations

import sys
from datetime import timezone
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING or sys.version_info >= (3, 11):
    from datetime import UTC
    from enum import StrEnum
else:
    UTC = timezone.utc  # noqa: UP017

    class StrEnum(str, Enum):  # noqa: UP042
        """Backport of :class:`enum.StrEnum` for Python 3.10."""

        def __str__(self) -> str:
            return str(self.value)
