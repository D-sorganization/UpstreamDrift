"""Shim: re-exports from src.shared.python.plotting for backward compatibility."""

from src.shared.python.plotting.base import (
    MplCanvas as MplCanvas,
)
from src.shared.python.plotting.base import (
    RecorderInterface as RecorderInterface,
)
from src.shared.python.plotting.core import GolfSwingPlotter as GolfSwingPlotter

__all__ = [
    "GolfSwingPlotter",
    "MplCanvas",
    "RecorderInterface",
]
