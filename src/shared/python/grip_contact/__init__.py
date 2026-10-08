"""Engine-agnostic grip interface description (issue #11739, OSV-7)."""

from src.shared.python.grip_contact.interface import GripFrame, GripInterface
from src.shared.python.grip_contact.parameters import (
    BushingParameters,
    ContactMaterial,
    default_bushing,
)

__all__ = [
    "BushingParameters",
    "ContactMaterial",
    "GripFrame",
    "GripInterface",
    "default_bushing",
]
