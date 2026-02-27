"""Bio/Lab - Biomechanics data readers and laboratory tools.

Modules:
    c3d_reader: C3D motion capture file reader with event and metadata parsing
"""

from .c3d_reader import C3DDataReader, C3DEvent, C3DMetadata

__all__ = [
    "C3DDataReader",
    "C3DEvent",
    "C3DMetadata",
]
