"""Bio/Lab - Biomechanics data readers and laboratory tools.

Modules:
    c3d_reader: C3D motion capture file reader with event and metadata parsing
"""

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from .c3d_reader import C3DDataReader, C3DEvent, C3DMetadata

__all__ = [
    "C3DDataReader",
    "C3DEvent",
    "C3DMetadata",
]
