"""
I/O module for BunkerShot3D.
"""

from .schema import (
    FIELD_DIGEST_ATTR,
    FIELD_GROUP,
    FIELD_METADATA_ATTR,
    SCHEMA_VERSION,
    SCHEMA_VERSION_ATTR,
    SUPPORTED_SCHEMA_VERSIONS,
    BunkerShotResultReader,
    BunkerShotResultWriter,
    SandFieldPayload,
)

__all__: list[str] = [
    "FIELD_DIGEST_ATTR",
    "FIELD_GROUP",
    "FIELD_METADATA_ATTR",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_ATTR",
    "SUPPORTED_SCHEMA_VERSIONS",
    "BunkerShotResultReader",
    "BunkerShotResultWriter",
    "SandFieldPayload",
]
