"""Knowledge packs: cited, freshness-checked retrieval over repository documents.

One engine backs the Disciple and Vision Quest staff roles (a findings pack
over UpstreamDrift and AffineDrift) and the per-product Sidekick Wizards (one
pack per product). A pack is a single SQLite FTS5 file ranked by BM25.

The package depends only on the standard library and PyYAML and imports
nothing else from Tools, so Runner_Dashboard can vendor it unchanged
(Tools#5345, Repository_Management#1772).
"""

from __future__ import annotations

from .chunking import Chunk, chunk_document
from .manifest import (
    AUTHORITIES,
    HIDDEN_STATUSES,
    STATUSES,
    ManifestError,
    PackManifest,
    SourceSpec,
    load_manifest,
    manifest_from_dict,
)
from .pack import (
    FORMAT_VERSION,
    KnowledgePack,
    PackFormatError,
    PackInfo,
    Passage,
    build_pack,
)

__all__ = [
    "AUTHORITIES",
    "FORMAT_VERSION",
    "HIDDEN_STATUSES",
    "STATUSES",
    "Chunk",
    "KnowledgePack",
    "ManifestError",
    "PackFormatError",
    "PackInfo",
    "PackManifest",
    "Passage",
    "SourceSpec",
    "build_pack",
    "chunk_document",
    "load_manifest",
    "manifest_from_dict",
]
