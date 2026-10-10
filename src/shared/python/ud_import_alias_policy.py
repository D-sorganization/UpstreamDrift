"""UpstreamDrift policy on top of Tools ``SharedImportAliasFinder``.

``model_generation`` and ``humanoid_character_builder`` are ruled ud-canonical in
``docs/shared_tools/seam_rulings.v1.json``. Their implementations live under
``src/shared/python``; ``src.shared.python.<root>`` must resolve there instead
of being rewritten into the pinned Tools tree by the alias finder (issue #12177).

This module is UD-owned so we do not edit the Tools child copy
``import_aliases.py`` directly.
"""

from __future__ import annotations

import sys

from src.shared.python.import_aliases import (
    SharedImportAliasFinder,
    _bind_legacy_src_namespaces,
    _coalesce_loaded_aliases,
    _DOWNSTREAM_SRC_ALIAS_ROOTS,
    _SHARED_ROOTS,
    _external_src_package_is_available,
)

__all__ = [
    "UD_CANONICAL_SRC_ROOTS",
    "UdSharedImportAliasFinder",
    "install_ud_canonical_shared_import_aliases",
]

UD_CANONICAL_SRC_ROOTS = frozenset({"humanoid_character_builder", "model_generation"})


class UdSharedImportAliasFinder(SharedImportAliasFinder):
    """Decline ``src.shared.python`` aliasing for ud-canonical package roots."""

    def _parse(self, fullname: str) -> tuple[str | None, str]:
        parts = fullname.split(".")
        if len(parts) >= 3 and parts[:2] == ["shared", "python"]:
            return (
                (parts[2], ".".join(parts[3:]))
                if parts[2] in _SHARED_ROOTS
                else (None, "")
            )
        if len(parts) >= 4 and parts[:3] == ["src", "shared", "python"]:
            root = parts[3]
            if root in UD_CANONICAL_SRC_ROOTS:
                return (None, "")
            allowed_roots = (
                _DOWNSTREAM_SRC_ALIAS_ROOTS
                if _external_src_package_is_available()
                else _SHARED_ROOTS
            )
            return (root, ".".join(parts[4:])) if root in allowed_roots else (None, "")
        if parts and parts[0] in _SHARED_ROOTS:
            return parts[0], ".".join(parts[1:])
        return None, ""


def install_ud_canonical_shared_import_aliases() -> None:
    """Install the UD policy finder once per interpreter."""
    _bind_legacy_src_namespaces()
    for index, finder in enumerate(sys.meta_path):
        if isinstance(finder, SharedImportAliasFinder):
            if type(finder) is UdSharedImportAliasFinder:
                _coalesce_loaded_aliases(finder)
                return
            sys.meta_path[index] = UdSharedImportAliasFinder()
            _coalesce_loaded_aliases(sys.meta_path[index])
            return
    finder = UdSharedImportAliasFinder()
    _coalesce_loaded_aliases(finder)
    sys.meta_path.insert(0, finder)
    _coalesce_loaded_aliases(finder)
