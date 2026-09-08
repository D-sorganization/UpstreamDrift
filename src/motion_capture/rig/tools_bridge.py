"""Fail-closed bridge to the Tools mocap schema.

ADR-0041: the camera, capture, timebase and session *contracts* are owned by
Tools ``sidekick.lab.mocap`` and reach UpstreamDrift through the pinned vendor
tree. Until that pin ships the schema, this module reports exactly what is
present and refuses to invent a mapping. UpstreamDrift #9422 owns the real
export once the pinned release documents its builders.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal

TOOLS_MOCAP_MODULE = "sidekick.lab.mocap"
REQUIRED_SUBMODULES = ("devices", "timebase", "session", "serialization")

SchemaStatus = Literal["ready", "incompatible", "unavailable"]


@dataclass(frozen=True)
class SchemaProbe:
    """What the pinned Tools tree offers for mocap sessions."""

    status: SchemaStatus
    reason: str | None
    module: str = TOOLS_MOCAP_MODULE
    version: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "module": self.module,
            "status": self.status,
            "reason": self.reason,
            "version": self.version,
        }


def _spec_present(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def probe_tools_schema(module_name: str = TOOLS_MOCAP_MODULE) -> SchemaProbe:
    """Report whether the Tools mocap schema is importable and complete.

    Never raises for a missing or partial installation; the answer is data.
    """
    if not _spec_present(module_name):
        return SchemaProbe(
            "unavailable",
            f"{module_name} is not in the pinned vendor tree (Tools #4706 / PR #4734)",
            module_name,
        )
    missing = [
        s for s in REQUIRED_SUBMODULES if not _spec_present(f"{module_name}.{s}")
    ]
    if missing:
        return SchemaProbe(
            "incompatible",
            f"{module_name} lacks expected submodules: {', '.join(missing)}",
            module_name,
        )
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    return SchemaProbe("ready", None, module_name, str(version) if version else None)
