"""Export a full-body model specification to a URDF bundle with shared contact.

Embeds the physical solids and scalar 1-DOF joint chain of the full-body model
specification (41 coordinates total: 27 upper-body joints + 14 lower-limb joints).
Adds four foot contact sphere links on the calcaneus bodies (calcn_r, calcn_l)
and generates sidecar metadata preserving closure, coordinate order, and frames.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.model_generation.export.model_bundle import export_model_bundle


def export_full_body_urdf(model_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """Export the full-body specification to URDF with scalar joints and contact links.

    Note:
        Moved behind the shared model_generation public API in MV-01.
        Callers should prefer `src.shared.python.model_generation.export.model_bundle.export_model_bundle`.
    """
    bundle = export_model_bundle(model_bytes)
    return bundle.urdf_xml, bundle.sidecar or {}
