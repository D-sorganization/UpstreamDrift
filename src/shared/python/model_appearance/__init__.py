"""Engine-agnostic model appearance: schema, material library, smooth meshes."""

from src.shared.python.model_appearance.geometry import (
    Mesh,
    ellipsoid_mesh,
    lofted_segment,
)
from src.shared.python.model_appearance.library import (
    CLOTHING,
    MATERIALS,
    classify_body,
    library_materials,
)
from src.shared.python.model_appearance.schema import (
    SCHEMA_VERSION,
    AppearanceDocument,
    Environment,
    Material,
    SegmentRule,
    Texture,
    appearance_path_for,
    document_from_dict,
    document_to_dict,
    load_appearance,
    physics_spec_sha256,
    save_appearance,
    validate_document,
)

__all__ = [
    "CLOTHING",
    "MATERIALS",
    "SCHEMA_VERSION",
    "AppearanceDocument",
    "Environment",
    "Material",
    "Mesh",
    "SegmentRule",
    "Texture",
    "appearance_path_for",
    "classify_body",
    "document_from_dict",
    "document_to_dict",
    "ellipsoid_mesh",
    "library_materials",
    "load_appearance",
    "lofted_segment",
    "physics_spec_sha256",
    "save_appearance",
    "validate_document",
]
