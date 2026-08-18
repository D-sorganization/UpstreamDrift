"""Parametric wedge geometry for BunkerShot3D (ADR-0032, issue #8609).

The wedge is a first-class parametric model following the Acushnet
sole-geometry schema; meshes and mass properties are *derived* from it
and verified before use:

* :mod:`.wedge` - the ``WedgeGeometry`` design vector and its invariants
* :mod:`.bounce` - the two non-interchangeable bounce conventions
* :mod:`.delivery` - effective loft, bounce and aim at impact
* :mod:`.presets` - named grinds with per-number provenance
* :mod:`.profile` - parametric sole cross-sections, and the band of
  camber areas a given sole width can actually realise
* :mod:`.lofting` - the lofted, watertight head mesh, and the account of
  what its sole realised against what it declared
* :mod:`.design_bounds` - screening a design space against that band
  before a sweep spends solver time inside it
* :mod:`.mesh` - the mesh value object and its validity preconditions
* :mod:`.mass_properties` - divergence-theorem volume/centroid/inertia
* :mod:`.solids` - analytic solids used to verify the integrator
"""

from .bounce import (
    BounceAngle,
    BounceConvention,
    GeometricBounce,
    MarketedBounce,
    geometric_from_marketed,
    marketed_from_geometric,
)
from .clubhead import ClubheadGenerator
from .delivery import (
    DeliveredGeometry,
    DeliveryCondition,
    deliver_wedge,
    effective_bounce_deg,
    effective_loft_closed_form_deg,
    effective_loft_deg,
)
from .design_bounds import check_camber_design_space
from .lofting import (
    CamberFit,
    LoftedWedge,
    StationCamber,
    build_wedge_mesh,
    loft_wedge,
    shaft_axis,
    wedge_mass_properties,
)
from .mass_properties import MassProperties, compute_mass_properties
from .mesh import (
    MeshValidationError,
    MeshValidity,
    TriangleMesh,
    check_mesh_validity,
    require_watertight,
)
from .presets import (
    GRIND_PRESETS,
    GrindPreset,
    ProvenanceKind,
    get_preset,
    preset_names,
)
from .profile import (
    InconstructibleCamberError,
    SoleProfile,
    build_section_polygon,
    build_sole_profile,
    constructible_camber_range_m2,
)
from .wedge import PatentBand, WedgeGeometry

__all__: list[str] = [
    "GRIND_PRESETS",
    "BounceAngle",
    "BounceConvention",
    "CamberFit",
    "ClubheadGenerator",
    "DeliveredGeometry",
    "DeliveryCondition",
    "GeometricBounce",
    "GrindPreset",
    "InconstructibleCamberError",
    "LoftedWedge",
    "MarketedBounce",
    "MassProperties",
    "MeshValidationError",
    "MeshValidity",
    "PatentBand",
    "ProvenanceKind",
    "SoleProfile",
    "StationCamber",
    "TriangleMesh",
    "WedgeGeometry",
    "build_section_polygon",
    "build_sole_profile",
    "build_wedge_mesh",
    "check_camber_design_space",
    "check_mesh_validity",
    "compute_mass_properties",
    "constructible_camber_range_m2",
    "deliver_wedge",
    "effective_bounce_deg",
    "effective_loft_closed_form_deg",
    "effective_loft_deg",
    "geometric_from_marketed",
    "get_preset",
    "loft_wedge",
    "marketed_from_geometric",
    "preset_names",
    "require_watertight",
    "shaft_axis",
    "wedge_mass_properties",
]
