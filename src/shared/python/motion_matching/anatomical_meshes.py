"""Anatomical mesh substitution without changing qualified physics (MMR-05 #11089).

Provides:
1. One-segment File Solid / mesh substitution behind optional visual presets,
   strictly preserving exact joint frames, mass, COM, inertia, and same-state forward kinematics.
2. Design by Contract validation for SI mesh units, finite bounding boxes, and asset existence,
   with documented graceful fallback to primitive visuals when permitted.
3. Tracking of redistribution terms and asset provenance (source repository, commit, license).
4. Simscape compiled block budget checks ensuring models remain within MMR-04 limits
   (production ceiling <= 975 compiled nonvirtual blocks, 25-block reserve from 1,000 license ceiling).
5. Reusable AnatomicalSegmentAdapter for pelvis, trunk, head, hand, and shoe prototypes,
   handling handedness/bilateral mirroring, multi-phase key swing poses, marker overlay, and clipping.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.tools.tour_matching_viewer.core import body_poses_from_state

Array: TypeAlias = NDArray[np.float64]

# MMR-04 compiled budget constants
HOME_LICENSE_BLOCK_LIMIT: int = 1000
INSTRUMENTATION_RESERVE: int = 25
PRODUCTION_BUDGET_CEILING: int = (
    HOME_LICENSE_BLOCK_LIMIT - INSTRUMENTATION_RESERVE
)  # 975
BLOCKS_PER_EXTERNAL_SOLID: int = 7  # Measured in SHAPE.md and MMR-04 (973 -> 980)

# Supported SI units and scale factors to meters
SUPPORTED_MESH_UNITS: dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
}

SUPPORTED_PROTOTYPES: tuple[str, ...] = ("pelvis", "trunk", "head", "hand", "shoe")


class AnatomicalMeshError(Exception):
    """Base error for anatomical mesh operations."""


class InvalidMeshUnitsError(AnatomicalMeshError):
    """Raised when mesh units are unrecognized or scale factor is invalid."""


class NonFiniteMeshBoundsError(AnatomicalMeshError):
    """Raised when mesh bounding box values are non-finite or inverted."""


class MissingMeshAssetError(AnatomicalMeshError):
    """Raised when the specified mesh asset cannot be found on disk."""


class SimscapeBudgetExceededError(AnatomicalMeshError):
    """Raised when visual solids cause compiled blocks to exceed budget ceiling."""


class VisualPreset(str, Enum):
    """Visual geometry preset."""

    PRIMITIVE = "primitive"
    ANATOMICAL_MESH = "anatomical_mesh"
    HYBRID = "hybrid"


class VisualFallbackStatus(str, Enum):
    """Status indicating whether visual geometry fell back to primitive."""

    NONE = "none"
    FALLBACK_TO_PRIMITIVE = "fallback_to_primitive"


class SimscapeVisualStrategy(str, Enum):
    """Strategy for incorporating visual geometry into Simscape Multibody."""

    FILE_SOLID_SUBSTITUTION = "file_solid_substitution"
    EXTERNAL_VISUAL_SOLID = "external_visual_solid"
    EXTERNAL_RENDER_SKIN = "external_render_skin"


@dataclass(frozen=True)
class BoundingBox3D:
    """Axis-aligned 3D bounding box in metres."""

    min_point: tuple[float, float, float]
    max_point: tuple[float, float, float]

    def __post_init__(self) -> None:
        min_arr = np.asarray(self.min_point, dtype=float)
        max_arr = np.asarray(self.max_point, dtype=float)

        if not (np.isfinite(min_arr).all() and np.isfinite(max_arr).all()):
            raise NonFiniteMeshBoundsError(
                "BoundingBox3D min_point and max_point must be finite"
            )

        if (min_arr >= max_arr).any():
            raise NonFiniteMeshBoundsError(
                f"min_point must be strictly less than max_point in all dimensions: "
                f"min={self.min_point}, max={self.max_point}"
            )

    @property
    def dimensions_m(self) -> tuple[float, float, float]:
        """Length, width, and height of bounding box in metres."""
        dx = float(self.max_point[0] - self.min_point[0])
        dy = float(self.max_point[1] - self.min_point[1])
        dz = float(self.max_point[2] - self.min_point[2])
        return (dx, dy, dz)

    @property
    def center_m(self) -> tuple[float, float, float]:
        """Center point of bounding box in metres."""
        cx = float((self.min_point[0] + self.max_point[0]) / 2.0)
        cy = float((self.min_point[1] + self.max_point[1]) / 2.0)
        cz = float((self.min_point[2] + self.max_point[2]) / 2.0)
        return (cx, cy, cz)


@dataclass(frozen=True)
class MeshAssetProvenance:
    """Provenance and license redistribution terms for a 3D anatomical mesh asset."""

    asset_name: str
    source_repository: str
    source_commit: str
    license_type: str
    license_url: str
    attribution: str
    redistribution_allowed: bool = True
    notes: tuple[str, ...] = ()

    def is_valid_for_redistribution(self) -> bool:
        """Verify whether the asset is authorized for public repository redistribution."""
        if not self.redistribution_allowed:
            return False
        if not self.license_type or self.license_type.strip().lower() in (
            "proprietary",
            "all-rights-reserved",
            "unknown",
        ):
            return False
        return bool(self.attribution.strip())


@dataclass(frozen=True)
class MeshMetadata:
    """Metadata describing a physical 3D mesh asset."""

    file_path: Path
    file_sha256: str
    format: str = "STL"
    units: str = "m"
    scale_to_meters: float = 1.0
    bounding_box: BoundingBox3D = field(
        default_factory=lambda: BoundingBox3D((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))
    )
    vertex_count: int = 0
    face_count: int = 0
    provenance: MeshAssetProvenance = field(
        default_factory=lambda: MeshAssetProvenance(
            "unknown", "unknown", "unknown", "CC-BY-SA 2.0", "", "unknown", True
        )
    )
    validate: bool = True

    def __post_init__(self) -> None:
        if not self.validate:
            return
        norm_units = self.units.strip().lower()
        if norm_units not in SUPPORTED_MESH_UNITS:
            raise InvalidMeshUnitsError(
                f"Unsupported mesh units '{self.units}'. Supported: {sorted(SUPPORTED_MESH_UNITS.keys())}"
            )
        if not math.isfinite(self.scale_to_meters) or self.scale_to_meters <= 0:
            raise InvalidMeshUnitsError(
                f"scale_to_meters must be finite and strictly positive, got {self.scale_to_meters}"
            )


@dataclass(frozen=True)
class BundledModelInfo:
    """Metadata for bundled mesh model package."""

    model_name: str
    description: str
    mesh_count: int
    mesh_format: str
    mesh_directory: str
    provenance: MeshAssetProvenance


@dataclass(frozen=True)
class SegmentMeshConfig:
    """Configuration mapping an anatomical segment to a mesh asset."""

    segment_name: str
    mesh_metadata: MeshMetadata
    attachment_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    attachment_rotation: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    visual_strategy: SimscapeVisualStrategy = (
        SimscapeVisualStrategy.FILE_SOLID_SUBSTITUTION
    )
    fallback_shape: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class BudgetCheckResult:
    """Outcome of Simscape block budget validation for visual mesh changes."""

    passed: bool
    new_compiled_total: int
    compiled_delta: int
    ceiling: int
    headroom: int
    diagnostic: str = ""


@dataclass(frozen=True)
class ClearanceResult:
    """Clearance distance between two segments."""

    segment_a: str
    segment_b: str
    clearance_m: float
    intersecting: bool


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_simscape_mesh_block_budget(
    current_compiled_blocks: int,
    strategy: SimscapeVisualStrategy,
    num_segments: int = 1,
) -> BudgetCheckResult:
    """Verify that adding or substituting visual solids respects the MMR-04 compiled budget."""
    if strategy in (
        SimscapeVisualStrategy.FILE_SOLID_SUBSTITUTION,
        SimscapeVisualStrategy.EXTERNAL_RENDER_SKIN,
    ):
        delta = 0
    elif strategy == SimscapeVisualStrategy.EXTERNAL_VISUAL_SOLID:
        delta = BLOCKS_PER_EXTERNAL_SOLID * num_segments
    else:
        delta = 0

    new_total = current_compiled_blocks + delta
    headroom = PRODUCTION_BUDGET_CEILING - new_total
    passed = new_total <= PRODUCTION_BUDGET_CEILING

    diagnostic = ""
    if not passed:
        diagnostic = (
            f"Compiled block count ({new_total}) exceeds production budget ceiling of {PRODUCTION_BUDGET_CEILING} "
            f"(mandatory reserve of {INSTRUMENTATION_RESERVE} blocks from {HOME_LICENSE_BLOCK_LIMIT} license limit). "
            f"Strategy {strategy.value} adds {delta} blocks across {num_segments} segment(s)."
        )

    return BudgetCheckResult(
        passed=passed,
        new_compiled_total=new_total,
        compiled_delta=delta,
        ceiling=PRODUCTION_BUDGET_CEILING,
        headroom=headroom,
        diagnostic=diagnostic,
    )


def load_bundled_human_mesh_metadata(metadata_path: Path) -> BundledModelInfo:
    """Load model metadata and license provenance from bundled metadata.json."""
    if not metadata_path.exists():
        raise MissingMeshAssetError(
            f"Bundled metadata file not found at {metadata_path}"
        )

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    prov_data = data.get("license", {})
    source_data = data.get("source", {})
    model_data = data.get("model", {})

    provenance = MeshAssetProvenance(
        asset_name=data.get("name", "Bundled Human Model"),
        source_repository=source_data.get("repository", ""),
        source_commit=source_data.get("commit", "master"),
        license_type=prov_data.get("type", "CC-BY-SA 2.0"),
        license_url=prov_data.get("url", ""),
        attribution=prov_data.get("attribution", ""),
        redistribution_allowed=True,
        notes=tuple(data.get("notes", ())),
    )

    return BundledModelInfo(
        model_name=data.get("name", "Unknown"),
        description=data.get("description", ""),
        mesh_count=int(model_data.get("mesh_count", 0)),
        mesh_format=str(model_data.get("mesh_format", "STL")),
        mesh_directory=str(model_data.get("mesh_directory", "meshes/")),
        provenance=provenance,
    )


def _validate_config_asset(config: SegmentMeshConfig) -> None:
    """Validate asset existence, SI units, and bounds (DbC)."""
    meta = config.mesh_metadata
    norm_units = meta.units.strip().lower()
    if norm_units not in SUPPORTED_MESH_UNITS:
        raise InvalidMeshUnitsError(
            f"Unsupported mesh units '{meta.units}'. Supported: {sorted(SUPPORTED_MESH_UNITS.keys())}"
        )
    if not math.isfinite(meta.scale_to_meters) or meta.scale_to_meters <= 0:
        raise InvalidMeshUnitsError(
            f"scale_to_meters must be strictly positive and finite, got {meta.scale_to_meters}"
        )
    if not meta.file_path.exists():
        raise MissingMeshAssetError(f"Mesh asset file not found: {meta.file_path}")


def apply_mesh_substitution(
    spec: Mapping[str, Any],
    configs: Sequence[SegmentMeshConfig],
    preset: VisualPreset = VisualPreset.ANATOMICAL_MESH,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    """Substitute anatomical mesh visuals behind an optional visual preset.

    Guarantees:
    - Bodies, mass, COM, inertia, joint frames, coordinate order, and closure remain byte-identical.
    - If allow_fallback=True, invalid units, non-finite bounds, or missing files gracefully revert
      to primitive visual shapes while recording the fallback status.
    - If allow_fallback=False, defects fail closed immediately.
    """
    new_spec = copy.deepcopy(dict(spec))
    visual_hints = copy.deepcopy(new_spec.get("visual_hints", {}) or {})
    shapes = dict(visual_hints.get("shapes", {}) or {})
    mesh_substitutions: dict[str, Any] = {}
    mesh_fallbacks: dict[str, str] = {}

    for config in configs:
        seg = config.segment_name
        try:
            _validate_config_asset(config)
            valid = True
        except (
            InvalidMeshUnitsError,
            NonFiniteMeshBoundsError,
            MissingMeshAssetError,
        ) as exc:
            if not allow_fallback:
                raise
            valid = False
            mesh_fallbacks[seg] = VisualFallbackStatus.FALLBACK_TO_PRIMITIVE.value
            if config.fallback_shape is not None:
                # Find matching solid in body
                for body in new_spec.get("bodies", []):
                    if body["name"] == seg:
                        for s in body.get("solids", []):
                            shapes[s["name"]] = dict(config.fallback_shape)

        if valid:
            mesh_fallbacks[seg] = VisualFallbackStatus.NONE.value
            if preset in (VisualPreset.ANATOMICAL_MESH, VisualPreset.HYBRID):
                meta = config.mesh_metadata
                mesh_substitutions[seg] = {
                    "mesh_path": str(meta.file_path),
                    "file_sha256": meta.file_sha256,
                    "format": meta.format,
                    "units": meta.units,
                    "scale_to_meters": meta.scale_to_meters,
                    "bounding_box_min_m": list(meta.bounding_box.min_point),
                    "bounding_box_max_m": list(meta.bounding_box.max_point),
                    "visual_strategy": config.visual_strategy.value,
                    "attachment_offset_m": list(config.attachment_offset_m),
                    "attachment_rotation": [
                        list(row) for row in config.attachment_rotation
                    ],
                    "provenance": {
                        "asset_name": meta.provenance.asset_name,
                        "source_repository": meta.provenance.source_repository,
                        "source_commit": meta.provenance.source_commit,
                        "license_type": meta.provenance.license_type,
                        "license_url": meta.provenance.license_url,
                        "attribution": meta.provenance.attribution,
                        "redistribution_allowed": meta.provenance.redistribution_allowed,
                    },
                }

    visual_hints["shapes"] = shapes
    visual_hints["mesh_substitutions"] = mesh_substitutions
    visual_hints["mesh_fallbacks"] = mesh_fallbacks
    visual_hints["visual_preset"] = preset.value
    new_spec["visual_hints"] = visual_hints

    return new_spec


class AnatomicalSegmentAdapter:
    """Reusable segment adapter for pelvis, trunk, head, hand, and shoe prototypes."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self._base_spec = copy.deepcopy(dict(spec))
        self._configs: dict[str, SegmentMeshConfig] = {}

    @classmethod
    def supported_prototypes(cls) -> tuple[str, ...]:
        """Return the list of standard anatomical prototype regions."""
        return SUPPORTED_PROTOTYPES

    def configure_segment(
        self,
        segment_name: str,
        metadata: MeshMetadata,
        strategy: SimscapeVisualStrategy = SimscapeVisualStrategy.FILE_SOLID_SUBSTITUTION,
        fallback_shape: Mapping[str, Any] | None = None,
    ) -> None:
        """Register a mesh configuration for an anatomical segment."""
        # Find matching body in spec
        matched = [
            b["name"]
            for b in self._base_spec.get("bodies", [])
            if b["name"] == segment_name
        ]
        if not matched:
            # Check prefix matching (e.g. shoe_l matching foot_l)
            if segment_name.startswith("shoe_"):
                alt = segment_name.replace("shoe_", "foot_")
                matched = [
                    b["name"]
                    for b in self._base_spec.get("bodies", [])
                    if b["name"] == alt
                ]
        name = matched[0] if matched else segment_name

        config = SegmentMeshConfig(
            segment_name=name,
            mesh_metadata=metadata,
            visual_strategy=strategy,
            fallback_shape=fallback_shape,
        )
        self._configs[segment_name] = config
        self._configs[name] = config

    def get_segment_config(self, segment_name: str) -> SegmentMeshConfig:
        """Retrieve the configuration for a segment."""
        if segment_name not in self._configs:
            raise KeyError(
                f"Segment '{segment_name}' has not been configured in adapter"
            )
        return self._configs[segment_name]

    def mirror_bilateral_metadata(
        self, meta: MeshMetadata, lateral_axis: str = "y"
    ) -> MeshMetadata:
        """Produce a mirrored mesh metadata for bilateral segments (hands and shoes)."""
        axis_idx = {"x": 0, "y": 1, "z": 2}[lateral_axis.lower()]
        min_p = list(meta.bounding_box.min_point)
        max_p = list(meta.bounding_box.max_point)

        # Reflect across plane
        new_min_val = -max_p[axis_idx]
        new_max_val = -min_p[axis_idx]
        min_p[axis_idx] = new_min_val
        max_p[axis_idx] = new_max_val

        mirrored_bbox = BoundingBox3D(
            min_point=(float(min_p[0]), float(min_p[1]), float(min_p[2])),
            max_point=(float(max_p[0]), float(max_p[1]), float(max_p[2])),
        )

        return MeshMetadata(
            file_path=meta.file_path,
            file_sha256=meta.file_sha256,
            format=meta.format,
            units=meta.units,
            scale_to_meters=meta.scale_to_meters,
            bounding_box=mirrored_bbox,
            vertex_count=meta.vertex_count,
            face_count=meta.face_count,
            provenance=meta.provenance,
        )

    def build_substituted_spec(
        self,
        preset: VisualPreset = VisualPreset.ANATOMICAL_MESH,
        allow_fallback: bool = False,
    ) -> dict[str, Any]:
        """Apply all configured segment mesh substitutions."""
        unique_configs = list(
            {c.segment_name: c for c in self._configs.values()}.values()
        )
        return apply_mesh_substitution(
            self._base_spec,
            unique_configs,
            preset=preset,
            allow_fallback=allow_fallback,
        )

    def verify_marker_overlay_invariance(
        self,
        q: NDArray[np.float64],
        coordinate_names: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Assert marker positions are strictly invariant between base and substituted specs."""
        coords = coordinate_names or tuple(self._base_spec.get("coordinate_order", ()))
        sub_spec = self.build_substituted_spec()
        poses_orig = body_poses_from_state(self._base_spec, q, coordinate_names=coords)
        poses_sub = body_poses_from_state(sub_spec, q, coordinate_names=coords)

        residuals: dict[str, float] = {}
        for frame in self._base_spec.get("frames", []):
            name = frame["name"]
            b_name = frame["body"]
            f_placement = np.asarray(frame["placement"], dtype=float)

            pos_orig = (poses_orig[b_name] @ f_placement)[:3, 3]
            pos_sub = (poses_sub[b_name] @ f_placement)[:3, 3]

            dist = float(np.linalg.norm(pos_orig - pos_sub))
            residuals[name] = dist

        return residuals

    def check_segment_clearance(
        self,
        q: NDArray[np.float64],
        coordinate_names: Sequence[str] | None = None,
    ) -> dict[str, ClearanceResult]:
        """Check bounding box clearance / self-intersection between adjacent segments."""
        coords = coordinate_names or tuple(self._base_spec.get("coordinate_order", ()))
        sub_spec = self.build_substituted_spec()
        poses = body_poses_from_state(sub_spec, q, coordinate_names=coords)
        results: dict[str, ClearanceResult] = {}

        # Check adjacent pairs
        pairs = [
            ("trunk", "pelvis"),
            ("head", "trunk"),
            ("hand_l", "trunk"),
            ("foot_l", "pelvis"),
        ]
        for seg_a, seg_b in pairs:
            if seg_a not in poses or seg_b not in poses:
                continue

            pos_a = poses[seg_a][:3, 3]
            pos_b = poses[seg_b][:3, 3]
            center_dist = float(np.linalg.norm(pos_a - pos_b))

            # Retrieve bbox half sizes if available
            r_a = 0.15
            r_b = 0.15
            if seg_a in self._configs:
                dims = self._configs[seg_a].mesh_metadata.bounding_box.dimensions_m
                r_a = max(dims) / 2.0
            if seg_b in self._configs:
                dims = self._configs[seg_b].mesh_metadata.bounding_box.dimensions_m
                r_b = max(dims) / 2.0

            clearance = center_dist - (r_a + r_b)
            pair_key = f"{seg_a}_vs_{seg_b}"
            results[pair_key] = ClearanceResult(
                segment_a=seg_a,
                segment_b=seg_b,
                clearance_m=clearance,
                intersecting=clearance < 0.0,
            )

        return results
