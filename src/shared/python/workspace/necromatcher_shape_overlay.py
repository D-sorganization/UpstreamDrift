"""Source-bound model visual proxies, distinct from measured anatomical surfaces."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any
import numpy as np
from src.shared.python.body_part_viz import (
    BindingKind,
    BodyPartShape,
    FittedShape,
    MarkerBinding,
)
from src.shared.python.body_part_viz.fitters.between_two import BetweenTwoMarkersFitter
from src.shared.python.body_part_viz.shapes import (
    BoxShape,
    CapsuleShape,
    EllipsoidShape,
    MeshShape,
)
from src.shared.python.body_part_viz.renderers.projective_renderer import (
    SurfaceMesh,
    SurfaceLayer,
    composite_surface,
    render_surface_layer,
    validate_surface_size,
    validate_surface_source,
)
from .necromatcher_native import NativeFitBinding
from src.shared.python.body_part_viz.overlay_options import ShapeOverlayOptions
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton


def shape_overlay_provenance(
    native_definition: Mapping[str, Any],
    native_xml_sha256: str,
    options: ShapeOverlayOptions,
) -> dict[str, Any]:
    """Compute expected display provenance without importing or compiling an SDK."""
    if not isinstance(native_definition, Mapping) or not isinstance(
        options, ShapeOverlayOptions
    ):
        raise TypeError("Shape provenance requires native definition and typed options")
    if (
        not isinstance(native_xml_sha256, str)
        or len(native_xml_sha256) != 71
        or not native_xml_sha256.startswith("sha256:")
        or any(c not in "0123456789abcdef" for c in native_xml_sha256[7:])
    ):
        raise ValueError("Native XML identity must be a prefixed SHA256")
    definition = json.dumps(native_definition, allow_nan=False).encode("utf-8")
    skeleton = derive_visual_skeleton(native_definition)
    visual = json.dumps(
        asdict(skeleton), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()

    def digest(value: bytes) -> str:
        return "sha256:" + hashlib.sha256(value).hexdigest()

    return {
        "options": options.to_record(),
        "geometry_basis": "model_conditioned_visual_proxy",
        "definition_sha256": digest(definition),
        "native_xml_sha256": native_xml_sha256,
        "visual_description_sha256": digest(visual),
        "uncertainty_calibrated": False,
        "physical_geometry_qualified": False,
        "projection": "saved_camera",
        "skeleton_retained": True,
        "renderer": "shared_projective_triangles/1",
        "tessellation": {"longitude": 16, "latitude": 8},
        "near_plane_m": 1e-6,
        "occlusion": "nearest_model_surface; original_scene_occlusion_unknown",
        "derived_radii": "shared_visual_skeleton mass-derived defaults unless saved hints",
        "authored_visual_hints": json.loads(definition).get("visual_hints", {}),
    }


@dataclass(frozen=True)
class LocalSurface:
    """Cached shared-shape mesh in its saved native body's local frame."""

    identity: str
    body: str
    shape: MeshShape


def _capsule_surface(capsule: Any, index: int) -> LocalSurface:
    """Use shared capsule tessellation and axis fitter, without radius inference."""
    identity = f"capsule:{index}:{capsule.body}"
    length = capsule.length_m()
    shape = CapsuleShape(length, capsule.radius_m, shape_id=identity)
    binding = MarkerBinding(BindingKind.BETWEEN_TWO, ("start", "end"), (length,))
    start = np.asarray(capsule.start_m, dtype=float)[None, :]
    end = np.asarray(capsule.end_m, dtype=float)[None, :]
    fitted = BetweenTwoMarkersFitter().fit(shape, binding, {"start": start, "end": end})
    # Capsule rest coordinates span [0,length], unlike a centered marker cylinder.
    fitted = replace(fitted, centroid=start)
    vertices = shape.transform(fitted)[0]
    return LocalSurface(
        identity,
        capsule.body,
        MeshShape(
            vertices,
            shape.faces(),
            (length, capsule.radius_m, capsule.radius_m),
            shape_id=identity,
        ),
    )


def _hint_surface(hint: Any, index: int) -> LocalSurface:
    """Use exact saved visual half sizes/center; no anatomical mesh substitution."""
    identity = f"hint:{index}:{hint.label}"
    shape: BodyPartShape
    if hint.kind == "ellipsoid":
        shape = EllipsoidShape(*hint.half_size_m, shape_id=identity)
    elif hint.kind == "box":
        shape = BoxShape(tuple(hint.half_size_m), shape_id=identity)
    else:
        raise ValueError("Unsupported saved visual shape")
    vertices = shape.vertices_at_rest() + np.asarray(hint.center_m)
    return LocalSurface(
        identity,
        hint.body,
        MeshShape(vertices, shape.faces(), tuple(hint.half_size_m), shape_id=identity),
    )


@dataclass(frozen=True)
class NativeShapeOverlay:
    """Prepared bound visual proxies; source/native identities stay independently checked."""

    options: ShapeOverlayOptions
    surfaces: tuple[LocalSurface, ...]
    provenance: dict[str, Any]

    @classmethod
    def prepare(
        cls, binding: NativeFitBinding, options: ShapeOverlayOptions
    ) -> "NativeShapeOverlay":
        """Cache only geometry derived by the existing shared visual provider."""
        definition = json.loads(binding.definition_bytes)
        skeleton = derive_visual_skeleton(definition)
        surfaces = tuple(
            _capsule_surface(c, i) for i, c in enumerate(skeleton.capsules)
        )
        surfaces += tuple(_hint_surface(h, i) for i, h in enumerate(skeleton.shapes))
        if not surfaces:
            raise ValueError("Native model has no visual proxies")
        return cls(
            options,
            surfaces,
            shape_overlay_provenance(definition, binding.model_hash, options),
        )

    def composite(
        self, binding: NativeFitBinding, pose: np.ndarray, source: np.ndarray
    ) -> np.ndarray:
        """Batch canonical FK once, then use saved camera and shared shape transforms."""
        validate_surface_source(source)
        if self.options.opacity == 0:
            return source.copy()
        layer = self.surface_layer(binding, pose, (source.shape[1], source.shape[0]))
        return composite_surface(source, layer, self.options)

    def surface_layer(
        self, binding: NativeFitBinding, pose: np.ndarray, image_size: tuple[int, int]
    ) -> SurfaceLayer:
        """Return exact sealed pixels, mask, camera-depth metres and geometry IDs.

        This is the compositing geometry, even at zero display opacity; it adds
        no second camera/FK path or claim about original-scene occlusion.
        """
        validate_surface_size(image_size)
        bodies = tuple(sorted({s.body for s in self.surfaces}))
        mapping = {name: (name, (0.0, 0.0, 0.0)) for name in bodies}
        poses = binding.plant.frame_poses(mapping, pose)
        camera, _ = binding.review_inputs()
        meshes = []
        for surface in self.surfaces:
            rotation, position = poses[surface.body]
            fitted = FittedShape(
                surface.identity,
                MarkerBinding(BindingKind.ON_MARKER, (surface.body,)),
                np.asarray(position)[None, :],
                np.asarray(rotation)[None, :, :],
                np.ones((1, 3)),
                np.ones(1, dtype=bool),
            )
            vertices = np.asarray(surface.shape.transform(fitted)).reshape(-1, 3)
            meshes.append(
                SurfaceMesh(
                    surface.identity, vertices, surface.shape.faces(), (165, 145, 110)
                )
            )
        return render_surface_layer(tuple(meshes), camera, image_size)
