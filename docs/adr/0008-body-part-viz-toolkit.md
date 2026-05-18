# ADR 0008: Body-Part Visualisation Toolkit

- Status: Accepted
- Date: 2026-05-08
- Related issues: #4755 (epic), #4768 (this docs pass)

## Context

The motion-matching pipeline ([ADR 0006](0006-multi-source-motion-targets.md))
already accepts full-body capture targets and runs forward-kinematic
predictions against them, but the rendering surface lagged behind the
data surface. Body segments could only be drawn as lines or as fixed
cylinders, and three different tools each grew their own ad-hoc
geometry stack:

- The C3D Viewer's Segments tab built per-frame line segments inline,
  with hard-coded radii and colour groups in `segments_tab.py`.
- The starting-pose matcher used a parallel "draw cylinder between two
  markers" helper that was a sibling re-implementation of the viewer's,
  not a shared call.
- The humanoid character builder generated URDF visuals from yet
  another set of geometry primitives that did not know about the
  viewer's segment styling at all.

That meant a user who imported a custom forearm mesh into the C3D
Viewer could not re-use it as a URDF visual link without remodelling,
and the matcher's "preview" pose could not match what the viewer
actually drew. Three implementations of what is conceptually the same
shape vocabulary had drifted, and every change to one had to be
manually mirrored to the other two.

A separate forcing function was extensibility. Adding capsule or
ellipsoid shapes — which the matcher needs for limb visualisation that
is closer to anatomy than a uniform-radius cylinder — would have meant
touching all three call sites, plus the segment-set persistence layer,
plus the URDF generator. The ad-hoc shape stacks made each addition
cost more than the last.

## Decision

Ship `src/shared/python/body_part_viz/` as the canonical shape stack
for every tool that draws body segments. The package defines three
runtime-checkable Protocols (`BodyPartShape`, `ShapeFitter`,
`ShapeRenderer`) and ships concrete implementations of each, plus an
asset library of bundled default meshes:

- **Shapes**: `LineShape`, `CylinderShape`, `EllipsoidShape`,
  `CapsuleShape`, `MeshShape`, `CompositeShape`. Each reports its
  rest-pose vertices / faces and applies a fitted transform.
- **Fitters**: `BetweenTwoMarkersFitter`, `ClusterKabschFitter`,
  `ProcrustesAnisotropicFitter`. Each computes a per-frame
  `FittedShape` (centroid, rotation, scale, valid mask) from a
  `MarkerBinding` and a dict of marker trajectories.
- **Renderers**: `MatplotlibRenderer` is the canonical 3D backend;
  a `PyQtGLRenderer` ships alongside for tools that need GPU-rate
  redraws. Both implement the same `ShapeRenderer` Protocol so the
  same shape / fitter pair targets either backend.
- **Persistence**: `SegmentVizSet` / `SegmentVizSpec` define a JSON v2
  schema that round-trips bindings, shape kinds, fitter kinds, and
  themes. A `migrate_v1_to_v2` helper lifts the legacy `SegmentSet`
  shape used by the C3D Viewer into v2 on load.
- **URDF bridge**: `urdf_bridge.shape_to_urdf` / `urdf_to_shape` map
  the same shape vocabulary onto URDF `<visual>` elements, so a mesh
  imported into the Segments tab can be re-used as a URDF visual link
  without re-modelling.
- **Asset library**: `ShapeLibrary` resolves named shapes (head,
  torso, upper_arm, …) to `MeshShape` instances backed by bundled
  procedural STL meshes under `assets/body_part_shapes/default/`.

The C3D Viewer's Segments tab, the starting-pose matcher, and the
humanoid character builder all consume `body_part_viz` directly —
there is no per-tool re-implementation left.

### Generic-naming policy

The naming policy carries forward unchanged from
[ADR 0006](0006-multi-source-motion-targets.md):

- File-on-disk names (vendor-specific xlsx workbooks, named C3D
  files, the bundled STL meshes) remain whatever the source publishes
  them as.
- Everything in code, directories, and UI stays source-agnostic:
  `BodyPartShape` not `GolferBodyShape`, `ShapeLibrary` not
  `<vendor>_meshes`, `default_body_segments` not `<sport>_segments`.
- This keeps the public surface stable when the same toolkit is used
  for non-golf full-body captures (gait, climbing, generic
  biomechanics).

## Alternatives Considered

### A. Extend the C3D Viewer's segment-tab geometry directly

Keep the geometry stack inside `segments_tab.py` and grow it with
new shape kinds in place. Other tools would either copy the helpers
across or import from inside the viewer package.

Rejected. The C3D Viewer is a desktop tool, not a shared library; its
import surface pulls in PyQt6, matplotlib, and the viewer's
configuration loaders. Forcing the URDF generator (a headless,
test-friendly module) to depend on a Qt application root to draw a
cylinder is the wrong direction. The viewer's segment-tab code was
also already over the file-size budget when the epic started.

### B. Per-tool implementation

Let each tool keep its own shape stack, with a shared "spec" file
listing the conventional dimensions and colours so the three
implementations can converge by hand-coordination.

Rejected. The three implementations had already drifted before the
epic — that is the situation the epic exists to fix. A spec without
an enforcement mechanism is the same trap repeated. DRY violations
across tools are exactly the case AGENTS.md section A is meant to
catch.

### C. Reach for an external library

Reuse one of the existing biomechanics-visualisation packages
(`OpenSim`'s GUI, `meshcat`, the URDF visualiser inside `pinocchio`)
in place of writing our own.

Rejected. None of the candidates ship the combination we need: per-
shape rest dimensions, anisotropic Procrustes fitting against mocap
markers, JSON-roundtrippable segment sets, AND a URDF bridge that
preserves the same shape kinds. We would still have written the
fitters and the persistence layer; the renderer is a thin enough
matplotlib wrapper that adopting an external dependency to avoid it
costs more than it saves.

## Consequences

### Positive

- Cross-tool consistency: the C3D Viewer, the matcher, and the URDF
  generator share one source of truth for what a body segment looks
  like.
- Mesh import: a custom STL / OBJ / PLY / GLB imported in the Segments
  tab is automatically usable as a URDF visual link.
- Extensibility: adding a new shape kind means adding one module under
  `shapes/` plus an entry in `VALID_SHAPE_KINDS`; no tool has to
  change.
- The `MatplotlibRenderer` becomes the canonical 3D rendering helper
  for any new tool; tools that need GPU-rate redraws can drop in
  `PyQtGLRenderer` without changing the shape or fitter code.

### Negative

- One extra package between the tools and matplotlib. Small in
  practice — the package is pure Python with numpy + matplotlib
  imports — but it is still a layer.
- The URDF bridge encodes ellipsoid and capsule shapes as logical
  mesh filenames (`__bpv_ellipsoid__a_b_c.obj`,
  `__bpv_capsule__length_radius.obj`) because URDF has no native
  capsule and only an isotropic-scale sphere. The convention is
  documented in `urdf_bridge.py`, but it is a convention, not a
  format.

## Validation Strategy

- Every public dataclass (`MarkerBinding`, `FittedShape`, `ShapeTheme`,
  `SegmentVizSpec`, `SegmentVizSet`) runs post-init validation. Frozen
  - validated is the design contract; cost terms and renderers can
    trust the shape of what they receive.
- The Protocols are `@runtime_checkable` so tests can assert that a
  newly added shape / fitter / renderer satisfies the contract before
  the type checker sees it.
- Cross-tool integration tests pin the contract: a fixture builds a
  `SegmentVizSet`, round-trips it through JSON, runs it through the
  fitters, renders it via `MatplotlibRenderer`, and exports it to URDF
  via `urdf_bridge`. Any divergence between tools surfaces as a
  test failure, not as a runtime UI bug.
- The asset library validates the manifest schema version and the
  per-shape entries on load; an unknown schema version raises
  `ValueError` with the supported set.

## Migration

- The legacy `SegmentSet` v1 JSON shape used by the C3D Viewer is
  auto-migrated by `SegmentVizSet.from_dict` / `migrate_v1_to_v2`. No
  on-disk format change is forced on existing users; saving always
  writes v2.
- Tools that previously inlined matplotlib geometry now construct a
  `SegmentVizSet`, instantiate a `MatplotlibRenderer`, and add each
  fitted shape via `add_shape`. The shim retains the same external
  API (file open, segment list, frame slider) so user-visible
  behaviour is unchanged apart from the new shape options.
- The URDF generator path is opt-in: existing URDF outputs are
  byte-stable until the caller flips to `shape_to_urdf` for visuals.

## References

- `src/shared/python/body_part_viz/` — the package itself.
- `src/shared/python/body_part_viz/contracts.py` — the three Protocols.
- `src/shared/python/body_part_viz/persistence.py` — JSON v2 schema +
  v1 migration.
- `src/shared/python/body_part_viz/asset_library.py` — bundled mesh
  resolver.
- `src/shared/python/body_part_viz/urdf_bridge.py` — shape-to-URDF
  mapping.
- `assets/body_part_shapes/default/` — bundled procedural STL meshes
  - manifest + LICENSES.
- `docs/user_guide/body_part_viz/quickstart.md` — companion user guide.
- `docs/user_guide/body_part_viz/mesh_import.md` — custom-mesh import
  guide.
- `docs/user_guide/body_part_viz/asset_author_guide.md` — guide for
  adding new shapes to the default library.
- `docs/api/body_part_viz.md` — public API reference.
- [ADR 0006 — Multi-Source Motion Targets](0006-multi-source-motion-targets.md)
  — predecessor decision that introduced `BodyTarget` and
  `default_body_segments`, on which this toolkit builds.
