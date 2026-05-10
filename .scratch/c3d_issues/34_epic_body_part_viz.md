# [EPIC] `body_part_viz` — mesh-driven segment visualization toolkit (cross-tool, professional grade)

## Vision

Today, mocap data renders as marker dots and the C3D viewer can connect them with **lines** or **cylinders** (PR #4664). The user has asked for a **professional animation-grade visualization stack** where:

- segments display as **lines, cylinders, ellipsoids, capsules, or arbitrary mesh files** (STL / OBJ / PLY / GLB);
- mesh shapes can be **imported and stretched/fitted** to mocap dimensions (length between two markers, anisotropic scaling, Procrustes fitting to a marker cluster);
- the same toolkit drives the **C3D Viewer, Motion-Match Preview matcher, and URDF generator** — one source of truth for "what does a body segment look like";
- a **shape library** of curated default body-part meshes (head, torso, upper-arm, forearm, hand, thigh, shin, foot) ships with the repo;
- the API is **DRY, orthogonal, LOD-correct, DbC-validated**, and **TDD-covered** end to end.

This epic creates a new shared package `src/shared/python/body_part_viz/` and integrates it into the three existing surfaces.

## Architecture (high level)

```
                 ┌─────────────────────────────────────────────┐
                 │ src/shared/python/body_part_viz/            │
                 │                                             │
                 │  Contracts (Protocols + dataclasses):       │
                 │   • BodyPartShape                           │
                 │   • MarkerBinding                           │
                 │   • ShapeFitter                             │
                 │   • ShapeTheme                              │
                 │   • FittedShape                             │
                 │                                             │
                 │  Built-in shapes:                           │
                 │   • LineShape, CylinderShape,               │
                 │     EllipsoidShape, CapsuleShape,           │
                 │     MeshShape, CompositeShape               │
                 │                                             │
                 │  Built-in fitters:                          │
                 │   • BetweenTwoMarkersFitter (trivial)       │
                 │   • ClusterKabschFitter                     │
                 │   • ProcrustesFitter (anisotropic scale)    │
                 │                                             │
                 │  Renderers (backend-agnostic):              │
                 │   • MatplotlibRenderer (Poly3DCollection)   │
                 │   • PyQtGLRenderer (pyqtgraph.opengl)       │
                 │                                             │
                 │  Asset manifest:                            │
                 │   • assets/body_part_shapes/default/        │
                 │   • manifest.json                           │
                 │                                             │
                 │  Persistence:                               │
                 │   • SegmentVizSpec dataclass + JSON v2      │
                 └────────────┬────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┬─────────────────────────┐
              ▼               ▼               ▼                         ▼
      C3D Viewer       Motion-Match      URDF Generator         Future: Tauri /
      Segments tab     Preview view      visual links            React 3D view
```

## Children (13 issues)

| # | Title | Type | Priority |
|---|---|---|---|
| 1 | feat(body-part-viz): core contracts + dataclasses (Shape / Binding / Fitter / Theme) | architecture | high |
| 2 | feat(body-part-viz): primitive shapes — Line, Cylinder, Ellipsoid, Capsule, Composite | feature | high |
| 3 | feat(body-part-viz): MeshShape with STL/OBJ/PLY/GLB loaders via trimesh | feature | high |
| 4 | feat(body-part-viz): fitters — BetweenTwoMarkers, ClusterKabsch, ProcrustesAnisotropic | feature | high |
| 5 | feat(body-part-viz): MatplotlibRenderer (3D Poly3DCollection backend) | feature | high |
| 6 | feat(body-part-viz): PyQtGLRenderer (pyqtgraph.opengl backend) — performant 3D | feature | medium |
| 7 | feat(body-part-viz): asset library — default body-part meshes + manifest | feature | medium |
| 8 | feat(body-part-viz): SegmentVizSpec JSON v2 persistence (extends current SegmentSpec) | feature | high |
| 9 | feat(c3d-viewer): integrate body_part_viz into Segments tab — shape picker, mesh-import dialog | integration | high |
| 10 | feat(motion-matcher): integrate body_part_viz into live view controller | integration | medium |
| 11 | feat(urdf-generator): bind URDF link visuals to body_part_viz shapes (cross-tool reuse) | integration | medium |
| 12 | test(body-part-viz): comprehensive TDD coverage + golden snapshots | testing | high |
| 13 | docs(body-part-viz): ADR + user guide + asset-author guide | docs | medium |

## Cross-cutting principles

These apply to every child PR:

### TDD discipline

Every child issue mandates: write failing tests first, then make them pass. Tests live alongside production code (`tests/unit/body_part_viz/`, `tests/integration/body_part_viz/`). Each PR ships ≥ 80% line coverage on its new code.

### DbC (Design by Contract)

Every public callable enforces preconditions + postconditions:

- Use the existing `src/shared/python/core/contracts/decorators.py` `precondition` / `postcondition` decorators.
- Dataclasses validate in `__post_init__` and raise `ValueError`/`TypeError` with descriptive messages.
- Public functions document postconditions in their docstring.

### LOD (Law of Demeter)

- No method chains > 2 levels (`a.b.c.d()` violates).
- Public APIs return data, not widgets — UI integration code is the only place that touches Qt or matplotlib.

### DRY

- Single canonical place for: shape rendering, marker binding, fitter algorithms, asset paths, theme tokens.
- The C3D Viewer's `Poly3DCollection` cylinder code from PR #4664 is **deleted** as part of issue #9 once the toolkit absorbs it.

### Orthogonality

- A new shape can be added without touching fitters, renderers, or persistence.
- A new fitter can be added without touching shapes or renderers.
- A new renderer (e.g. WebGL) can be added without touching shapes or fitters.

This is the test of architectural success.

### Generic naming

No vendor / lab / person / study names anywhere in code, asset filenames in the default library, error messages, log messages, docstrings, or test names. The library's default human meshes ship as anonymous low-poly meshes generated from open-licensed sources (MakeHuman, OpenSim defaults, or in-house procedurally-generated capsules per #2).

### Performance budgets

- `set_frame()` (per-frame update) at ≥ 60 fps for: 26 default segments × 200-vertex meshes on a 654-frame swing.
- Mesh decimation in #3 brings huge meshes (> 10k vertices) under that budget automatically.

## Out of scope for v1

- Real-time skinning / blend-shapes (we render rigid segments only).
- Inverse kinematics on the loaded mesh (markers drive segment geometry; the mesh deforms by anisotropic scale, not by IK).
- Material / texture editing — flat shaded (or single-color toon) only in v1.
- Shadow / GI / ray-traced rendering — Lambert lighting at most.

These are explicitly punted to a v2 epic if there's appetite.

## Definition of done (epic)

- All 13 children closed.
- C3D Viewer Segments tab can pick from {line, cylinder, ellipsoid, capsule, mesh}; user can import a custom STL and bind it to a 2-marker segment with anisotropic stretch.
- Motion-Match Preview tile renders the same shapes for body markers.
- URDF Generator's link visuals are produced by `body_part_viz` (single source of truth).
- ≥ 80% line coverage across `body_part_viz/`; ≥ 70% branch.
- Performance budget met on the 26-segment driver test.
- ADR + user guide on `main`.

## Sequencing

```
1. Contracts ──► 2. Primitives ──┐
                                  │
                ── 3. MeshShape ──┤
                                  │
                ── 4. Fitters ────┼──► 5. Mpl renderer ──┐
                                  │                       │
                                  └──► 6. PyQtGL renderer ┤
                                                          │
                                                          ▼
                                                  7. Asset library
                                                          │
                                                          ▼
                                                  8. JSON v2 persistence
                                                          │
                       ┌───────────────────────────┬──────┴───────────────┐
                       ▼                           ▼                      ▼
              9. C3D Viewer integ        10. Matcher integ       11. URDF integ
                       │                           │                      │
                       └───────────────────────────┴──────┬───────────────┘
                                                          ▼
                                                  12. Tests + snapshots
                                                          │
                                                          ▼
                                                   13. Docs + ADR
```
