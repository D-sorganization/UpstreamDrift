# ADR-0027: Canonical 3D Viewport Backend

- Status: Accepted
- Date: 2026-05-31
- Decision Makers: Canonical Core maintainers
- Related Issues/PRs: #6806, #6772, ADR-0008, ADR-0026

## Context

CC-33 needs a shared 3D viewport for canonical-v2 trajectories, marker
keypoints, and GRF/wrench overlays. The viewport must be engine-agnostic and
must not make Rerun, MeshCat, and VTK hard dependencies of the core package.

The repo already has MeshCat usage in Drake/Pinocchio-adjacent workflows and a
`meshcat>=0.3.0` dependency in the optional `pinocchio` extra. ADR-0008 keeps
body-part shape rendering separate from heavyweight viewer runtimes, while
ADR-0026 defines the world Z-up, SI canonical-v2 frame used by the overlay
payload.

## Decision

Use **MeshCat** as the default CC-33 viewport backend and add a small provider
evaluation layer in `src/shared/python/visualization/viewport.py`.

The layer records metadata for MeshCat, Rerun, and VTK/PyVista; checks provider
availability lazily via import discovery; and returns explicit degradation
reasons when an optional viewer is not installed. The layer also defines a
backend-neutral `ViewportOverlayPayload` built from the existing Trace v2
schema:

- `q[:, :3]` as the canonical-v2 base trajectory when present.
- `Trace.markers` as marker keypoint overlays.
- `Trace.contacts` as optional contact anchor points.
- `Trace.wrench` as GRF/wrench overlay data.
- `convention="canonical-v2"`, `frame="world_Zup"`, and `units="SI"`.

Concrete rendering adapters remain follow-up work so app-shell work (#6805) can
own embedding and lifecycle wiring.

## Alternatives Considered

1. **Rerun default.** Rerun has the strongest timeline/logging model and remains
   a good future diagnostic export target. It is not currently used by the
   engine stack, and making it the first embedded viewport would add a new
   runtime before the app shell is ready.
2. **VTK/PyVista default.** VTK is mature and strong for scientific meshes and
   offscreen rendering. Its native dependency footprint is heavier, and it is
   less aligned with existing Drake/Pinocchio MeshCat workflows.
3. **No provider decision yet.** Deferring the decision would leave engine
   adapters without a stable payload target and encourage per-tool viewport
   branching.

## Consequences

- Positive: MeshCat reuses the existing optional dependency path, works for
  Drake/Pinocchio-native workflows, and can be embedded through web views.
- Negative: MeshCat is less durable than Rerun as a persisted analysis log.
- Follow-ups: add the concrete MeshCat renderer after #6805 settles the app
  shell host; add Rerun export if recorded review artifacts become a first-class
  workflow; keep VTK/PyVista behind opt-in tool environments.

## Validation

Unit tests cover provider metadata, lazy selection, missing-dependency
degradation, requested-provider behavior, and Trace-to-payload validation for
trajectory, markers, contacts, and wrench overlays.
