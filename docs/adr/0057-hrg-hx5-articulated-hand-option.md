# ADR-0057: HRG-HX5 Articulated-Finger Hand Option

- Status: Accepted
- Date: 2026-10-25
- Decision Makers: Issue Remediator (specification; implementation remains gated)
- Related Issues/PRs: #13762, #13757, #13330

## Context

The current flagship hand representation has no finger degrees of freedom and
uses the grip contract documented in
[`HANDS_AUDIT.md`](../development/full_body_models/HANDS_AUDIT.md). A later
articulated-hand implementation must compare geometry, inertial properties,
and grip contact without changing that contract. Issue #13762 is a docs-only
specification spike. It does not authorize shipping runtime geometry.

The repository also contains the bilateral MyoHand research fixture from
HRG-GS-M8 (#13330). That fixture is an isolated two-hand and club contact
experiment; it is not a qualified full-body golfer hand implementation.
Its source pin and generated-scene hashes are recorded in the
[HRG research record](../research/human_realistic_golfer/human_realistic_golfer.tex).
The MyoSuite wrapper is Apache-2.0; the underlying geometry lineage from
OpenSim models has not been verified. HX6 licensing clearance remains a gate
before any distribution or reuse decision involving those assets.

## Decision

For a future full-body option, prefer a breadth-scaled palm box or ellipsoid
with phalanx capsules placed in frozen poses for each supported grip style.
Keep the existing grip frame and hand attachment contract unchanged. Treat
this representation as a rigid, style-specific collision proxy: the finger
poses do not add runtime finger degrees of freedom or muscle actuation.

Do not select a single rigid capsule as the general hand proxy. A capsule is
roll-axisymmetric and cannot represent palm breadth or its asymmetric inertia.
Do not select MyoHand as the shipping default while its geometry lineage is
unverified. The existing M8 fixture remains research evidence only and does
not satisfy this ADR's acceptance spec by itself.

## Alternatives Considered

| Option                                           | Advantages                                                                                                              | Limitations                                                                                                                                 | Decision                                                                                  |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| One rigid capsule                                | Small, stable collision shape; simple to export                                                                         | Roll-axisymmetric; misses palm breadth, palm inertia asymmetry, and style-specific finger contact                                           | Reject as general option; acceptable only as a coarse fallback if its limits are explicit |
| Breadth-scaled palm plus frozen phalanx capsules | Represents palm breadth and different overlap, interlock, and ten-finger contact layouts without adding finger dynamics | Requires calibrated shape and pose data; frozen geometry cannot model finger motion during a swing                                          | Select for the future low-complexity full-body option                                     |
| MyoHand articulated asset                        | Detailed articulated anatomy and existing MuJoCo research fixture                                                       | Geometry lineage from OpenSim models is unverified; substantial joints, actuators, and licensing scope; M8 is not a full-body qualification | Do not ship or use as default until HX6 clearance and separate full-body qualification    |

## Acceptance Specification for Later Implementation

This child issue ships no geometry, collision shapes, mass properties, or
runtime behavior. A follow-up implementation must pass all of the following
before the option is enabled:

1. **Mass and Inertia Parity.** Record the source hand mass, center of mass,
   and inertia tensor in declared units and frames. The exported palm and
   phalanx bodies must preserve the approved hand aggregate mass and inertia
   target. Compare analytic properties with each compiled engine model using
   the existing model-parity tolerances; do not widen tolerances to make a
   variant pass. Tests must cover both lead and trail hands and each supported
   grip style.
2. **Collision Geometry.** Validate finite positive dimensions, valid body
   frames, and non-degenerate collision shapes. Each configured overlap,
   interlock, and ten-finger pose must have an explicit shape and contact
   mapping. Check the intended hand-to-grip and hand-to-hand pairs and reject
   unintended self-collision pairs.
3. **Contact-Patch Behavior.** Exercise neutral and perturbed grip poses.
   Contact must occur on the declared palm/finger patches, remain unilateral,
   and satisfy the engine's existing force and friction contracts. Separated
   shapes must report no contact. Include mirrored left/right fixtures and
   driver and iron clubs where the backend supports both.
4. **Grip-Contract Preservation.** The existing HX1 grip-frame tests in
   `tests/unit/motion_matching/test_grip_contract.py` must pass unchanged for
   every variant. New tests must verify each emitted hand frame equals the
   existing contract transform; shape changes may not change frame origin,
   orientation, or sign conventions.
5. **Future-Implementation Test List (TDD).** Add failing tests first for
   aggregate mass/inertia parity, collision-pair registration and exclusion,
   contact-patch behavior, left/right mirroring, and HX1 frame invariance.
   Preserve the first failing test output in the implementation PR. The
   existing M8 fixture test is useful precedent, but passing it alone is not
   acceptance evidence for full-body integration.
6. **Asset Provenance Gate.** For any third-party asset, record the exact
   upstream revision, source files and hashes, wrapper license, geometry
   lineage, and redistribution rights. MyoHand remains excluded until HX6
   verifies the OpenSim-derived geometry lineage and authorizes its use.

The implementation report must distinguish a contact fixture, inverse or
feedback-assisted tracking, and independently replayed open-loop dynamics.
An animation or successful test process alone does not establish physical
acceptance.

## Consequences

- Positive: the future option can express palm breadth and style-specific
  contact while preserving the established golfer grip frame.
- Negative: it needs explicit geometry and pose calibration; frozen poses do
  not represent finger articulation during motion.
- Follow-up: create a separately scoped runtime issue after HX6 is resolved.
  That issue must implement the tests above before adding model geometry.

## Proposed Addendum (HRG-GB, Epic #15122 / GB-1 #15123)

**Status within this addendum:** Proposed (moves to Accepted when GB-3 builder
goldens pass).

1. **Authoring contract.** A versioned, engine-free `GripSpec`
   (`src/shared/python/grip_spec/`) is the single source for grip geometry
   parameters (per-hand club cross angle, palm-versus-fingers depth, thumb
   azimuth, style, handedness, and hand size). Legacy
   `GripPose` remains the compatibility projection via
   `GripSpec.to_grip_pose()`; existing receipts and default meshes stay
   unchanged when `GripSpec.legacy()` is used.
2. **Export format.** The procedurally built, frozen phalanx capsules per grip
   style are the export format for the breadth-scaled palm option. Runtime
   finger degrees of freedom during the swing remain out of scope for this ADR.
3. **Acceptance gate.** This addendum becomes Accepted only after GB-3 golden
   builder poses pass with checksum evidence recorded on the epic; until then,
   no full-body geometry change is authorized by this addendum alone.

## Proposed Amendment (Epic #15150): Shared `hand_backends` Interface

**Status within this amendment:** Proposed. It stays Proposed until the hand
geometry is fixed (owner direction on epic #15150). It authorizes a visual
interface only. It does not change the decisions above, the HX1 grip frame,
or any contact or collision geometry.

**Why.** Seven parallel hand tracks (#15143 Shadow, #15147 MyoHand, #15146
procedural skin and comparison harness, #15161 SDF, #15153 Allegro, #15149
skinned human, #15164 poser) each carry their own copy of
`hand_backends/__init__.py` and `registry.py`. The copies disagree: some
overwrite a duplicate name and some raise; each declares a different default;
the result types differ (`material_rgba`, `skin_rgb`, `dict[str, Mesh]`). They
cannot all merge. This amendment fixes one interface. Every track is kept and
adopts it; there are no winners or losers.

### A1. Package and scope

- The package is `src/shared/python/model_appearance/hand_backends/`. It holds
  visual hand meshes only.
- **Contact geometry stays on capsules.** The capsule pads built by
  `grip_hand_mesh` and `grip_contact_patch` remain the only hand contact and
  collision geometry for every engine. A backend returns a visual mesh. It
  never returns, replaces or changes contact geometry, mass properties or
  engine models.
- **`posing.py` is not part of this interface.** The shared curl-to-contact
  poser (`hand_backends/posing.py`) is owned by PR #15149 (the skinned human
  track) and stays on that branch. The registry and its conformance test do
  not import it, and no other PR edits or moves it. A backend may use it once
  it lands on `main`, through its `PosableHand` protocol.

### A2. Backend protocol

A backend is a module that registers one `HandVisualBackend` record and one
builder function.

`HandVisualBackend` (frozen dataclass):

| Field             | Meaning                                                                                                                                                                     |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`            | Unique, lower-case `snake_case` identifier, e.g. `skinned_human`, `shadow`, `myohand`. It equals `HandVisualResult.backend`.                                                |
| `builder`         | The build function below.                                                                                                                                                   |
| `description`     | One line for harness legends and logs.                                                                                                                                      |
| `capabilities`    | `frozenset[str]` of declared extras, e.g. `"explicit_joint_angles"`, `"vertex_normals"`, `"self_posing"`. Informational; the conformance test checks only what is declared. |
| `default_enabled` | `True` when the backend appears in default listings and comparison sheets. `False` keeps it registered but opt-in.                                                          |

Builder signature (keyword-only after `joint_angles`):

```python
def build_hand_visual(
    grip_spec: GripSpec,
    side: HandSide,                     # "lead" | "trail"
    joint_angles: HandJointAngles | None = None,
    *,
    grip_radius_m: float,
    skin_rgb: tuple[float, float, float] | None = None,
) -> HandVisualResult: ...
```

Build or load work happens on the first call, not at import or registration
time: a backend that loads assets (MJCF, STL, `.npz`) does so lazily and may
cache the result. Registering a backend must not import MuJoCo or read asset
files.

`HandVisualResult` (frozen dataclass) carries the fields every renderer needs:

- `backend: str` and `side: HandSide`;
- `mesh: Mesh`, the visual skin in the **hand-grip frame** (the frame of
  `grip_hand_mesh`; the shaft axis is `APPEARANCE_GRIP_SHAFT_AXIS`, +z);
- `skin_rgb: tuple[float, float, float]`, each channel in `[0, 1]`;
- `grip_radius_m: float`, the radius the hand was built around.

It may also carry these optional audit fields, all defaulting to `None` or
empty: `vertex_normals`, `joint_angles`, `contact_gaps_m` (visual gaps only,
not contact), and `frame_focus` (vertices a renderer frames on). Backends
that need more audit data keep it in a backend-specific result field or
subclass, not in the shared type.

**Joint and finger naming.** Joint angles are anatomical hinge flexion in
radians keyed by the joint's distal bone, as in PR #15149: fingers are
`index`, `middle`, `ring`, `little`; `<finger>_prox` is the MCP, `<finger>_mid`
the PIP and `<finger>_dist` the DIP; `meta_<finger>` is a metacarpal arch;
`thumb_prox` and `thumb_dist` are the thumb MCP and IP and `thumb_press` the
thumb base pressing onto the grip. Zero is a straight digit and positive
curls toward the palm. `joint_angles=None` means the backend poses itself for
`grip_spec`. A backend whose native rig uses other names (Shadow `FFJ*`,
Allegro joint indices, MyoHand MJCF joints) maps from these names inside the
backend.

**Per-hand rotation about the shaft axis.** Each hand is rotated about the
club shaft axis by `resolved_shaft_roll_deg(grip_spec, side)` from
`grip_spec.shaft_roll` (GB1b, #15139), which includes left-handed mirroring.
The roll is a rigid rotation about `APPEARANCE_GRIP_SHAFT_AXIS` through the
hand-grip origin, applied after posing, so a rolled build equals the unrolled
build rotated by that angle. Backends do not re-solve the pose for roll.

**Skin colour.** `skin_rgb=None` uses the backend's settings default. An
explicit `skin_rgb` is returned unchanged in `HandVisualResult.skin_rgb`.
Colour defaults live in the backend's settings file, not in code.

**Resizing.** Hand size comes only from `grip_spec.hand_size` (`length_m`,
`breadth_m`). A larger `length_m` must give a longer hand. Backends must not
take a second size parameter.

**Handedness.** `grip_spec.handedness == "left"` mirrors the hands. The mirrored
lead hand has the same extent as the right-handed lead hand.

### A3. Registry API

`hand_backends/registry.py` is the single registry, re-exported from
`hand_backends/__init__.py`:

- `register_hand_visual_backend(backend: HandVisualBackend) -> None` raises
  `ValueError` on a duplicate name (no silent overwrite) and on a name that
  is not lower-case `snake_case`.
- `get_hand_visual_backend(name: str) -> HandVisualBackend` raises `KeyError`
  for an unknown name; the message lists the registered names.
- `list_hand_visual_backends(*, include_disabled: bool = False) -> tuple[str, ...]`
  returns sorted names; disabled backends are listed only on request.
- `build_hand_visual(grip_spec, side, joint_angles=None, *, grip_radius_m,
backend, skin_rgb=None) -> HandVisualResult` dispatches by name. A disabled
  backend still builds when it is named explicitly.

Backends are registered lazily: the registry holds a tuple of built-in backend
module paths and imports each one the first time the registry is read. A track
adds its backend by appending one module path to that tuple and calling
`register_hand_visual_backend` from its module's `register()` function. The
registry declares no default backend; callers name the backend they want,
and the capsule mesh stays the default drawn hand until a later decision.

**MyoHand is disabled by default.** The name `myohand` is registered with
`default_enabled=False`, and the registry rejects any attempt to register it
enabled. MyoHand stays research-only until HX6 (#13763) clears its geometry
lineage (see the Decision above and acceptance item 6).

### A4. Conformance contract

A parametrised test,
`tests/unit/model_appearance/hand_backends/test_hand_backend_conformance.py`,
runs against every registered backend, including disabled ones. A backend
merges only if it passes all of the following for `GripSpec.legacy()` and a
default grip radius, for both `lead` and `trail`:

1. **Identity.** `result.backend == backend.name` and `result.side == side`.
2. **Mesh validity.** At least four vertices; finite `(N, 3)` vertices; integer
   `(M, 3)` faces indexing inside the vertex array; no degenerate
   (zero-area) faces; a non-zero bounding box on all three axes.
3. **Frame.** The mesh lies near the grip: its centroid is within one hand
   length of the shaft axis in the hand-grip frame.
4. **Skin colour.** The default `skin_rgb` is in `[0, 1]^3`; an explicit
   `skin_rgb` is returned unchanged.
5. **Shaft roll.** Building with `shaft_roll_deg = +20` equals the roll-0
   build rotated by `+20` degrees about `APPEARANCE_GRIP_SHAFT_AXIS`
   (vertex-wise, within a tolerance setting); the same holds for the trail
   hand independently.
6. **Resizing.** A larger `hand_size.length_m` gives a strictly larger extent
   along the hand's long axis.
7. **Handedness.** The left-handed lead hand's bounding-box extents match the
   right-handed lead hand's within a relative tolerance setting.
8. **Determinism.** Two identical builds give identical vertices and faces.
9. **Registry hygiene.** The name is lower-case `snake_case` and unique;
   `myohand`, if registered, is not default-enabled.

These are structural checks. They do not judge appearance. The visual bar
(separate fingers, an opposable thumb, a cupped and arched palm, correct
per-hand roll, working skin colour and resizing), the render set and the
owner sign-off from epic #15150 still apply to every visual PR, alongside
this test.

### A5. Delivery

1. This amendment (docs only).
2. One small registry PR: the conformance test first (red), then the minimal
   `__init__.py`, `registry.py` and shared types to turn it green, plus a
   test-only reference backend. No backend implementations and no
   `posing.py`.
3. After the registry merges, the stalled tracks are revived one at a time,
   each rebased onto `main` and adopting the registry in place of its own
   copy: Shadow #15143, MyoHand #15147, procedural skin and comparison
   harness #15146, SDF #15161, Allegro #15153. #15149 and #15164 adopt it on
   their own branches; #15149 keeps `posing.py`.

## Validation

This ADR is documentation only. Its specification was checked against the
current grip-frame contract, hand capability audit, grip-binding manifests,
and the HRG-GS-M8 provenance and limitations. No implementation acceptance is
claimed here. Future implementation commands must be recorded with that
issue's results.
