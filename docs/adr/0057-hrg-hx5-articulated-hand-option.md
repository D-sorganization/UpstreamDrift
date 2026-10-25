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

| Option | Advantages | Limitations | Decision |
| --- | --- | --- | --- |
| One rigid capsule | Small, stable collision shape; simple to export | Roll-axisymmetric; misses palm breadth, palm inertia asymmetry, and style-specific finger contact | Reject as general option; acceptable only as a coarse fallback if its limits are explicit |
| Breadth-scaled palm plus frozen phalanx capsules | Represents palm breadth and different overlap, interlock, and ten-finger contact layouts without adding finger dynamics | Requires calibrated shape and pose data; frozen geometry cannot model finger motion during a swing | Select for the future low-complexity full-body option |
| MyoHand articulated asset | Detailed articulated anatomy and existing MuJoCo research fixture | Geometry lineage from OpenSim models is unverified; substantial joints, actuators, and licensing scope; M8 is not a full-body qualification | Do not ship or use as default until HX6 clearance and separate full-body qualification |

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

## Validation

This ADR is documentation only. Its specification was checked against the
current grip-frame contract, hand capability audit, grip-binding manifests,
and the HRG-GS-M8 provenance and limitations. No implementation acceptance is
claimed here. Future implementation commands must be recorded with that
issue's results.
