# Impact Shaft Provider Integration

Canonical continuation: [docs/development/HANDOFF.md](docs/development/HANDOFF.md).
Development entry: DL-#9912. Branch: feat/9912-impact-provider-pin.
Issue #9912 is a child of #9703/#9701; PR #9916. Current commit is SELF.

The candidate Tools pin is 4dabe900c6ef7767b565c778cda9d9449bed28cf.
UD #9916 fixes six provider-migration failures and a further runtime UI
shadow: four byte-identical files are retired through the existing split
resolver, preserving 27 UD-only widgets. Offscreen widget construction works.
All 271 source controls pass after main 90c3d0b77 integration. The isolated
32a8b36ec wheel passes provider and Qt widget checks with gui-tools installed. Realtime debt has an explicit #8942 review exception
through 2026-10-09; it is not resolved. See PROVIDER_PIN_RESULTS.json.
Tools #5133 is merged as 2c9a8d6c from 0cd6dce22. Post-merge launcher
correction #5143 needs a separate reviewed provider revision. Physical and
acoustic qualification remains open.

## Incoming Main Integration

Main 90c3d0b77 is now being integrated normally, preserving merged C3D fitting
and club/volume/handedness overlays (#9918/#9922) and attributed club catalog
(#9919). Conflicts are limited to six shared documentation/inventory files.
Both task scopes remain in canonical HANDOFF.md; its numerical/regression
evidence retains the original owners. All 271 integration controls pass.
[Incoming Main Handoff](https://github.com/D-sorganization/UpstreamDrift/blob/90c3d0b770cd0ced3e14c947c68a29555f14e877/AGENT_HANDOFF.md).

Main 8fce9f238 is integrated after capture PR #9917 merged. Its LoD fixes and
capture journey implementation are preserved. Only the root handoff and
development log conflicted; both owners' entries remain. Canonical HANDOFF.md
retains the incoming capture handoff.
The full incoming root handoff remains at its immutable source:
[Incoming Main Handoff](https://github.com/D-sorganization/UpstreamDrift/blob/8fce9f238ce89876dd363fb41b4ba1169a87d1b6/AGENT_HANDOFF.md).
The capture setup, catalog and wizard work stays with its original owners.

## Preserved Repository Context

The previous 516-line root handoff is preserved at the immutable base:
[Full Base Handoff](https://github.com/D-sorganization/UpstreamDrift/blob/6e3610a9b/AGENT_HANDOFF.md).
Its details remain historical evidence, not new completion claims.
Existing capture-product, reference-comparison, instructor, fleet-guide and
Ubuntu CI work belongs to its original owners; preserve those changes.
The canonical handoff retains their incoming text below the current section.

## Standing Constraints

- Read AGENTS.md and CLAUDE.md; maintain issue leases and session presence.
- Shared physics belongs in Tools. Do not copy or edit vendored source.
- Use topic branches and normal hooks; preserve protected CI and reviews.
- UP-D0 (#9066) and UP-D1 (#9067) remain a separate design-manual program.
  manuals/upstreamdrift QMD is the editable authority; generated LaTeX, PDF,
  DOCX and HTML remain non-editable artifacts. Run the configured
  scripts.check_design_manual_governance checks before changing calculations.
- Numerical convergence, measured calibration and perceptual evidence are
  distinct; source hashes alone do not qualify a physical model.
- Refresh this file, canonical handoff and DL-#9912 in implementation commits.
