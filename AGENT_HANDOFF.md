# Impact Shaft Provider Integration

Canonical continuation: [docs/development/HANDOFF.md](docs/development/HANDOFF.md).
Development entry: DL-#9912. Branch: feat/9912-impact-provider-pin.
Issue #9912 is a child of #9703/#9701; PR #9916. Current commit is SELF.

The existing vendor mechanism now selects Tools 608e85b249e6f61238ac96abbe7dc37428629b9e for review.
Six new contracts fail against the old pin and pass against this candidate.
All 24 provider contracts and clean installed-wheel provider checks pass.
The Python-only wheel omits UI assets; physical qualification remains open.
Tools #5133 CI repairs, final provider revision and protected review remain.
Do not merge this consumer before the provider is reviewed and green.

## Incoming Main Integration

Main 18c8f922e is integrated into the provider branch with its capture calibration
and industrial-readiness implementations preserved. Only shared turnover/SPEC
documents conflicted; canonical HANDOFF.md retains the incoming capture handoff.
The full incoming root handoff remains at its immutable source:
[Incoming Main Handoff](https://github.com/D-sorganization/UpstreamDrift/blob/18c8f922e87c92f6f518da05c0819f71ce3193ba/AGENT_HANDOFF.md).
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
- Manuals/upstreamdrift QMD remains the engineering-manual authority.
- Numerical convergence, measured calibration and perceptual evidence are
  distinct; source hashes alone do not qualify a physical model.
- Refresh this file, canonical handoff and DL-#9912 in implementation commits.
