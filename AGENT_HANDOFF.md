# Impact Shaft Provider Integration

Canonical continuation: [docs/development/HANDOFF.md](docs/development/HANDOFF.md).
Development entry: DL-#9912. Branch: feat/9912-impact-provider-pin.
Issue #9912 is a child of #9703/#9701; PR #9916. Current commit is SELF.

The existing vendor mechanism selects Tools 00d17e7f91fe8541bc8882ee745fda58ee2ad7af for review.
The repaired provider/theme/fallback/manual contract set passes all 79 tests.
The obsolete theme color child is retired; UD-only theme modules stay local.
Realtime has a split/pending-cleanup ruling tied to #8942; its API is preserved.
The old 608e85b24 Python-only wheel remains historical installation evidence.
A clean wheel for the current candidate, protected review and final pin remain.
Tools #5133 has a Python 3.11 momentum-oracle failure under investigation.
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
- UP-D0 (#9066) and UP-D1 (#9067) remain a separate design-manual program.
  manuals/upstreamdrift QMD is the editable authority; generated LaTeX, PDF,
  DOCX and HTML remain non-editable artifacts. Run the configured
  scripts.check_design_manual_governance checks before changing calculations.
- Numerical convergence, measured calibration and perceptual evidence are
  distinct; source hashes alone do not qualify a physical model.
- Refresh this file, canonical handoff and DL-#9912 in implementation commits.
