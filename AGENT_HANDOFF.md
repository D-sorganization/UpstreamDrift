# Impact Shaft Provider Integration

Canonical continuation: [docs/development/HANDOFF.md](docs/development/HANDOFF.md).
Development entry: DL-#9912. Branch: feat/9912-impact-provider-pin.
Issue #9912 is a child of #9703/#9701; PR #9916. Current commit is SELF.

The existing vendor mechanism selects Tools 00d17e7f91fe8541bc8882ee745fda58ee2ad7af for review.
The repaired provider/theme/fallback/manual contract set passes all 79 tests.
The obsolete theme color child is retired; UD-only theme modules stay local.
Realtime has a split/pending-cleanup ruling tied to #8942; its API is preserved.
The efe44846e Python-only wheel at this pin installs cleanly: shaft wire,
source tamper refusal, theme/layout/realtime ownership and pip check pass.
The 79-contract suite also passes after main 8fce9f238. Protected review remains.
Tools #5133 has a Python 3.11 momentum-oracle failure under investigation.
Do not merge this consumer before the provider is reviewed and green.

## Incoming Main Integration

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
