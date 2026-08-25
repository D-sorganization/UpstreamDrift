# Agent Handoff — UpstreamDrift

Last updated: 2026-08-24

This file records current operational state, not history. Git and GitHub retain
history. Epic #8557 is the canonical proximal-to-distal completion authority.

## Repository Authority

- UpstreamDrift owns scientific sources, models, evidence registers, and the
  release bundle. AffineDrift is an immutable revision-pinned public
  projection. Tools owns reusable consumers; do not copy implementations into
  this repository or edit `vendor/ud-tools`.
- Remote `main` is `3a5b9d630b38b6e017a3568dc638b22f9839b3c0`.
- The computational paper is 239 pages with SHA-256
  `be85b7b62bba060a26ce3fea8355aa8b01dcf8c1b1ccf09304450898a4e5e78b`.
  Archival qualification remains false because the PDF is untagged and retains
  Type 3 and unembedded font resources.

## Numeric Claim Audit (#8918)

- Worktree: `UpstreamDrift-worktrees/8918-numeric-claim-audit`; branch:
  `fix/8918-numeric-claim-audit`; base: exact remote `main` above.
- Status: protected PR #9042 is open for required human review. Its first full
  protected run passed 11,816 tests except the deterministic adjudication
  freshness gate. The canonical summary, 1,100-candidate inventory, reviewed
  migration lock, integrity records, and release hashes are now refreshed;
  73 focused tests pass serially.
- All 303 material claims are covered. The 124 numeric claims contain 380
  numeric literals, each bound to a reviewed JSON Pointer, transform, scope,
  and tolerance. Exact statement digests and literal inventories fail closed.
- Evidence scopes deliberately distinguish 172 semantically matched local JSON
  values, 144 registered claim values not independently recomputed, 57
  reported external values, and seven protocol or notation values.
- Pointer agreement is not physical validation. Four representative headline
  tests independently recompute planar, spatial, articulated-shaft, and
  finite-ground quantities from committed CSV/NPZ arrays.
- Cross-engine spatial comparison uses independently stored arrays and must be
  close but nonidentical. Exact-zero parity is rejected as degenerate.
- Release validation covers 2,232 evidence references, 321 local artifacts, 78
  external URLs, and a 600-artifact computational bundle.
- The maintainer scaffold is not a release generator. It proposes conservative
  semantic matches for protected review and falls back to explicitly weaker
  ledgers rather than inventing source authority.

## Completed and Incomplete Campaigns

- #8752 completed on ControlTower at source commit
  `13146cdcece879e7156e06e2dca6626c1a54e045`; terminal evidence is committed on
  `origin/research/8752-articulated-uncertainty` at
  `2fa6cf8861eeaf7ae111dd8dd18c4053a9f82e65`. The computation completed, but
  terminal publication integration is not yet an ancestor of remote `main`.
- #8800 did not complete. The registered campaign requires 830 atomic
  checkpoints across 83 feasible states: 332 shaft and 498 ground. Current
  evidence is 93/830: nominal shaft 48/48 and nominal ground 45/72. No campaign
  Python process is running; persisted `state=running` is stale and
  `release_evidence=false`.
- Preserve #8800 source, plan identities, 93 checkpoints, and the corrupted
  ControlTower WSL VHDX. Do not claim three missing branches or a 45/48 ground
  denominator; 737 registered checkpoints remain absent.

## Human-Evidence Boundaries

- #9004 remains open because no governed participant trajectory dataset or
  held-out human outcome is registered. Simscape exports, fixtures, tutorials,
  GolfDB labels, and launch-monitor records are not substitutes.
- #8556 remains externally blocked by the absence of governed synchronized
  bilateral six-axis grip-wrench participant data.
- Synthetic evidence never substitutes for unavailable human validation.
  Model results do not establish anatomy, physiology, equipment calibration,
  injury, coaching strategy, or a universal speed benefit.

## Scientific Boundaries

- Distinguish energy transfer, momentum redistribution, joint work, constraint
  forces, and clubhead speed. Preserve falsifiers, adverse cases,
  identifiability limits, uncertainty, countermodels, and unavailable states.
- A supported registry outcome means only that the declared estimand survives
  its registered evidence and boundary. It may still describe null, mixed, or
  adverse findings and does not imply theory confirmation.

## Repository and Review Rules

- PRs target `main`; use full PRs, never drafts. Human review is required.
- Never force-push, admin-merge, bypass hooks/checks, add quarantine debt, or
  edit generated/vendor authority by hand.
- Use TDD, DbC, DRY, and LoD. Edit canonical sources, regenerate governed
  artifacts, and use Title Case for document headings and captions.
- Run resource-intensive tests serially on DeskComputer. Dispatch checkpointed
  campaigns to ControlTower only through reviewed, recoverable plans.
- Verify exact PR head, reviews, checks, merge SHA, remote-main ancestry, and a
  clean worktree before reporting completion.

## Focused Validation

```powershell
python -m scripts.research.proximal_distal_energy.build_claim_numeric_comparison_evidence check
python -m scripts.research.proximal_distal_energy.register_numeric_claim_evidence check
python -m scripts.research.proximal_distal_energy.claim_audit numeric
python -m scripts.research.proximal_distal_energy.claim_audit validate
python -m scripts.research.proximal_distal_energy.claim_adjudication_summary validate
python -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate
python -m scripts.research.proximal_distal_energy.release_claim_review validate
python -m scripts.research.proximal_distal_energy.external_source_review validate
python -m scripts.research.proximal_distal_energy.qualify_open_release validate --source-revision (git rev-parse HEAD) --publication-profile computational
python -m pytest tests/unit/research/test_proximal_distal_claim_audit.py tests/unit/research/test_claim_adjudication_summary.py tests/unit/research/test_claim_numeric_audit.py tests/unit/research/test_register_numeric_claim_evidence.py tests/unit/research/test_scaffold_numeric_claim_contracts.py tests/research/test_claim_numeric_registry.py tests/research/test_claim_headline_recomputation.py -q -n 0 --timeout=120
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/ci/check_file_size_budget.py
```

Passing shared gates does not close a scientific child whose narrower evidence
or governed external-data requirement remains incomplete.
