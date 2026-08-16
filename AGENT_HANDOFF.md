# Agent Handoff — UpstreamDrift

Last updated: 2026-08-16

This is current operational state, not a changelog. History belongs in git and
GitHub.

## Program Authority

- Epic [#8557](https://github.com/D-sorganization/UpstreamDrift/issues/8557)
  governs the proximal-to-distal research program; #8595 retains the photographed
  momentum-transfer agenda.
- #8668 governs subject-scaled articulated contact. Its inertia, contact,
  bilateral-forwarding, and typed-slack children are complete.
- #8684 governs distributed grip, shaft, and ground pathways. Grip discretization
  and the passive shaft slice are merged. Ground issue #8719 closed through
  protected PR #8723; merge `a1a613999` is verified on remote `main`.
- [#8724](https://github.com/D-sorganization/UpstreamDrift/issues/8724) is active
  on branch `research/8557-claim-adjudication-8724`, based exactly on
  `a1a613999`. This branch is the takeover authority for the checkpoint below.
- #8556 remains open: no governed participant dataset contains synchronized
  bilateral six-axis grip wrenches. Synthetic traces cannot replace it.
- NotebookLM review requires manual Google reauthentication. Never automate
  credentials, authentication dialogs, CAPTCHA, or two-factor steps. Continue
  original-source and local-corpus review without treating NotebookLM as authority.

## Qualified Modeling Baseline

- The articulated baseline has 234 closed states across six profiles, three grip
  spans, and 13 phases. MuJoCo and Pinocchio independently qualify the rigid tree.
  Broad bounds/spheres are not anatomy, clinical ranges, or coaching evidence.
- The distributed-grip atlas establishes discretization sensitivity, not measured
  pressure or benefit.
- The passive shaft atlas contains 384 trajectories and 1,536 nested summaries.
  Its speed effect changes sign, rejecting a universal passive-shaft benefit.
- The finite-ground atlas contains 384 primary and 192 control trajectories. Its
  preregistered matched screen admits 0/384 cells; a labeled post-hoc screen admits
  60 with 20 positive and 40 negative speed differences. Do not present this as a
  universal ground-pathway benefit.
- Current paper baseline: 231 pages, 1,764,016 bytes, 192 URI links, 246 outline
  entries, 1,063 narrative candidates, 295 registered claims, and 40 release claims.

## #8724 Claim-Adjudication Checkpoint

- Registry and inventory use `proximal-distal-claim-audit-v2`. All 295 claims have
  an estimand-level outcome: 275 `supported`, 5 `inconclusive`, 15 `untested`, and
  0 `contradicted`.
- Zero contradicted claims does not mean every originating hypothesis survived.
  The paper had already narrowed overbroad statements; detailed status, uncertainty,
  and falsifier fields remain essential context.
- Evidence locators now resolve by type: 3 bibliography keys, 65 DOI links,
  48 external URLs, 849 generated artifacts, 1 local anchor, and 1,134 local files.
  Missing BibTeX keys and local anchors fail closed.
- `migrate_claim_adjudication_v2.py` is a one-time source-digest-locked migration.
  Its exception sets were manually reviewed, not inferred from old status strings.
  Do not relax its digest or claim-count locks after paper edits.
- `claim_adjudication_summary.py` deterministically generates reviewer JSON/CSV.
  These are summaries, not evidence or independent human review. Existing Codex
  audits must not be described as external replication.
- Checkpoint validation passes: 28 focused tests, registry validation for all
  1,063 candidates/295 claims, and summary validation. No paper, release, PDF, or
  full pre-push gate has yet been completed for this branch.

## Takeover Sequence

1. Scientifically review the five `inconclusive` and 15 `untested` assignments;
   split composite claims where needed.
2. Add a normalized summary and limitations table to the paper without implying
   independent validation. Any paper edit changes the source digest/census: rerun
   inventory, review changed candidates, update reciprocal mappings and digest;
   never weaken the validator to make the edit pass.
3. Register summary artifacts in the release bundle and add explicit contradiction
   and source-independence gates required by #8724.
4. Run claim/evidence/release audits, title case, size checks, PDF build and visual
   inspection, then the full pre-push hook. Resolve actionable failures before a
   single consolidated push.
5. Open a ready-for-review protected PR, preserve reviews/checks, merge only when
   required contexts pass, and verify its merge commit on remote `main`.
6. Continue calibrated unilateral 3D contact, full-delivery matching/uncertainty,
   and governed human holdout under #8557. Do not close #8556 without qualifying
   participant data.

## Checkpoint Reproduction

```powershell
python -m scripts.research.proximal_distal_energy.claim_adjudication_summary write
python -m scripts.research.proximal_distal_energy.claim_adjudication_summary validate
python -m scripts.research.proximal_distal_energy.claim_audit validate
python -m pytest tests/unit/research/test_proximal_distal_claim_audit.py tests/unit/research/test_claim_adjudication_summary.py -q
```

Before publication also run:

```powershell
python -m scripts.research.proximal_distal_energy.claim_audit inventory
python -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate
python -m scripts.research.proximal_distal_energy.momentum_question_readiness validate
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/check_doc_size_budget.py
pre-commit run --hook-stage pre-push --all-files
```

Do not infer human technique, physiology, injury, timing demand, or coaching
advice; close #8556/#8557; bypass protection; force-push; admin-merge; or rerun
unchanged runner-capacity failures.
