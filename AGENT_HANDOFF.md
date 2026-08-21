# Agent Handoff — UpstreamDrift

Last updated: 2026-08-21

This is current operational state, not a changelog. Epic #8557 is the canonical
proximal-to-distal completion authority.

## Verified Remote State

- UpstreamDrift `origin/main` is
  `c1d23413cc5b2079d9fff2174e1059d1df725804`; PR #8962 and its 235-page
  scientifically bounded release remain ancestors.
- Tools PR #4623 merged through human review as
  `b886d4373fae6b435fe223b7948feda6e806cd64`, verified as Tools remote `main`.
  It provides the dependency-owned functional QA requested by UpstreamDrift.
- The UpstreamDrift #8963 branch has merged current `origin/main`; do not rebase
  or force-push after publication.

## Active Architecture-Debt Closure (#8963)

- Worktree: `UpstreamDrift-worktrees/8963-orchestrator-decomposition`.
- Branch: `fix/8963-orchestrator-decomposition`; merge commit `5318c8154`
  incorporates current remote main after seven scoped commits.
- Decomposed below 100 lines without changing registered behavior:
  `register_articulated_ground_claims.py`,
  `register_articulated_shaft_claims.py`,
  `register_remaining_claim_reviews.py`,
  `register_subject_scaled_spatial_geometry_claims.py`,
  `run_spatial_forward_contact_study.py`, and `spatial_full_body.py`.
- All seven temporary #8963 exceptions are removed from
  `scripts/config/architecture_budget.json`. A unit test prevents their return.
- Exact isolated registry comparisons pass:
  ground `1,321,917` bytes, remaining review `1,320,846` bytes, and
  subject-scaled review `1,318,516` bytes. Never run historical registrars
  directly against the canonical registry; compare temp snapshots.
- Changed-file architecture gate passes against `origin/main`; architecture
  unit suite passes 7/7. Full-tree `--all` remains red on unrelated legacy debt
  and is diagnostic, not the protected changed-file gate.
- Claim audit, evidence-integrity, release-review, and release-bundle suites
  pass 29/29. The qualified release writer passed and refreshed only the six
  changed source attestations and checksum lines.
- Four focused spatial model/action-reaction/geometry controls passed before
  the live campaign saturated the workstation.

## Required #8963 Finish

1. Wait for PID `18404` to release CPU; do not compete with or restart it.
2. Run the governed spatial-forward generator, compare JSON excluding source
   hashes and compare all NPZ arrays to the committed baseline, then retain the
   regenerated source digest only if numerical evidence is equivalent.
3. Regenerate claim evidence, release manifest, and checksums through
   `qualify_open_release write`; never hand-edit governed evidence.
4. Re-run spatial-forward, cross-formulation, claim/release, publication, Ruff,
   changed-file architecture, title, size, and SPEC-freshness gates.
5. Update this handoff, commit, push a full PR closing #8963, request
   `dieterolson`, and shepherd protected CI without auto-merge or bypass.

The current spatial-forward evidence test fails only because
`run_spatial_forward_contact_study.py` now hashes to
`3e5c8165037dc3baeb0b1152b19775d9357f440710b0612b064cac997a03bca7`
while the governed JSON retains the original source digest. The expensive
cross-formulation test timed out under campaign CPU saturation without an
assertion failure. Do not treat either result as numerical evidence.

## Publication Baseline Defect

- Remote main's unchanged canonical PDF is 235 pages, SHA-256
  `ce51e6fe4f3d9033bf730c0fe2538c72bf88b1b9707f77a7b6385923a1b5fdcf`,
  with 235 successful renders, 194 valid URI links, and 246 outline entries.
- `test_publication_quality.py`, `OPEN_RELEASE_QUALIFICATION.md`, and
  `PUBLICATION_QUALITY.md` still say 233 pages; the latter also retains an old
  PDF digest and 192-link count. This defect exists on `origin/main` and is not
  caused by #8963. Issue #8977 owns a separate repair with a reconciled
  visual-QA record; do not conceal it inside the behavior-preserving PR.

## Articulated Uncertainty Campaign (#8752)

- Worktree: `UpstreamDrift-worktrees/goal-8752-uncertainty`.
- Parent PID `18404` and its 20 children are active, not orphaned. The campaign
  has 19 registered corners and persists completed pathways plus atomic,
  digest-bound ground branches.
- Current durable state: nine fully terminal corners; corner 10
  (`shaft_damping_ratio-low`) has its shaft result and one of 72 ground branches
  saved. The top-level record remains `in_progress`.
- Do not kill individual workers, clean/delete the worktree, edit campaign
  sources, or launch another campaign. An interruption preserves terminal
  pathways and completed branches but repeats in-flight branches.
- Atomic top-level checkpoint hardening exists separately at commit `9f850a67f`
  and must be integrated only after the source-pinned campaign result lands.

## Scientific Boundaries

- The model ladder is synthetic and model-conditional. It does not establish
  participant mechanics, anatomy, physiology, equipment calibration, injury,
  coaching strategy, or a universal speed benefit.
- #8556 remains externally data-gated: no governed participant dataset with
  synchronized bilateral six-axis grip wrenches is available. Never substitute
  synthetic traces for human validation.
- Retained failures, adverse cases, nonzero cross-engine discrepancies, and
  identifiability limits are evidence; never optimize them away post outcome.

## Repository Rules

- Full PRs target `main`; never drafts. Human review is mandatory. Never enable
  auto-merge, force-push, admin-merge, bypass checks/hooks, or edit
  `vendor/ud-tools`.
- Use TDD, DbC, DRY, and LoD. Stage explicit paths only. Keep this handoff below
  150 lines and update it on every push/PR transition.
- Research outputs are hash-pinned. Use governed generators rather than
  hand-editing JSON, NPZ, figures, manifests, checksums, or claim records.
- Document titles and captions use title case.

## Focused Commands

```bash
python3 scripts/ci/check_architecture_budget.py --base-ref origin/main
python3 -m pytest -q -n 0 tests/scripts/wave10_quality/test_check_architecture_budget.py
python3 -m pytest -q -n 0 tests/unit/research/test_proximal_distal_claim_audit.py \
  tests/unit/research/test_proximal_distal_claim_evidence_integrity.py \
  tests/unit/research/test_release_claim_review.py \
  tests/research/test_proximal_distal_release_bundle.py
python3 -m scripts.research.proximal_distal_energy.qualify_open_release validate \
  --source-revision "$(git rev-parse HEAD)" --publication-profile computational
python3 scripts/check_document_title_case.py --changed-from origin/main
python3 scripts/check_doc_size_budget.py
```

Passing common gates does not close a child issue whose scientific criteria
remain unmet. Verify exact PR head, human review, checks, merge SHA, and
remote-main ancestry before reporting protected completion.
