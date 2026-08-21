# Agent Handoff — UpstreamDrift

Last updated: 2026-08-21

This is current operational state. Historical detail belongs in git and GitHub.

## Program Authority and Critical Findings

- Epic [#8557](https://github.com/D-sorganization/UpstreamDrift/issues/8557)
  is the canonical proximal-to-distal completion authority; #8595 retains the
  photographed agenda and #8789 owns release/CI truth recovery.
- The 2026-08-21 adversarial reviews are indexed in
  `docs/audits/2026-08-21-uiux-performance-review.md` and
  `docs/audits/2026-08-21-adversarial-integration-review.md`.
- #8909 found that the distributed-grip atlas's claimed MuJoCo–Pinocchio parity
  compared MuJoCo to itself. #8910 found tautological manufactured controls in
  the current #8752 evidence. Do not close #8751/#8752 or promote their claims
  until independently regenerated evidence passes the corrected gates.
- #8913/#8914 bound interpretation of the mixed-sign shaft and ground results.
  A moving base, compliant shaft, or higher proximal rate is not a universal
  benefit.
- #8556 remains externally blocked: no governed participant dataset contains
  synchronized bilateral six-axis grip wrenches. Synthetic traces cannot
  replace human validation.

## Active #8752 Uncertainty Campaign — Preserve It

- Worktree: `UpstreamDrift-worktrees/goal-8752-uncertainty`.
- The checkpointed 20-worker campaign was live on 2026-08-21 under parent PID 18404. It advanced during inspection and was processing the 72-branch ground
  atlas. Inspect the terminal/checkpoint before acting; do not infer completion
  from elapsed time.
- Do not edit, terminate, or redundantly rerun these frozen campaign sources:
  - `scripts/research/proximal_distal_energy/articulated_headline_uncertainty.py`
  - `scripts/research/proximal_distal_energy/articulated_shaft_atlas.py`
  - `scripts/research/proximal_distal_energy/articulated_ground_atlas.py`
  - `tests/research/test_articulated_headline_uncertainty.py`
- The campaign is computational evidence only. It does not repair #8909/#8910
  unless its independent-engine and manufactured-solution gates actually do so.

## Tools #4430 Rotating-Base Consumer

- Tools PR #4618 merged the canonical rotating-base provider, complete 18-run
  catalog, and matched PyQt6/React reviewer surfaces at `87ff0ea8c`.
- Tools PR #4619 merged its Rust 1.98 gate and handoff correction at
  `1664d806df8a2c7b184d2d3fbcea93b714caaee5`.
- Upstream branch `feat/4430-rotating-base-consumer` pins that exact merge in
  `vendor/ud-tools` and `TOOLS_GITLINK_SHA`; no solver or catalog is copied.
- Cross-repository tests require all 18 ordered cases, 13 valid cases, adverse
  indices 6/7/8/15/16, exact study and catalog digests, and every run's
  nonanatomical/no-human-validation/noncoaching boundaries.
- Upstream study authority remains
  `967c40f54cc03f8cae89cde09268d62771d220fe`; semantic study digest is
  `e6a55e6cf91e51f21fe3eb8bcb07b990a7798f18abcaf5ca73f5214cb6c5f9ec`; full
  run-catalog digest is
  `66493b833955c6492a00eae4a600df795df60a6f473f9a11c403084b58e51678`.
- Before opening the protected PR, run the focused vendored-provider tests,
  document/title/size/architecture gates, and inspect the exact diff. Repository
  policy prohibits auto-merge; human approval is required.

## Scientific Baseline and Boundaries

- Native MuJoCo and robotics Pinocchio independently qualify the rigid
  20-coordinate tree over 234 closed states. All evidence is synthetic and
  structural—not anatomy, physiology, equipment calibration, or coaching.
- Of 384 coupled-versus-rigid shaft cells, 126 match load/work; speed changes
  span `-0.0285` to `+0.0212 m/s`, including 82 negative outcomes.
- The preregistered ground comparison admits 0/384 cells; a post-hoc screen
  admits 60 with mixed signs. Unmatched positive differences are not evidence
  of a ground-pathway benefit.
- The release ledger has 295 atomic claims and 40 reviewed release claims; all
  40 retain at least one open scientific gate. #8724 owns normalized four-way
  adjudication and independent review.

## Publication and AffineDrift

- #8451/#8793 own the publication-quality contract. UpstreamDrift is the source
  authority; AffineDrift is a pinned independent projection; Tools only links.
- The current 232-page candidate renders cleanly and passes the computational
  profile, but it is not fast-web-access linearized; missing tags, 110 Type 3
  resources, and two unembedded resources also block archival qualification.
- After a protected Upstream merge, AffineDrift must pin the exact revision and
  verify the PDF, manifest, removed-artifact behavior, and anonymous public links.
- Archival DOI/PID deposit, equipment calibration, and governed human validation
  remain open; do not describe the program as scientifically complete.

## Vendored Tools Boundary

- Tools is a leaf dependency at `vendor/ud-tools`. Never edit inside the
  submodule or create a shadow under `src/shared/python/`.
- Production launchers resolve only the exact clean gitlink. Missing, dirty,
  escaped, mismatched, or mutable sibling checkouts fail closed.
- User-facing Tools additions require launcher and PyQt/React parity evidence;
  preserve adverse rows and provenance rather than filtering favorable runs.

## Immediate Next Steps

1. Finish local gates and protected delivery of the Tools #4430 consumer pin.
2. Verify the merge commit on Upstream remote `main`, then update Tools #4430
   with exact cross-repository evidence; close it only if every criterion passes.
3. Let the #8752 campaign reach a checkpoint, then adjudicate its failure map
   against #8909/#8910 before any governed regeneration or claim update.
4. Continue #8789 truth recovery and #8451 archival remediation without
   weakening the human/equipment/data boundaries.

## Gate Commands

```bash
python3 -m pytest tests/shared_contracts/test_tools_provider_contracts.py \
  tests/launchers/test_tools_vendor_authority.py \
  tests/launchers/test_launcher_model_sources.py \
  --tools-mode vendored -n 0 -q
python3 scripts/check_spec_paths.py
python3 scripts/check_root_clutter.py
python3 scripts/check_test_layout.py
python3 scripts/check_pytest_intree_testpaths.py
python3 scripts/ci/check_suite_marker_ratchet.py
python3 scripts/ci/check_dry_duplication_gate.py
python3 scripts/ci/check_architecture_budget.py
python3 scripts/check_module_size_budget.py --max-lines 1500 --include src
python3 scripts/check_doc_size_budget.py
python3 scripts/check_document_title_case.py --changed-from origin/main
python3 -m ruff check <changed-python-files>
python3 -m ruff format --check <changed-python-files>
```

Do not infer human technique, physiology, injury, timing demand, or coaching;
close #8556/#8557 without evidence; bypass protection; force-push/admin-merge;
add debt-ledger entries; or edit hash-pinned/Tools-owned evidence without its
governed regeneration path.
