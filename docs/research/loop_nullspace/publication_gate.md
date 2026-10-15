# NSX Publication Gate Checklist

This checklist defines what must be true before any NSX result is published
externally. It implements NSX-74 (#12592) under epic #12593 and applies the
binding rules R1 to R14 of that epic. Every item names the file, test or
command that checks it. Where no automated check exists yet, the item says so
and names the child issue that will add one.

A passing check is not publication approval. A successful render, a green test
run or a passing governance gate does not establish scientific, semantic,
visual or publication approval (see `CLAUDE.md`, Engineering Design Manual
Authority).

## Scope and Current Status

- [ ] Applies to every externally published NSX result: reports, figures, plots,
  papers, web or PyQt output and exported data.
- [ ] Current status: **publication blocked**. External validation is deferred,
  design-manual release is blocked, and several NSX children that the
  checklist depends on are open (see Open Blockers below).

## Verified Citations

- [ ] Every hypothesis used by a published quantity has a declared
  `verification_status` in the registry.
  Check: `tests/config/loop_nullspace_registry/test_registry_seed.py`
  (`test_every_hypothesis_has_a_declared_status`).
- [ ] Every hypothesis with a `citation_id` resolves to a citation whose status
  is VERIFIED, and no user-facing statement relies on an UNVERIFIED or REMOVED
  source.
  Check: `tests/config/loop_nullspace_registry/test_registry_seed.py`
  (`test_statements_never_rest_on_unverified_sources`).
- [ ] Every citation in the registry is VERIFIED and has a readable DOI or URL.
  Registry citations: `CIT-STANEV-2019`, `CIT-CHOI-PARK-2020` and
  `CIT-NESBIT-SERRANO-2005`. Check: `src/config/loop_nullspace_registry.json`
  (`citations`).
- [ ] Each hypothesis with no empirical claim is `NOT_APPLICABLE` and cites no
  source, as the registry `schema_note` requires. Check:
  `src/config/loop_nullspace_registry.json` (`schema_note`, `hypotheses`).
  Status: the six seed hypotheses are all `NOT_APPLICABLE`. The literal reading
  of "VERIFIED with a readable source" would fail for all six, so the schema
  rule above is the operative one until the owner decides otherwise.
- [ ] Every torque capacity used in a published figure has a verified source.
  Entries without one carry `assumed = true` and `UNVERIFIED`. Check:
  `src/config/loop_nullspace_registry.json` (`torque_capacity`) and
  `docs/research/loop_nullspace/contracts.md` (section `NSX-04 Torque Capacity`).
  Status: open. `torque_capacity` is empty until NSX-04 (#12553), and the
  presets `envelope_only`, `capacity_min_effort` and `smooth_capacity` all use
  `H-TORQUE-CAPACITY`.
- [ ] Every preset shown in a figure resolves to existing hypothesis and
  convention IDs. Check: `tests/config/loop_nullspace_registry/test_registry_seed.py`
  (`test_presets_reference_existing_ids`).
- [ ] Each citation claim was checked against the source text, and the
  verification table is current. Check: the Citation Verification Table in
  `docs/research/loop_nullspace/loop_nullspace.tex` (section
  `sec:citations`, retrieval date 2026-10-14).
- [ ] No UNVERIFIED or REMOVED reference from that table appears in any
  published text. The table lists the Thornton 2025, Mehta 2024 and Sorokin 2025
  percentages and the Nesbit "two-thirds" statement as UNVERIFIED, and the SPAR
  and Bosch dead-zone claims as REMOVED. Check (manual, until the NSX-11 gate
  exists): `git grep -n -i "thornton\|mehta\|sorokin\|two-thirds\|SPAR\|dead zone"`
  on the figure source and report text.
- [ ] The runtime display gate exists and is enforced. Check (not yet
  implemented): `require_verified_for_display` and `load_registry`, specified
  in `docs/research/loop_nullspace/contracts.md` (section `NSX-11 Registry
  Loader`). Status: open, NSX-11 (#12556). No `load_registry` exists under
  `src/`.
- [ ] Hypothesis IDs in the external validation record match the registry. Check:
  `git grep -n "H-TORQUE-CAPACITY-BOX\|H-GRIP-FRICTION-CONE" -- docs`.
  Status: discrepancy. `docs/development/planning/DV-12554.md` uses
  `H-TORQUE-CAPACITY-BOX`, `H-GRIP-FRICTION-CONE`, `H-EFFORT-CAPACITY-QUADRATIC`
  and `H-TORQUE-RATE-SMOOTHNESS`. The registry uses `H-TORQUE-CAPACITY`,
  `H-FRICTION-CONE`, `H-EFFORT-CAPACITY` and `H-SMOOTH-TORQUE-RATE`. Reconcile
  before any validation claim cites either set.

## External Validation Status

- [ ] The NSX-05 deferred-validation record exists and validates. Check:
  `docs/development/planning/DV-12554.md`, its catalog entry `DV-12554` in
  `docs/development/planning/catalog.json`, and the command
  `python3 shared_scripts/deferred_validation.py`, which reports
  "Deferred-validation catalog OK". The hook test is
  `tests/unit/repo_hygiene/test_deferred_catalog_hook.py`.
- [ ] Status is stated as deferred or unvalidated, never as validated. Check:
  the `status` field of `DV-12554` in `docs/development/planning/catalog.json`
  (currently `deferred`), and `ValidationStatus.UNVALIDATED` as the default in
  `src/shared/python/loop_nullspace/evidence.py`.
- [ ] The board decision is recorded before any reactivation. Check:
  the `board` block of `DV-12554` in `docs/development/planning/catalog.json`
  (currently `pending`). The record also states "Reactivate only after a recorded
  Board decision", in `docs/development/planning/DV-12554.md` (section `Board
  Review and Reactivation`).
- [ ] Pre-registered predictions and acceptance criteria were committed before
  any held-out data were unblinded. Check (manual): compare the commit time of
  `docs/development/planning/DV-12554.md` with the data-receipt times. Current
  record: `git log -1 --format=%cI -- docs/development/planning/DV-12554.md`
  gives 2026-10-14. No held-out data exist.
- [ ] Mock-only, synthetic or simulated data are excluded from validation
  evidence. Check: "Synthetic Data Exclusion" in `DV-12554.md`, and the test
  `test_synthetic_fixtures_never_claim_conformance` in
  `tests/integration/cross_engine/test_loop_nullspace_conformance.py`, which
  asserts that synthetic adapters decline the closed-loop snapshot capability.
- [ ] Surface EMG is described only as an activation proxy and timing
  observable, never as a hand-wrench ground truth. Check: section `Surface EMG
  Onset Timing and Activation Proxy Observation Model` in `DV-12554.md`.
- [ ] Instrumented-grip and EMG data, when they exist, follow the provenance and
  anonymisation rules in `DV-12554.md` (section `Data Provenance and
  Anonymization Requirements`). Check: see Reproducibility below.
- [ ] Status: the validation has not started. No instrumented-grip, EMG or
  held-out swing dataset exists in the repository (`DV-12554.md`, section
  `Existing Capabilities and Evidence`).

## Cross-Engine Conformance

- [ ] The NSX-56 harness runs for every engine a figure names. Command:
  `python3 -m pytest tests/integration/cross_engine/test_loop_nullspace_conformance.py -n 0 -q --timeout=300`.
  Harness: `tests/integration/cross_engine/test_loop_nullspace_conformance.py`
  and `tests/integration/cross_engine/loop_nullspace_conformance_support.py`.
  Status on this branch: the NSX unit and conformance selection gives
  231 passed and 19 skipped. The skips are MyoSuite only, because the `myosuite`
  package is not installed on this host. MyoSuite conformance is therefore not
  verified here.
- [ ] Track N tolerances are registered per engine pair and per metric, each with
  a rationale. Check: `tests/integration/cross_engine/divergence_registry.yaml`
  (`check: closed_loop_snapshot_conformance`) and the test
  `test_every_pair_metric_is_registered_with_a_rationale`.
- [ ] No tolerance was widened to pass. Any widening carries a
  `Tolerance-Change-Evidence:` commit trailer (`AGENTS.md`, Git Safety and
  Fleet-Guard Hooks). Check (manual): `git log --grep="Tolerance-Change-Evidence"`
  for any change to `divergence_registry.yaml`.
- [ ] Track S, the derived-quantity comparison, passes its propagated bound. Check:
  `test_track_s_constrained_acceleration_and_grip_wrench`,
  `test_track_s_energy_power_and_closure_spectrum` and
  `test_track_s_grip_wrench_envelope` in the conformance test file.
- [ ] The perturbed-engine control fails Track N and stays inside the Track S
  bound. Check: `test_a_perturbed_engine_fails_track_n_and_stays_inside_the_track_s_bound`.
- [ ] Engines that were skipped, unavailable or fail-closed are named in the
  figure. Simscape has no native snapshot: `test_simscape_native_adapter_fails_closed`
  asserts this. NSX-55 (#12583) is open, and no deferred record for it appears
  in `docs/development/planning/catalog.json`.
- [ ] Agreement between engines is stated as numerical conformance only, not
  physical validation (`docs/research/loop_nullspace/contracts.md`, section
  `NSX-56 Conformance`).
- [ ] Results from differing anatomies are not described as equivalent envelopes
  (epic rule R12, Scientific comparison track). Check (manual).

## Non-Claims and Evidence Axes

- [ ] Each published figure states its non-claims. The minimum set is the
  Non-Claims section of `docs/research/loop_nullspace/README.md` and section
  `sec:nonclaims` in `docs/research/loop_nullspace/loop_nullspace.tex`. Automated
  enforcement is open under NSX-41 (#12573).
- [ ] Each quantity states its `allocation_dependence`: `INVARIANT`,
  `FAMILY_DEPENDENT` or `UNAVAILABLE`. Check: the `AllocationDependence` enum
  and `QuantityRecord` in `src/shared/python/loop_nullspace/evidence.py`, and
  `tests/unit/loop_nullspace/test_contracts.py`.
- [ ] `INVARIANT` is shown as "identical for every admissible allocation", never
  as "measured". Check: the module docstring in
  `src/shared/python/loop_nullspace/evidence.py`.
- [ ] `FAMILY_DEPENDENT` quantities list at least one hypothesis, and `INVARIANT`
  quantities list none. Check: `QuantityRecord.__post_init__` in
  `src/shared/python/loop_nullspace/evidence.py`, covered by
  `tests/unit/loop_nullspace/test_contracts.py`.
- [ ] `UNAVAILABLE` quantities carry a reason and all-NaN values. Check: the same
  `__post_init__` and tests.
- [ ] Each figure lists the binding hypotheses and conventions, plus its
  `validation_status`, which must read `UNVALIDATED` until external validation
  passes (epic rule R1). Check (manual, until NSX-42 #12574 and NSX-43 #12575
  add the plot model and tabs).
- [ ] Replay results separate open-loop replay from feedback-assisted tracking,
  and the feedback torque budget is shown on its own line (R10). Check (manual,
  until NSX-31 #12571).
- [ ] Constrained counterfactual totals are shown as lambda-invariant, and
  per-actuator attributions as `FAMILY_DEPENDENT` (R9). Check (manual, until
  NSX-30 #12570).
- [ ] No figure claims a single "true" hand split. The product maps the feasible
  family (epic section 1). Check (manual).
- [ ] No figure infers stiffness, co-contraction or muscle force from net joint
  torques, and joint power is not presented as muscle power (README Non-Claims,
  epic rule R6). Check (manual).
- [ ] No golfer-specific or physiological result is published. The fixtures are
  generic geometry, not golfer models (`loop_nullspace.tex`, section
  `sec:nonclaims`).

## Design-Manual Governance

- [ ] The governance gate passes and states that release is blocked. Command:
  `python3 -m scripts.check_design_manual_governance`. Current output:
  "66 QMD sources, 0 registered calculations, release=blocked-inventory-required."
  Config: `scripts/config/design_manual_governance.json`.
- [ ] A passing gate is not approval. The config keeps publication at
  `deny-until-approved` and `public_projection_allowed: false`. Check:
  `scripts/config/design_manual_governance.json` (`publication`).
- [ ] Engineering manual content is edited only as QMD in `manuals/upstreamdrift`.
  The NSX chapter is `manuals/upstreamdrift/chapters/65-loop-null-space.qmd`
  (provisional). Generated LaTeX, PDF, DOCX and HTML are not edited (`CLAUDE.md`,
  Engineering Design Manual Authority).
- [ ] The NSX blocker is open. Check: `UP-NSX-loop-nullspace-exploration` in
  `manuals/upstreamdrift/calculation-registry.json`. Its resolution says
  calculations stay empty and publication blocked until implementation, UP-D1
  inventory, independent validation and review.
- [ ] Any change to a governed NSX calculation pathway updates the calculation
  registry, the SPEC and the handoff in the same change (`CLAUDE.md`). Check
  (manual): `scripts/config/design_manual_governance.json` (`agent_context`).
- [ ] Each required evidence item is present: `immutable_source_commit`,
  `source_tree_sha256`, `calculation_registry_sha256`, `toolchain_lock_sha256`,
  `artifact_sha256`, `semantic_parity`, `pdf_page_review`, `docx_page_review`
  and `human_approval`. Check: `scripts/config/design_manual_governance.json`
  (`publication.required_evidence`). Status: none is recorded for NSX.
- [ ] The standalone LaTeX reference is labelled as a research record and is
  not design-manual approval (`AGENTS.md`, Modeling Reference Documentation).
  Check: `docs/research/loop_nullspace/README.md`.
- [ ] Document titles, headings and captions use title case. Check:
  `python3 scripts/check_document_title_case.py --changed-from origin/main`
  (`AGENTS.md`, Document Title Capitalization).

## Reproducibility

- [ ] The goldens regenerate exactly. Command:
  `python3 docs/research/loop_nullspace/derivations/generate_goldens.py --check`.
  Test: `tests/unit/loop_nullspace/test_golden_derivations.py`
  (`test_checked_in_golden_matches_regeneration`). Status on this branch: passes.
- [ ] Golden values were not changed to make a result pass. Check: the golden
  rule in `docs/research/loop_nullspace/contracts.md` (section `Shared
  Conventions`) and `git log -- tests/fixtures/loop_nullspace/goldens/`. Golden
  files: `tests/fixtures/loop_nullspace/goldens/` (`planar_two_arm_full.json`,
  `envelope_analytic_lp.json`, `closure_convention.json`,
  `grasp_map_worked_example.json`, `spatial_counts.json` and the other three
  golden files in the same directory).
- [ ] The NSX unit suite passes. Command:
  `python3 -m pytest tests/unit/loop_nullspace tests/config/loop_nullspace_registry -n 0 -q --timeout=300`.
- [ ] Every figure records its model provenance (spec SHA, inertial source,
  bare-club bench ID) and input provenance (measured or reconstructed inputs,
  derivative method). Check: `ModelProvenance` and `InputProvenance` in
  `src/shared/python/loop_nullspace/evidence.py`.
- [ ] Every published figure records the commit SHA and the exact command used
  to produce it. Check (manual). No automated check exists yet; the NSX report
  schema (#12572) is the intended place for it.
- [ ] Every stochastic output records its seed. Check (manual). The seeded
  ensembles for NSX-61 and NSX-62 are not yet implemented
  (`docs/research/loop_nullspace/contracts.md`, section `NSX-61 and NSX-62
  Ensembles`).
- [ ] Private captures are anonymised. Participants are replaced by salted
  SHA-256 subject tokens at ingest (`DV-12554.md`, section `Data Provenance and
  Anonymization Requirements`). Native capture files are never tracked. Check:
  `tests/unit/repo_hygiene/test_no_private_capture_data.py`, with the capture
  location defined in `src/motion_capture/capture_registry.py`.
- [ ] Private execution receipts keep the input hashes and detailed provenance,
  and repository artifacts stay anonymous (`AGENTS.md`, Modeling Reference
  Documentation). Check (manual).
- [ ] The LaTeX reference compiles. Command:
  `tectonic -X compile --outdir build docs/research/loop_nullspace/loop_nullspace.tex`
  (`docs/research/loop_nullspace/README.md`, section `Build Locally`). A compile
  is not scientific approval. This checklist revision did not run it.
- [ ] Changes to governed artefacts pass the repository gates before merge:
  `python3 scripts/ci/check_file_size_budget.py`, ruff check and ruff format.
  These are the CI requirements in `CLAUDE.md`.

## Open Blockers

These children must land, or be recorded as deferred, before the matching item
can pass.

| Blocker | Issue | Items affected |
| --- | --- | --- |
| Torque capacity table | NSX-04 (#12553) | Verified citations (capacity) |
| Typed registry loader and display gate | NSX-11 (#12556) | Verified citations (display gate) |
| Output policy and report surface | NSX-41 (#12573), NSX-40 (#12572) | Non-claims, commit provenance |
| Plot model and UI tabs | NSX-42 (#12574), NSX-43 (#12575) | Non-claims, validation status display |
| Replay and counterfactual labelling | NSX-30 (#12570), NSX-31 (#12571) | Non-claims |
| MyoSuite conformance run | NSX-52 (#12580) | Cross-engine conformance |
| Simscape snapshot or deferred record | NSX-55 (#12583) | Cross-engine conformance |
| Seeded ensembles | NSX-61 (#12586), NSX-62 (#12587) | Reproducibility |
| Hypothesis ID reconciliation | DV-12554 and registry | Verified citations |
| External validation | NSX-05 (#12554), board decision, equipment | External validation |
| Design-manual inventory and approval | UP-NSX-loop-nullspace-exploration, UP-D5, UP-D7, UP-D8 | Design-manual governance |
