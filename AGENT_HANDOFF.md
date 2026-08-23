# Agent Handoff — UpstreamDrift

Last updated: 2026-08-22

## Publication Quality Status (#8451 Closed; Archival Gates Open)

PR #8793 merged the PDF quality contract at `6e28baef54a0`: UpstreamDrift is authoritative, AffineDrift is its pinned publisher, and Tools/Sidekick only link. The web-linearized 231-page candidate passes full-page inspection; missing tags, 110 Type 3 resources, and two unembedded resources block archival release. Phase 0 landed in UpstreamDrift #8791, AffineDrift #3884, and Tools #4586. AffineDrift #3880/#3887/#3888 provide the immutable monograph projection, release pin, and verifier. #8789 owns Docker/quarantine/baseline; #8556, equipment calibration, and archival/PID release remain open. Details: `docs/research/proximal_distal_energy_transfer/PUBLICATION_QUALITY.md`.

## Protected Delivery Rules

- The only required check is the `ci-standard.yml` aggregate `quality-gate`;
  LoD/docs publish distinct gates. Never reuse the protected name.
- Same-repository `action_required` runs are approved automatically every five
  minutes; fork runs retain manual review. `repo-structure-gates` is sequential
  and fail-fast, and changed production functions are limited to 100 lines.
- `unit_gate_quarantine.json` (#8766) and `dry_duplication_quarantine.json`
  (#8763) are remove-only ledgers. Never add entries.
- Research evidence hash-pins sources. Regenerate it only through governed
  writers. Research/docs/test-only PR titles use `test:`, `docs:`, or `chore:`.

## Program Authority & Physics Epics State

- Epic #8557 governs the proximal-to-distal program; #8595 retains the agenda.
- Tools remote `main` is `428f832f9`; #4142 remains open because its full
  ensemble, quiet-zone, sensitivity, UI-parity, and UpstreamDrift-integration
  acceptance criteria are not yet complete. PR #4631 head `5c9ff05fa` has 18
  successful checks, one expected skip, and a fresh 20-view zero-drift hosted
  artifact. Its remaining
  ground/tee job passed 68/70 but exceeded two protected timing budgets while
  sharing this workstation with the 20-worker campaign; do not weaken the
  budgets or rerun under unchanged contention.
- Tools #4430 is **CLOSED**. PR #4618 added the qualified rotating-base reviewer
  surfaces and PR #4619 fixed the post-merge Rust gate; #4619 merge commit
  `1664d806df8a` remains an ancestor of Tools remote `main`. UpstreamDrift PR
  #8954 pinned that revision at merge commit `81cc731d0dd1`, which remains an
  ancestor of UpstreamDrift remote `main`.
- AffineDrift remote `main` is `1996deab6`; immutable projection PR #3888
  (`75ffcdce0860`) and title-governance PR #3798 (`e94237389270`) remain
  ancestors.
- #8557 is the single canonical epic with a dependency-ordered issue/evidence ledger. Legacy master epic #8426 was closed as superseded—not scientifically complete—on 2026-08-20; #8443 remains open. #8458, #8497,
  #8505, #8493, and #8499 are closed groundwork. README and conclusions now
  distinguish canonical/current governance from historical completed epics.
- #8789 is the Phase 0 truth-recovery gate for #8557. It keeps repository
  handoffs, GitHub issue states, release evidence, publication metadata, and CI
  from disagreeing.
- #8668 governs subject-scaled articulated contact; its qualification program
  remains open because #8751 and #8752 have unmet acceptance criteria.
- #8684 governs distributed grip, shaft, and ground:
  - #8751 is **CLOSED** by protected squash merge PR #8797 at remote-main
    commit `7c8f1547d39a5c182b52d44aeddd3330c3074b75`. Its distributed-grip
    atlas, event probes, and impulsive perfect-stick bound pass all registered
    gates. It is not a static-friction trajectory; #8796 owns that extension.
  - #8796 is **OPEN** under #8557/#8684 for the stronger attached-to-open and
    stateful finite-static-friction extension. It owns static-cone feasibility,
    subsequent stick--slip evolution, attached-state first failure, and an
    infeasible manufactured case. Its incorrect closure against #8797 was reversed on 2026-08-20 because that PR explicitly closed only #8751 and retained #8796 as separate scope. Do not fold those unqualified results into
    #8751 or describe the impulsive stick control as a trajectory law.
  - #8752 is **OPEN** on `research/8752-articulated-uncertainty`. The v2
    40-sample closed-state/LHS study is finite and energy-closed but every row
    retains partial opening. The 19-corner headline campaign is incomplete. At
    2026-08-22 21:26 PDT it retained 33 terminal pathway evaluations across 16
    fully accounted corners; `ground_free_moment_stiffness_scale-high` was
    active. Parent PID `18404` remained alive. Nominal remains shaft 126/384 and
    ground 0/384. Adverse pathways remain failures rather than being filtered.
    `articulated_headline_record_audit` validates the snapshot as partial and
    rejects torn JSON, registration/source drift, invalid pathway states, and
    premature completion. The completion-only evidence test stays untracked.
    `fix/8752-atomic-campaign-checkpoint` at `9f850a67f` adds atomic top-level
    JSON replacement and an interruption test. Branch
    `research/8752-uncertainty-reviewer-table` in worktree
    `UpstreamDrift-worktrees/8752-reviewer-table-v2` adds a deterministic,
    fail-closed CSV projection with one row per corner--pathway pair, retained
    failures and not-affected rows, canonical-record/source-set hashes, and an
    explicit summary-not-trajectory scope label. Twenty focused tests, Ruff,
    and scoped mypy pass; partial input writes nothing. Integrate both follow-ups
    only after campaign completion, then generate the CSV from governed JSON.
  - #8800 is **OPEN** and blocks #8752. Its governed generator regenerates all
    13 phase states for cases 0/8/9/17; nominal is 52/52 feasible and reproduces
    committed states within 1e-8 rad. JSON/NPZ evidence rejects source/content
    drift. `ArticulatedAtlasAuthority` binds exact scaled models, builds per case,
    gates per phase, and retains failures. Seven corners completed: nominal,
    height-high, both mass, and both joint-limit are 52/52; height-low retains
    case 0/phase 12 `ik_nonconvergence`. Joint-limit-low retains 0.0885 rad
    minimum margin; joint-limit-high retains 0.1185 rad. Their maximum closure
    errors are 1.03e-10 m and minimum collision clearances are 0.0491 m. The
    evidence passes 3/3. Write/validate preflight excludes workers, requires common
    support, binds 83/1 states, full hashes, invalidators, scaling-mismatch and
    infeasible-limit controls, a preoutcome 0.001 m/s numerical resolution floor,
    engine/step discrepancy thresholds, and span-normalized one-sided secants that
    cannot be ranked as cross-parameter importance.
    Its v1 evidence contract preserves checkpoint identity and per-cell arrays. A shared runtime authority boundary now resolves planned/feasible states without deleting the retained low-height failure, rebuilds and revalidates exact scaled case models, emits detached digest-bound provenance, and rejects untyped authorities and invalid designs; the frozen runners are not yet wired to consume it. A separate v1 scientific-model digest canonicalizes finite parameters to 15 significant digits, eliminating observed native-runtime last-bit hash drift while detecting 1e-10 kg perturbations. The regenerated plan publishes independently reproducible shaft/ground source-set and scientific-configuration checkpoint digests, excludes worker count, and is byte-reproducible under pytest and standalone runtimes. A fail-closed resolver converts these into immutable corner/pathway checkpoint prefixes and rejects authority, source, configuration, or pathway drift. It also constructs and validates an exact JSON-safe v1 persisted-checkpoint envelope binding every immutable prefix field plus the registered state-slot mapping and pathway-specific branch design; missing, extra, malformed, out-of-design, or tampered fields fail closed. Atomic no-pickle persistence now enforces exact numeric/Boolean field, shape, and dtype contracts, rejects infinity and corrupt archives, retains pathway-defined NaN for downstream semantic gates, and preserves prior checkpoints on interrupted writes. Its directory auditor validates every branch contract before reuse, rejects unregistered/torn files, classifies exact partial versus complete coverage, and hashes the same immutable bytes it validated; checkpoints remain explicitly non-release evidence. A restart inventory returns detached validated branch payloads plus the exact remaining registered descriptor sequence, including a distinct empty fresh-start state. Independent shaft/ground contract generation reproduces every actual runner-local field, shape, and dtype, binds the scientific configuration digest, and keeps worker count operational. Both pathway identities bind all resolver, persistence, and contract-generator sources. All 88 focused authority/runtime/contract/checkpoint/plan/identity tests pass. Shaft matching is coupled/rigid station-force/dissipated-work; ground matching is coupled/fixed grip-force/total-work. Planned, feasible, and executed denominators remain separate; partial records cannot qualify.
    The figure contract exposes support transitions, resolution, secants, and failures with searchable vector text and color-independent status encoding. Its v1 deterministic data layer consumes all 14 digest-bound packs without favorable filtering, preserves nominal ground 0/384, and rejects redigested semantic tampering through ordered support/outcome/axis/failure reconciliation, result binding, exact JSON bytes, finite ordered secants, and reproduced nonmonotonic classifications. The five-panel SVG/PDF renderer uses searchable text, pattern/shape status encoding, explicit units, the 0/384 boundary, no-human/coaching language, and embedded result/data digests; PDF raster/font QA is clean. The end-to-end publisher revalidates the governed plan, complete result, and every cell pack before emitting assets; its module CLI is the reproducible publication command and 85 focused tests pass.
    Common-support analysis reproduces nominal 126/384 shaft and 0/384 ground while preserving identities, denominators, failures, resolution, and unpooled secants. Nominal-only/corner-only identities remain distinct. Cell-evidence v2 stores finite paired change/resolution only on persistent support and NaN/false elsewhere; matching/status and resolved/threshold relationships reproduce. Packs own detached input copies, derive identity/outcome/gates from one atlas, and use atomic no-pickle NPZ. Fail-closed gates combine both compared branches, broadcast parity across engines, and retain simultaneous failures. Corner assembly rejects partial execution, gate failures, missing authority, and executed/failed-state overlap. Axis assembly requires registered low/high pairs, keeps one-sided support separate, and emits null for empty shared support. The v2 result requires 14 corner/pathway records, safe cell-pack paths/digests, and six axis/pathway summaries. Bundle/plan validation reconcile NPZ contents, authorities, denominators, failures, and scales. All reject partial/tampered evidence and remain model-dependent, not causal, population, mechanism, ranking, or coaching claims.
    The governed feasibility/margin figure has embedded CID TrueType text and
    passes its retained-failure contract. Q2/H5 and the machine-readable
    prediction registry no longer repeat the obsolete failed-closure narrative.
    Release-manifest/checksum regeneration and full-paper rendering must wait
    until the live headline record is complete, or they could admit partial data.
- The refreshed 1,092-candidate census has no unadjudicated passages and 304 atomic claims. Deterministic claim and evidence validators pass over 2,172 references, 321 hash-pinned local artifacts, and 78 external URLs. PD-CLAIM-297--305 govern the local uncertainty screen, adverse partial opening, conditional PRCCs, structural authority, retained low-height failure, open propagation, and in-progress headline design without promoting any partial result; five focused contracts pass. The typed model-completion registry now covers every narrative hypothesis H1--H11 exactly once, retaining H9 as contradicted and H5/H10 as inconclusive; three alignment/loader tests pass. The 40 reviewed release claims retain scientifically open gates; #8724 still owns normalized four-way adjudication and independent review.
- PR #8793 canonicalizes UTF-8 claim evidence before hashing; #8789 retains
  Docker/quarantine/baseline and clean-checkout truth-recovery scope.
- #8556 remains open: no governed participant dataset contains synchronized
  bilateral six-axis grip wrenches. Synthetic traces cannot replace it. The 2026-08-20 NotebookLM network test and same-profile refresh both failed authentication; Chrome-cookie recovery could not decrypt the local cookies, so Biomechanics/Nonlinear Control collection mining still requires an interactive Google login and cannot yet support claims.

## Qualified Baseline — And Its Limits

Native MuJoCo and robotics Pinocchio independently qualify the 20-coordinate rigid tree over 234 closed states, with power, passivity, energy, refinement, geometry, and engine-parity controls. All of it is a synthetic structural reference — not equipment calibration, anatomy, physiology, or coaching guidance.

- Of 384 coupled-versus-rigid shaft cells, 126 match on load and work; speed
  differences span `-0.0285` to `+0.0212 m/s` (82 negative), **rejecting a
  universal passive-shaft speed benefit**.
- The preregistered ground screen admits **0/384** coupled--fixed cells (ground
  damping asymmetry). A post-hoc screen admits 60 cells with mixed signs.
  **Do not read unmatched positive differences as a ground-pathway benefit.**
- Initialization: natural-zero, gravity-only, and conditional starts gave peak
  ground forces of 32.8, 565.5, and 510.3 N.

## Immediate Next Steps

1. Complete #8752 and its blocking structural propagation issue #8800.
2. Continue #8789's remaining Docker/quarantine/baseline work without overlap.
3. Burn down #8766 without widening its registered contract.

## Gate Commands

```bash
python scripts/check_spec_paths.py && python scripts/check_root_clutter.py
python scripts/check_test_layout.py && python scripts/check_pytest_intree_testpaths.py
python scripts/ci/check_suite_marker_ratchet.py && python scripts/ci/check_dry_duplication_gate.py
python scripts/ci/check_architecture_budget.py
python scripts/check_module_size_budget.py --max-lines 1500 --include src
python scripts/check_doc_size_budget.py
python -m ruff check . && python -m ruff format --check .
```

```bash
python -m scripts.research.proximal_distal_energy.claim_audit validate
python -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate
python -m scripts.research.proximal_distal_energy.momentum_question_readiness validate
python -m scripts.research.proximal_distal_energy.qualify_open_release validate \
  --source-revision "$(git rev-parse HEAD)" \
  --publication-profile computational
pytest tests/research -q
```

Do not infer human technique/physiology/injury/coaching; close #8556/#8557
without evidence; bypass protection; force-push/admin-merge; add ledger entries;
edit hash-pinned or Tools-owned files without regeneration; or rerun unchanged
capacity failures.
