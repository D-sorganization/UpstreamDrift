# Agent Handoff — UpstreamDrift

Last updated: 2026-08-20

## Publication Quality Status (#8451 Closed; Archival Gates Open)

PR #8793 merged the PDF quality contract at `6e28baef54a0`: UpstreamDrift is authoritative, AffineDrift is its pinned publisher, and Tools/Sidekick only link. The web-linearized 231-page candidate passes full-page inspection; missing tags, 110 Type 3 resources, and two unembedded resources block archival release. Phase 0 landed in UpstreamDrift #8791, AffineDrift #3884, and Tools #4586. AffineDrift #3880/#3887/#3888 provide the immutable monograph projection, release pin, and verifier. #8789 owns Docker/quarantine/baseline; #8556, equipment calibration, and archival/PID release remain open. Details: `docs/research/proximal_distal_energy_transfer/PUBLICATION_QUALITY.md`.

## Read This First — How Merging Works Now

Three workflows used to publish `quality-gate`; the first could satisfy
protection while the real aggregate failed. #8747 fixed this; #8754 is closed.

- The **only** required check is `quality-gate`, published by the aggregate in
  `ci-standard.yml`; LoD/docs publish `lod-quality-gate`/`docs-quality-gate`.
  **Never rename another job to `quality-gate`** — that recreates the collision.
- If your run sits in `action_required`, wait: `approve-same-repo-runs.yml`
  approves same-repository runs every five minutes (fork PRs keep the manual
  gate). Do not disable it — branch updates come from a bot GitHub treats as a
  first-time contributor forever.
- `repo-structure-gates` is **sequential/fail-fast**. Run the battery below;
  `check_architecture_budget.py` limits changed production functions to 100 lines.

### Two Debt Ledgers — Remove-Only Ratchets

`unit_gate_quarantine.json` (520 node IDs, #8766) applies through
`_apply_unit_gate_quarantine` only when CI sets `UNIT_GATE_QUARANTINE=1`.
`dry_duplication_quarantine.json` (#8763) tracks duplicate fingerprints.
**Entries may only be removed, never added**.
PR #8768 cleared main's ruff, bandit, XML security, and frontend lock debt.

### Traps That Cost Real Time

- **Committed evidence hash-pins its sources.** Fourteen `tests/research/` files
  are frozen by `source_sha256` maps; edit only with `qualify_open_release write`.
- **Phantom Guard Scope Rules:** Conventional commit prefix for PR titles
  touching only `scripts/research/`, `docs/research/`, and `tests/research/`
  must use `test:`, `docs:`, or `chore:` rather than `feat:` (which requires
  diffs touching `src/`, `rust_core/`, or `api/`).

## Program Authority & Physics Epics State

- Epic #8557 governs the proximal-to-distal program; #8595 retains the agenda.
- Tools #4142 remains open; immutable ensemble/sensitivity consumption and cross-repository parity remain required before #8557 can close.
- Tools #4430 remains open after #4450 (`8f654b3a1552`); the rotating-base provider, Python/TypeScript parity, rendered QA, and UpstreamDrift pin remain due.
- Live issue audit confirms #8557/#8426/#8443 remain open; #8458, #8497,
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
    infeasible manufactured case. Do not fold those unqualified results into
    #8751 or describe the impulsive stick control as a trajectory law.
  - #8752 is **OPEN** on `research/8752-articulated-uncertainty`. The v2
    40-sample closed-state/LHS study is finite and energy-closed but every row
    retains partial opening. The 19-corner headline campaign is incomplete:
    nominal completed at shaft 126/384 and ground 0/384; every gate and all four
    source hashes match remote basis `fbff8dc53`. Exec session `64656` completed
    grip-stiffness-low at shaft 182/384 (+56 nominal) and ground 0/384
    with every gate passing. Its 72/72 checkpoints have exact unique identities,
    one v1 schema/design digest `32ccf54bee70`, finite payloads, and registered
    shapes. The campaign has 3/19 corners complete; stiffness-high finished at shaft 80/384 and ground 0/384 with every gate passing and all source hashes matching. Its 72/72 checkpoints have twelve complete state slots, exact identities/schema/design/shape/dtype parity, finite no-pickle payloads, and set digest `ffc89ddf3995`. The independent `articulated_headline_record_audit` reads one immutable snapshot and rejects torn JSON, schema/design/config/source drift, nonprefix corners, invalid pathway states, and premature completion. It validates the live prefix as 4/19 rows, 3 fully accounted, 7 terminal pathways, and explicitly not release evidence. Grip-damping-low shaft has completed; ground is valid at 53/72 checkpoints (eight complete of nine observed slots), design digest `6f7ee9f638d9`, set digest `521b11303ee0`; all 18 record-audit tests pass.
    All 72 nominal branch checkpoints retain exact trajectory/force/ground-force
    parity and restart equivalence.
    The completion-only headline evidence test stays untracked until data finish.
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
    Its v1 evidence contract preserves checkpoint identity and per-cell arrays. A shared runtime authority boundary now resolves planned/feasible states without deleting the retained low-height failure, rebuilds and revalidates exact scaled case models, emits detached digest-bound provenance, and rejects untyped authorities and invalid designs; the frozen runners are not yet wired to consume it. A separate v1 scientific-model digest canonicalizes finite parameters to 15 significant digits, eliminating observed native-runtime last-bit hash drift while detecting 1e-10 kg perturbations. The regenerated plan publishes independently reproducible shaft/ground source-set and scientific-configuration checkpoint digests, excludes worker count, and is byte-reproducible under pytest and standalone runtimes. A fail-closed resolver converts these into immutable corner/pathway checkpoint prefixes and rejects authority, source, configuration, or pathway drift. It also constructs and validates an exact JSON-safe v1 persisted-checkpoint envelope binding every immutable prefix field plus the registered state-slot mapping and pathway-specific branch design; missing, extra, malformed, out-of-design, or tampered fields fail closed. Atomic no-pickle persistence now enforces exact numeric/Boolean field, shape, and dtype contracts, rejects infinity and corrupt archives, retains pathway-defined NaN for downstream semantic gates, and preserves prior checkpoints on interrupted writes. Both pathway identities now bind the resolver and persistence sources. All 79 focused authority/runtime/checkpoint/plan/identity tests pass. Shaft matching is coupled/rigid station-force/dissipated-work; ground matching is coupled/fixed grip-force/total-work. Planned, feasible, and executed denominators remain separate; partial records cannot qualify.
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
  bilateral six-axis grip wrenches. Synthetic traces cannot replace it.

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
