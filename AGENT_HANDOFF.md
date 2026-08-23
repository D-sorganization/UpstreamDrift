# Agent Handoff — UpstreamDrift

Last updated: 2026-08-22

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
    retains partial opening. The 19-corner headline campaign is incomplete. Its
    20-worker process ended after 2026-08-22 22:53 PDT without writing a terminal
    record; do not describe it as complete. The JSON remains `in_progress` with
    17 corner records and 16 fully accounted corners. The active
    `ground_free_moment_stiffness_scale-high` corner retained 49/72 branches;
    1,160 atomic checkpoint files (12,343,744 bytes) exist overall, with no torn
    `.tmp` file. PID `18404` and its workstation workers are absent. A qualified
    eight-worker continuation started on ControlTower at 2026-08-23 09:28 PDT.
    Nominal remains shaft 126/384 and ground 0/384. Three
    adverse pathways are retained failures rather than filtered results. The independent
    `articulated_headline_record_audit` rejects torn JSON,
    schema/design/config/source drift, nonprefix corners, invalid pathway
    states, and premature completion. It validates the current snapshot as
    `partial` with `release_evidence: false`, source-set digest
    `a64f02a92a300afe9e1f8070dd4ddca9e48add2b2fee22a337a4df5c37b25e57`,
    and record digest
    `069d942a2a04ba1a7960e628d44d49e88b12ca1780722fb977ace03d538181ba`.
    All embedded campaign source hashes still match the unchanged worktree at
    commit `602b7bdf7b31127af24074e1725731845ed62d18`. Worker count is an
    execution control: both the top-level resume contract and branch execution
    digest intentionally exclude it, and a regression test permits worker-count
    changes while rejecting scientific-design drift. Therefore a lower-worker
    restart can reuse the checkpoints without changing the registered study.
    All 72 nominal branch checkpoints retain exact trajectory/force/ground-force
    parity and restart equivalence.
    ControlTower has a Ryzen 9 5950X (16 cores/32 threads), approximately 128 GB
    RAM, and ample free memory. Its dirty/stale primary checkout was not mutated.
    The continuation uses detached exact HEAD
    `13146cdcece879e7156e06e2dca6626c1a54e045` at
    `C:\Users\diete\Repositories\UpstreamDrift-worktrees\goal-8752-uncertainty`.
    Transfer manifest SHA-256
    `af88d9a25fb7a13033d729b73fe0da8cbf4eedced376e8acfc2feee939fcf2d5`
    verified all 1,162 transferred files; archive SHA-256 is
    `685393a555cc2dde6033e7d7a16a9d96153693fc055d2097e853eabcfdc55ab2`.
    A one-branch replay of completed corner
    `ground_free_moment_stiffness_scale-low` deliberately omitted and
    regenerated `state-00-control-00.npz`. It passed every registered model gate,
    exact metadata/dtype/shape and discrete-field comparisons, and the documented
    cross-CPU numerical criterion (`rtol=2e-8`, `atol=1e-9`). The exact original
    Ubuntu 22.04/Python 3.10.12 GCC, NumPy 2.2.4, SciPy 1.15.2, MuJoCo 3.8.0,
    and Pinocchio 3.8.0 runtime is containerized as
    `upstreamdrift-8752:ubuntu22`; the image manifest is
    `sha256:4be08c2d04e94b827137429167a2d78c027d93a2718b699dea90b34b29b003f6`.
    CPU-dependent differences were bounded by `1.38e-8` absolute and preserved
    all decision classifications. Production container
    `upstreamdrift-8752-campaign` (`7a348593a10e...`) is capped at eight CPUs and
    96 GB RAM with `on-failure:3`; runtime status and logs are in
    `C:\Users\diete\Campaigns\UpstreamDrift-8752\status.json` and `logs\campaign.log`.
    The RTX 3090 does not accelerate this CPU/NumPy/MuJoCo/Pinocchio workload.
    Do not start a duplicate container or mutate the dirty primary checkout.
    The completion-only headline evidence test stays untracked until data
    finish. Branch `fix/8752-atomic-campaign-checkpoint`, forked from exact
    campaign commit `8c537b660`, adds atomic replacement for the top-level JSON
    and a manufactured replacement-interruption test. Integrate that commit only
    after the live campaign finishes; changing campaign sources sooner would
    intentionally trigger the source-drift gate.
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

1. Monitor the single qualified ControlTower continuation for checkpoint growth,
   host load, terminal status, and complete JSON. Do not create a duplicate run.
   Then complete #8752 and its blocking structural propagation issue #8800.
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
