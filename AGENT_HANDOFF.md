# Agent Handoff — UpstreamDrift

Last updated: 2026-08-20

## Active Publication-Quality Slice (#8451)

PR #8793 owns the PDF quality contract: UpstreamDrift is authoritative,
AffineDrift is its pinned publisher, and Tools/Sidekick only link. The
web-linearized 231-page candidate passes full-page inspection; missing tags,
110 Type 3 resources, and two unembedded resources block archival release.
Phase 0 landed in UpstreamDrift #8791, AffineDrift #3884, and Tools #4586.
After the protected merge, sync the exact revision and PDF/manifest digests to
AffineDrift. #8789 owns Docker/quarantine/baseline; #8556, equipment calibration,
and archival/PID release remain open. Details:
`docs/research/proximal_distal_energy_transfer/PUBLICATION_QUALITY.md`.

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
    nominal shaft recomputation completed at 126/384 with all four computation
    hashes verified against remote basis `fbff8dc53`. Nominal ground is active
    in exec session `64656` with 20 workers and 72 independently digest-bound
    branch checkpoints; a real restart-equivalence test passes.
    Completion-only headline/structural evidence tests stay untracked until data finish.
  - #8800 is **OPEN** and blocks #8752. Its governed generator regenerates all
    13 phase states for cases 0/8/9/17; nominal is 52/52 feasible and reproduces
    committed states within 1e-8 rad. JSON/NPZ evidence rejects source/content
    drift. `ArticulatedAtlasAuthority` binds exact scaled models and retains
    infeasible phases. Session `15825` has nominal 52/52 with the exact prior
    digest; height-low retains case 0/phase 12 `ik_nonconvergence`; height-high
    and mass-low are 52/52 feasible; mass-high is running. Propagation is open.
- The release has 295 atomic claims and 40 reviewed release claims; all 40
  retain at least one scientifically open gate. #8724 still owns normalized
  four-way adjudication and independent review.
- PR #8793 canonicalizes UTF-8 claim evidence before hashing; #8789 retains
  Docker/quarantine/baseline and clean-checkout truth-recovery scope.
- #8556 remains open: no governed participant dataset contains synchronized
  bilateral six-axis grip wrenches. Synthetic traces cannot replace it.

## Qualified Baseline — And Its Limits

Native MuJoCo and robotics Pinocchio independently qualify the 20-coordinate
rigid tree over 234 closed states, with power, passivity, energy, refinement,
geometry, and engine-parity controls. All of it is a synthetic structural
reference — not equipment calibration, anatomy, physiology, or coaching guidance.

- Of 384 coupled-versus-rigid shaft cells, 126 match on load and work; speed
  differences span `-0.0285` to `+0.0212 m/s` (82 negative), **rejecting a
  universal passive-shaft speed benefit**.
- The preregistered ground screen admits **0/384** coupled--fixed cells (ground
  damping asymmetry). A post-hoc screen admits 60 cells with mixed signs.
  **Do not read unmatched positive differences as a ground-pathway benefit.**
- Initialization: natural-zero, gravity-only, and conditional starts gave peak
  ground forces of 32.8, 565.5, and 510.3 N.

## Vendored Tools Dependency (`vendor/ud-tools`)

Tools is a **leaf dependency vendored as a submodule**. Never edit the vendored
copy or child copies under `src/shared/python/`; Tools is the source of truth.

## Immediate Next Steps

1. Publish #8451, then sync its exact revision and digests through AffineDrift.
2. Continue #8789's remaining Docker/quarantine/baseline work without overlap.
3. Complete #8752 and burn down #8766 without widening either contract.

## Gate Commands

Run this locally before pushing; it mirrors the sequential CI steps:

```bash
python scripts/check_spec_paths.py && python scripts/check_root_clutter.py && \
  python scripts/check_test_layout.py && python scripts/check_pytest_intree_testpaths.py
python scripts/ci/check_suite_marker_ratchet.py && python scripts/ci/check_dry_duplication_gate.py
python scripts/ci/check_architecture_budget.py
python scripts/check_module_size_budget.py --max-lines 1500 --include src
python scripts/check_doc_size_budget.py
python -m ruff check . && python -m ruff format --check .
```

Research validation:

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
