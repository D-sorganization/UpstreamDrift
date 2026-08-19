# Agent Handoff — UpstreamDrift

Last updated: 2026-08-19

This is current operational state. Historical detail belongs in git/GitHub.

## Read This First — How Merging Works Now

Three workflows used to publish the required `quality-gate` status, so whichever
reported first satisfied branch protection and PRs merged while the real
aggregate was failing. #8747 fixed this and #8754 is closed. Consequences:

- The **only** required check is `quality-gate`, published solely by the
  aggregate job in `.github/workflows/ci-standard.yml`. The LoD gate publishes
  `lod-quality-gate`; docs CI publishes `docs-quality-gate`. **Never rename a
  job back to `quality-gate`** — that recreates the collision.
- If your run sits in `action_required`, wait: `approve-same-repo-runs.yml`
  approves same-repository runs every five minutes (fork PRs keep the manual
  gate). Do not disable it — branch updates come from a bot GitHub treats as a
  first-time contributor forever.
- `repo-structure-gates` runs its steps **sequentially and fail-fast**, so a red
  step hides every later one. Run the battery below first, or spend a CI cycle
  per hidden- Function length limit: `check_architecture_budget.py` enforces maximum 100
  lines per function for changed production Python files.

### Two Debt Ledgers — Remove-Only Ratchets

`scripts/config/unit_gate_quarantine.json` (520 node IDs, #8766) skips tests
that fail deterministically on main from the hollow-merge era, applied by
`_apply_unit_gate_quarantine` in `tests/conftest.py` only when
`UNIT_GATE_QUARANTINE=1` — which only the unit gate and `tests` job set.
`scripts/config/dry_duplication_quarantine.json` (#8763) tracks duplicate
fingerprints. **Entries may only be removed, never added**.
PR #8768 cleared main's ruff, bandit, XML security, and frontend lock debt.

### Traps That Cost Real Time

- **Committed evidence records hash-pin their sources.** Fourteen files under
  `tests/research/` are frozen by `source_sha256` maps in
  `docs/research/proximal_distal_energy_transfer/data/`. Never edit without
  updating the evidence manifest via `qualify_open_release write`.
- **Phantom Guard Scope Rules:** Conventional commit prefix for PR titles
  touching only `scripts/research/`, `docs/research/`, and `tests/research/`
  must use `test:`, `docs:`, or `chore:` rather than `feat:` (which requires
  diffs touching `src/`, `rust_core/`, or `api/`).

## Program Authority & Physics Epics State

- Epic #8557 governs the proximal-to-distal program; #8595 retains the agenda.
- #8668 governs subject-scaled articulated contact; all child issues completed.
- #8684 governs distributed grip, shaft, and ground:
  - #8751 (Grip Friction & Loss of Contact) **COMPLETED** (PR #8783): Multi-station
    Coulomb cone friction, power decomposition, unilateral contact transitions.
  - #8752 (Manufactured Controls & LHS Uncertainty) **COMPLETED** (PR #8783):
    Manufactured harmonic free-body & constrained checks, LHS parameter sweeps.
- Release Bundle: 568 artifacts and 295 claims validated with 0 mismatches.
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

Tools is a **leaf dependency vendored as a submodule**. Never edit shared code
inside the vendored copy — Tools is the source of truth. The same rule binds
child copies under `src/shared/python/`.

- Tools #4577 & #4579 merged: Delivered PyQt6 Club Tester workbench, React panel
  (488 LOC), model interchange (MJCF/URDF/.osim), and sanitized auth logging
  resolving `UpstreamDrift#8770`.

## Immediate Next Steps

1. Burn down #8766 by removing unit quarantine ledger entries as clusters are fixed.
2. Advance toward calibrated unilateral 3D contact, full-delivery matching and
   uncertainty, and a governed human holdout (#8556/#8557).
3. Ingest latest research release manifest hashes into companion articles.

## Gate Commands

Run this locally before pushing; it mirrors the sequential CI steps:

```bash
python scripts/check_spec_paths.py && python scripts/check_root_clutter.py
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
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
pytest tests/research -q
```

Do not infer human technique, physiology, injury, timing demand, or coaching
advice; close #8556/#8557; bypass branch protection; force-push; admin-merge;
add ledger entries; edit hash-pinned or Tools-owned files; or rerun unchanged
runner-capacity failures.erun unchanged
runner-capacity failures.
