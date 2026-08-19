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
  per hidden layer.
- The `tests` job is **change-scoped**: touching `src` coverage targets runs a
  dependency-light lane over `tests/unit/shared_python`, `tests/unit/sidekick`,
  `tests/unit/robotics`, and the mypy-budget test, where debt stays invisible
  until a PR does real source work.

### Two Debt Ledgers — Remove-Only Ratchets

`scripts/config/unit_gate_quarantine.json` (520 node IDs, #8766) skips tests
that fail deterministically on main from the hollow-merge era, applied by
`_apply_unit_gate_quarantine` in `tests/conftest.py` only when
`UNIT_GATE_QUARANTINE=1` — which only the unit gate and `tests` job set.
`scripts/config/dry_duplication_quarantine.json` (#8763) does the same for
duplication fingerprints. **Entries may only be removed, never added** — a new
failure must red the gate. #8768 clears main's own ruff/bandit/frontend debt.

### Traps That Cost Real Time

- **Committed evidence records hash-pin their sources.** Fourteen files under
  `tests/research/` are frozen by `source_sha256` maps in
  `docs/research/proximal_distal_energy_transfer/data/`. Editing one — even to
  add a pytest marker — breaks evidence currency; use
  `scripts/config/suite_marker_baseline.json` instead, and never rewrite a
  pinned hash without rerunning the experiment behind it.
- **A small PR can carry a huge payload.** Two "one-line optimization" PRs
  (#8638, #8746) were cut from stale bases and reverted main by thousands of
  lines. Compare diffstat against changed-file count before trusting one.
- **Never count CI failures with `tail`** — one showed six of 447.

## Program Authority

- Epic #8557 governs the proximal-to-distal program and #8595 retains the
  photographed agenda; both stay open.
- #8668 governs subject-scaled articulated contact; children #8676/#8678/#8680/
  #8682 completed inertia, contact projection, bilateral forwarding, and slack.
- #8684 governs distributed grip, shaft, and ground. Every child merged — grip
  (#8685), shaft (#8697 via #8715/#8717), ground (#8719 via #8723) — and it
  stays open on its declared gaps #8751 (grip friction and loss of contact) and
  #8752 (manufactured solutions, parameter uncertainty), the next research work.
- #8556 remains open: no governed participant dataset contains synchronized
  bilateral six-axis grip wrenches. Synthetic traces cannot replace it.
- NotebookLM review remains blocked on manual Google reauthentication. Never
  automate credentials, authentication dialogs, CAPTCHA, or two-factor steps.

## Qualified Baseline — And Its Limits

Native MuJoCo and robotics Pinocchio independently qualify the 20-coordinate
rigid tree over 234 closed states, and every tier through ground retains power,
passivity, energy, refinement, geometry, and engine-parity controls (numbers in
`SPEC.md`). All of it is a synthetic structural reference — not equipment
calibration, anatomy, physiology, or coaching guidance. The adverse results
below are load-bearing; do not quietly drop them.

- Of 384 coupled-versus-rigid shaft cells, 126 match on load and work; speed
  differences span `-0.0285` to `+0.0212 m/s` (82 negative), **rejecting a
  universal passive-shaft speed benefit**.
- The preregistered ground screen admits **0/384** coupled--fixed cells, because
  only the coupled path contains ground damping. A labeled post-hoc screen
  admits 60 cells with mixed signs — sensitivity evidence, not the registered
  estimand. **Do not read unmatched positive speed differences as a
  ground-pathway benefit.**
- Initialization is not innocuous: natural-zero, gravity-only, and conditional
  starts gave peak ground forces of 32.8, 565.5, and 510.3 N. The conditional
  solve balances only base generalized forces; use natural-zero for exact-state
  killswitch comparisons.
- Coarse steps leave the linear domain (1.0 ms at `(0,0)`; 0.50 ms at `(8,0)`).
  Regenerate the shaft FE basis under Windows — native WSL cannot solve the
  eigenproblem on its lean SciPy stack, and native runs hash-check that basis.

## Vendored Tools Dependency (`vendor/ud-tools`)

Tools is a **leaf dependency vendored as a submodule**. Never edit shared code
inside the vendored copy — Tools is the source of truth and edits here are
orphaned (Tools #4495 exists because three fixes were lost that way). The same
rule binds the **child copies** under `src/shared/python/`:
`tests/unit/repo_hygiene/test_tools_child_copy_contract.py` fails any PR editing
one, so when a gate demands a change inside a child copy the fix belongs in
Tools. `src/shared/python/ai/auth` is excluded from the placeholder scan for
exactly that reason; #8770 tracks the Tools-side cleanup.

Current pin: Tools `b0f7975ac` (PR #8767), carrying the **club-fitting** stack
(`shared/python/golf_club/`: mesh inertia tensor, shaft delivery deltas, OEM
fitting document, counterfactual engine) and the **heavy-hit** stack
(`impact_coupling.py` plus `shared/python/swing_sim/model_interchange/`, which
imports MJCF/URDF/`.osim` golfer models by XML parsing).

**Gate for any pin bump:** `tests/unit/test_gui_launcher_manifest_targets.py`
(25 tests) asserts every launcher manifest entry resolves to a real importable
target, catching a Tools rename that breaks a GUI. Run it with `PYTHONPATH` set
to repo root **and** `src`.

## Immediate Next Steps

1. Land #8768 (main's ruff, bandit, frontend-lock debt) — auto-merge armed.
2. Start #8751 or #8752, the declared gaps blocking #8684. Keep the adverse
   results above intact; a gap closes with evidence, never by restatement.
3. Burn down #8766 by removing ledger entries as clusters are fixed — theme API
   drift and the sidekick ownership manifest are the largest.
4. Do the Tools-side work in #8770 so the auth child copy can be unfrozen here.
5. Continue toward calibrated unilateral 3D contact, full-delivery matching and
   uncertainty, and a governed human holdout — #8556/#8557 need real data.

## Gate Commands

Run this locally before pushing; it mirrors the sequential CI steps:

```bash
python scripts/check_spec_paths.py && python scripts/check_root_clutter.py
python scripts/check_test_layout.py && python scripts/check_pytest_intree_testpaths.py
python scripts/ci/check_suite_marker_ratchet.py && python scripts/ci/check_dry_duplication_gate.py
python scripts/check_module_size_budget.py --max-lines 1500 --include src
python scripts/check_doc_size_budget.py
python -m ruff check . && python -m ruff format --check .
```

Research reproduction runs from `scripts/research/proximal_distal_energy/` as
`python -m` modules: `generate_articulated_shaft_structural_basis` (Windows
Python only), then `run_articulated_shaft_atlas`,
`run_articulated_ground_atlas`, and
`run_articulated_ground_posthoc_sensitivity`. Re-validate the ledger with
`claim_audit`, `claim_evidence_integrity`, `momentum_question_readiness`, and
`qualify_open_release`, each with `validate`, then `pytest tests/research -q`.

Do not infer human technique, physiology, injury, timing demand, or coaching
advice; close #8556/#8557; bypass branch protection; force-push; admin-merge;
add ledger entries; edit hash-pinned or Tools-owned files; or rerun unchanged
runner-capacity failures.
