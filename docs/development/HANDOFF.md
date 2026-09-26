# Implementation Handoff — Bump `vendor/ud-tools` to Tools Main With K0 + K3a (#10944)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/agy-10944`
- Branch: `agy/issue-10944`
- Baseline commit: `4e97be41b10355b3f61ce67aeffec7bc990ad864` (origin/main)
- Implementation commit: `SELF`
- Pull request: #10944
- Governing issue: #10944 (prereq for #10943)
- Lease session: `antigravity-deskcomputer-20260925-10944`

## Objective and Status

- Objective: Advance the `vendor/ud-tools` gitlink from `a9ed0e7c` to Tools `main` commit `95ed6b47857e9a47211ab1973d02b28beae718bc` containing K0 (Tools#5348, knowledge-pack engine) and K3a (Tools#5350, Sidekick Wizards). Synchronize the child copy of `src/shared/python/ai/`.
- Status: Complete / ready for draft PR
- Completed:
  1. Verified upstream Tools commit `95ed6b47857e9a47211ab1973d02b28beae718bc` is on Tools `main` via GitHub compare API.
  2. Checked out and staged `vendor/ud-tools` at `95ed6b47857e9a47211ab1973d02b28beae718bc`.
  3. Synchronized child copy of `src/shared/python/ai/` from pinned vendor:
     - Added `src/shared/python/ai/knowledge/` package (K0 engine: chunking, manifest, pack, sources, wizard).
     - Added `src/shared/python/ai/wizards.py` (K3a glue: `wizard_for`, `knowledge_for_context`).
     - Updated `src/shared/python/ai/adapters/base.py` to inject Wizard knowledge context.
     - Updated `src/shared/python/ai/gui/_panel_tools.py` to support `project_root` parameter and query Wizard before RAG store.
     - Preserved `src/shared/python/ai/gui/assistant_panel.py` without branch divergence to satisfy Tools child-copy contract.
     - Updated `src/shared/python/ai/rag/context_provider.py` and `tests/test_rag_context_provider.py` with deprecation notices.
  4. Synchronized Rust kernel and pip package pins:
     - `Cargo.toml`: `tools-core` rev -> `95ed6b47857e9a47211ab1973d02b28beae718bc`.
     - `requirements-tools.txt`: `ud-tools` git pin -> `95ed6b47857e9a47211ab1973d02b28beae718bc`.
     - `src/config/impact_acceptance.json`: `reconciled_against.tools` -> `95ed6b47857e9a47211ab1973d02b28beae718bc`.
     - `src/shared/python/tour_baselines/reconciliation.py`: `pinned_commit_sha` -> `95ed6b47857e9a47211ab1973d02b28beae718bc`.
     - `tests/unit/tour_baselines/test_reconciliation.py`: updated expected sha -> `95ed6b47857e9a47211ab1973d02b28beae718bc`.
  5. Regenerated divergence inventory and agent context views:
     - `docs/shared_tools/divergence_inventory.v1.json` and `.md`.
     - `docs/agent_context/README.md` and `docs/agent_context/index.html`.
  6. Added TDD test: `tests/unit/ai/test_knowledge_and_wizards.py`.
  7. Added Section 12 Change Log row in `SPEC.md` and `DL-#10944` entry in `docs/development/DEVELOPMENT_LOG.md`.

## Files and Decisions

- Files changed:
  - `Cargo.toml`: bumped `tools-core` git revision to `95ed6b47857e9a47211ab1973d02b28beae718bc`.
  - `requirements-tools.txt`: bumped `ud-tools` pip pin to `95ed6b47857e9a47211ab1973d02b28beae718bc`.
  - `src/config/impact_acceptance.json`: bumped `reconciled_against.tools` sha and date.
  - `src/shared/python/ai/knowledge/`: imported K0 knowledge-pack engine from vendor.
  - `src/shared/python/ai/wizards.py`: imported K3a Sidekick Wizards glue from vendor with `src.shared.python` imports.
  - `src/shared/python/ai/adapters/base.py`: integrated Wizard knowledge context into `build_prompt_context`.
  - `src/shared/python/ai/gui/_panel_tools.py`: registered Wizard knowledge query in `search_knowledge_base`.
  - `src/shared/python/ai/rag/context_provider.py`: added deprecation warning per Tools#5346.
  - `src/shared/python/ai/tests/test_rag_context_provider.py`: added deprecation warning test.
  - `src/shared/python/tour_baselines/reconciliation.py`: bumped `_TOOLS_REVISION.pinned_commit_sha`.
  - `tests/unit/tour_baselines/test_reconciliation.py`: updated test assertion to match new pin.
  - `tests/unit/ai/test_knowledge_and_wizards.py`: TDD acceptance tests for K0 and K3a.
  - `docs/shared_tools/divergence_inventory.v1.json` & `.md`: regenerated divergence ledger.
  - `docs/agent_context/README.md` & `docs/agent_context/index.html`: re-rendered agent context views.
  - `SPEC.md`: registered #10944 change log row.
  - `docs/development/DEVELOPMENT_LOG.md`: added `DL-#10944` active entry.
  - `vendor/ud-tools`: submodule gitlink bumped to `95ed6b47857e9a47211ab1973d02b28beae718bc`.

## Validation

- `check_tools_pins.py`: PASSED (gitlink, Cargo.toml, requirements-tools.txt consistent).
- `check_seam_drift.py`: PASSED (14 notes, 0 errors).
- `divergence_inventory.py --check`: PASSED (inventory is current).
- `agent_context --root . check`: PASSED (clean, 0 errors).
- `test_companion_catalog.py`: PASSED (40/40 passed).
- `test_tools_child_copy_contract.py`: PASSED (20/20 passed, including diff convergence).
- `test_impact_acceptance_matrix.py`: PASSED (27/27 passed).
- `test_reconciliation.py`: PASSED (3/3 passed).
- `test_divergence_inventory.py`: PASSED (10/10 passed).
- `test_check_seam_drift.py` + `test_check_tools_pins.py`: PASSED (18/18 passed).
- `test_check_spec_changelog_duplicates.py` + `test_spec_changelog_integrity.py`: PASSED (18/18 passed).
- `test_knowledge_and_wizards.py`: PASSED (3/3 passed).
- `test_rag_context_provider.py`: PASSED (24/24 passed).
- AI test suite (`tests/ai`, `tests/unit/ai`, `tests/unit/shared_python/ai`): Exactly 14 pre-existing failures matching stock main baseline (0 new failures, 0 regressions).

## Blockers and Risks

- None. Ready for review and merge.

## Next Steps

1. PR review & CI verification: ensure all quality gates pass.
2. After merge: K3b (#10943) wires the Sidekick Wizards into UD's assistant panel.

## Change Log

- `SELF` — #10602 C1/C2: club-only matrix scores only complete recorded fit outcomes; no synthetic package or defaulted metrics (DL-#10602).
- `SELF` — Reconcile child-copy convergence, divergence inventory, and agent context views (#10944).
