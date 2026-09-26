# Past Handoff — Main Red on Bandit B314 in the Coverage Gate Checker (#10989)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-10989`
- Branch: `fix/10989-coverage-gates-defusedxml` (baseline `84ac37579`)
- Commit: `SELF`
- Pull request: #10994 (draft)
- Issue: #10989 (fleet-main-health: CI Standard red on main); development-log entry DL-#10965
- Cause: #10988 landed `scripts/check_coverage_gates.py` parsing Cobertura XML with `xml.etree.ElementTree.parse`; the push-lane full-tree `bandit -ll -ii` flags B314 and fails `security-scans`, which fails `quality-gate`.
- Fix: parse with `defusedxml.ElementTree` (a core dependency, the convention in `scripts/config/coverage_enforcer.py`); `Element` imported under `TYPE_CHECKING` for the annotation. New test `test_xml_entity_expansion_is_rejected` was red on stdlib ET and is green now.
- Validation: `pytest tests/unit/scripts/test_check_coverage_gates.py` -> 13 passed (new test carries `@pytest.mark.unit` for the suite-marker ratchet); `bandit -ll -ii scripts/check_coverage_gates.py` clean; ruff and mypy clean.
- Next step: CI green, mark ready, arm via `automerge_guard.py`; #10989 closes itself when main's next CI Standard run succeeds.

---

# Implementation Handoff — #9406 Sidekick Shadow Retirement 2026-09-26

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-ud-9406-shadow`
- Branch: `claude/ud-9406-sidekick-shadow-retire`
- Baseline commit: `b8c27a7d2` (origin/main, after #10988)
- Implementation commit: `SELF`
- Pull request: #10991 (draft)
- Governing issue: #9406 (DL-#9406)
- Lease session: `claude-deskcomputer-20260925-ud`

## Objective and Status

- Objective: stop UpstreamDrift shipping stale copies of Tools-owned `sidekick` modules, so the
  quarantined tests that demand their absence pass.
- Status: agy (Gemini 3.8 Flash) slice on DeskComputer, reviewed and trimmed by Claude.
  - Deleted, now resolved from `vendor/ud-tools/src/shared/python/sidekick/`: `standalone/`
    (runner, window, session_store, preferences, onboarding), `persistence/`, `__main__.py`,
    `ui/tools_sidebar/default_tabs.py`.
  - `embedded_tool_bootstrap._bootstrap_python_paths`: vendored or explicit Tools paths now
    precede UpstreamDrift's own, as the bootstrap tests require.
  - `sidekick.spec`: Windows icon is the pinned `vendor/ud-tools/assets/tools_icon_hq.ico`.
  - Tests retargeted to the vendor API: pickle is no longer auto-detected (`test_data_io`), C3D
    export needs `C3D_ALLOW_ANY_EXPORT_PATH` (`test_c3d_reader`), and `test_cli` patches the
    launcher factory under every import prefix.
  - 14 quarantine IDs retired (155 → 141).
- Dropped from the agy output: an `architecture_budget.json` exception (the violation predates
  this branch) and an `ai/gui/assistant_panel.py` vendor copy (the backward-compat identity tests
  still fail because vendor imports through the `shared.python` prefix).

## Validation

- `pytest -n 6 -o addopts="" --tools-mode vendored <33 sidekick/packaging/bootstrap IDs>`:
  14 passed (all retired); the rest stay quarantined.
- `UNIT_GATE_QUARANTINE=1 pytest -n 6 --tools-mode vendored tests/unit/sidekick tests/unit/launcher
tests/launchers tests/unit/packaging tests/unit/repo_hygiene tests/integration/sidekick
tests/ui/c3d_viewer`: 67 failed / 2686 passed here vs 65 / 2687 on `b8c27a7d2`; the only new
  ID (`test_embed_adapter::test_create_main_widget_returns_qwidget`) passes alone twice
  (xdist order). The #9406 slice also fixed one main failure.
- `scripts/ci/check_unit_gate_quarantine.py`: contract passed, 141 IDs.

## Blockers and Risks

- The same launcher/launchers failures appear on main in this Windows environment.
- #9411 (quarantine burn-down) edits the same ledger; whichever lands second takes the union
  of removals.

## Next Steps

1. Open the draft PR, add the SPEC row keyed by its number, get `quality-gate` green.
2. Mark ready and arm through `scripts/automerge_guard.py`.
