# Review Rules: UpstreamDrift

Repo-specific rules for anyone reviewing an UpstreamDrift pull request. The
author checks them first with the `fleet-review` skill. A breach is a finding
only when it's cited here or in `AGENTS.md` / `CLAUDE.md`, by file and
section.

## Scope and Size

- Under 500 changed lines after formatters, targeting `main`, never stacked.
- Tools code (chat, sidekick, shared) is edited in the upstream Tools
  repository, never shadowed in `src/shared/python/` (AGENTS.md, Where to
  Edit).

## Reuse Before New Code

- New functionality that duplicates a primitive in `src/shared/python/`, or a
  tile in `src/tools/` or `src/launchers/` that already does most of the job,
  is a finding (AGENTS.md, section A).
- Code sits where the decision tree puts it (AGENTS.md, section C): one engine
  under `src/engines/<engine>/`, two or more engines under
  `src/shared/python/<topic>/`, a desktop tool as a package under `src/tools/`
  with a tile in `src/config/models.yaml`, and a shared math kernel under
  `rust_core/` with a pure-Python parity test.
- A relocation leaves a thin shim at the old path with a `DeprecationWarning`.
- No file over 1200 lines unless listed in
  `scripts/config/file_size_budget.json`. No `print()` in `src/`; use
  logging (CLAUDE.md, CI Requirements).
- A new user-facing PyQt6 feature adds or updates its entry in
  `src/config/feature_parity.json` (CLAUDE.md, Feature Parity Registry).

## Tests

- New behaviour needs a test that fails on `main` (TDD), under
  `tests/<area>/<subarea>/`, mirroring `src/` (`tests/README.md`).
- Each test carries the marker for its tier from `pyproject.toml`. A test that
  needs an engine, GL or mocap fixtures carries the matching `requires_*`
  marker, and a real simulation run is `live_simulation`.
- A test that imports PyQt6 or matplotlib skips cleanly when the import fails.
  Desktop tools keep pure-data math in a `core` module, testable without the
  GUI.
- Loosening a tolerance, budget or coverage floor needs a
  `Tolerance-Change-Evidence: #N — <numbers>` commit trailer.

## Docs and Generated Files

- Document titles and headings use title case
  (`scripts/check_document_title_case.py`).
- No loose Markdown at the `docs/` root other than its README and index
  (`tests/docs/test_docs_structure.py`).
- A new root file is added to both `scripts/check_root_clutter.py` and
  `scripts/config/root_allowlist.json`.
- Handoffs go in a change fragment under `changes/`
  (`shared_scripts/changes_fragment.py`). Direct edits to `SPEC.md`'s change
  log, `docs/development/DEVELOPMENT_LOG.md` or
  `docs/development/HANDOFF.md` are findings, and `AGENT_HANDOFF.md`
  is not the per-PR handoff target.

## Merging

- PRs open ready, never as drafts.
- A PR that deletes files says `deletions-acknowledged: yes` on its own line
  in the body. Otherwise the arm guard refuses it.
- PRs are armed only with `scripts/automerge_guard.py`, never with a
  merge-method flag, `--admin` or a ruleset change.
- Commits use a GitHub noreply email.
- `Closes #N` sits on its own line. An issue closes only through a merged PR
  or an exempt label.
