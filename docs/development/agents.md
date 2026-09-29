# Agent Guidance Reference

`CLAUDE.md` is authoritative. This page is reference-only background for
contributors and agents and preserves the non-binding operational context that
used to be duplicated in `AGENTS.md`.

## Canonical Sources

- `CLAUDE.md` defines contributor policy, quality gates, and repo conventions.
- `CONTRIBUTING.md` covers the local contributor workflow and minimum PR steps.
- `SPEC.md` documents the repository's current functionality and architecture.

## Working Notes

- PRs target `main`, but contributors may use focused topic branches such as
  `fix/...`, `feat/...`, `chore/...`, or `claude/...`.
- Python support starts at 3.11 from `pyproject.toml` and `install.sh`; the
  standard CI matrix tests Python 3.11 and 3.12.
- Run `python3 -m ruff check .`, `python3 -m ruff format --check .`,
  `python3 -m mypy .`, and `python3 -m pytest` before opening a PR when the
  affected surface makes those checks relevant.
- The optional Rust workspace lives under `rust_core/`; local development uses
  `maturin develop` from that tree when Rust-backed extensions are involved.
- Shared Tools code is primarily consumed through `vendor/ud-tools/`. Use
  `scripts/setup_tools_workspace.sh` only when you need an editable sibling
  checkout for coordinated work.

## Documentation Hygiene

- Store development notes, plans, and migration write-ups under
  `docs/development/` rather than the repository root.
- Treat this page as a background reference. When policy changes, update
  `CLAUDE.md` first and then adjust this page only if extra context is still
  useful.
