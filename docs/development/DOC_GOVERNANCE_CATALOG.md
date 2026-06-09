# Documentation Governance Catalog

Issue #3839 identified documentation sprawl across user manuals, process
directories, and generated Sphinx artifacts. The first enforceable cleanup step
is a catalog and size budget that makes future drift visible in CI without
requiring high-risk mass moves.

## Canonical Surfaces

- Rendered user documentation: <https://upstream-drift.readthedocs.io>
- Sphinx source and generated artifacts: `docs/sphinx/`
- Repository documentation inventory: `docs/index.md`
- Markdown and Quarto size budget: `scripts/config/doc_size_budget.json`

## Local Checks

```bash
python scripts/check_doc_catalog.py
python scripts/check_doc_size_budget.py
python scripts/check_docs_governance.py
```

`check_doc_catalog.py` verifies that every top-level directory under `docs/`
has an owner, stability tag, and description in `docs/index.md`. It also
requires the README Documentation Hub link to point at the rendered
documentation URL from `pyproject.toml`.

`check_doc_size_budget.py` fails when a committed Markdown or Quarto file is
larger than 50 KB unless it has a temporary exception with an owner and
expiration date.

## Remaining Issue #3839 Work

This PR intentionally avoids bulk document moves and manual decomposition of the
633 KB user manual because other agents are working in nearby documentation
areas. The budget exception for
`docs/user_guide/upstream_drift_user_manual.md` expires on 2026-08-01 and
should be removed after the manual is split into Sphinx chapters.
