---
issue: 15734
summary: "Add SRP-7 registry-generated SOPs, biomechanical semantic glossary, and list/describe CLI."
dl_state: "in_review"
next_step: "Merge after CI green."
branch: "ag/15734-srp7-registry-sops-glossary"
---

Adds `src/tools/swing_report_package/registry_docs.py` deriving item descriptions, runnable CLI/Python SOP snippets, deterministic Markdown/JSON documentation, and a biomechanical semantic glossary directly from `package_catalog.json` via `load_catalog()`, plus unit tests in `tests/unit/tools/swing_report_package/test_registry_docs_15734.py`.
