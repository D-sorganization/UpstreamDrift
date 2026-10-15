---
issue: 12544
summary: "Stop committing the generated agent-context views (README.md, index.html); render them on demand in CI and locally; reviews, catalog and contracts stay committed"
title: "Agent-Context Views Rendered on Demand"
owner: "claude"
branch: "fix/12544-agent-context-views-generated"
paths: ".gitignore; .pre-commit-config.yaml; AGENTS.md; docs/agent_context; docs/index.md; docs/README.md; scripts/generate_docs_map.py; tests/agent_context/test_generated_views_wiring.py; tests/unit/ci/test_generate_docs_map.py"
---

Depends on the workflow-only PR (branch fix/12544-agent-context-workflow) merging first; it adds the CI render step before agent_context check.
