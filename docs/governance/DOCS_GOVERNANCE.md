# Documentation Governance

## Canonical Documentation

- Repo docs index: `docs/README.md`
- Assessments index: `docs/assessments/README.md`
- ADR index: `docs/adr/README.md`

## Freshness Rules

- Any change in `docs/assessments/**` must update `docs/assessments/README.md`.
- Any change in `docs/adr/*.md` (excluding template/index) must update `docs/adr/README.md`.
- ADR changes must follow `docs/adr/ADR_TEMPLATE.md`.

## Ownership

- Architecture decisions: engineering leads.
- Assessment docs taxonomy and archive policy: maintainers.

## Engineering Design Manual

- Canonical editable source: `manuals/upstreamdrift` QMD.
- Governance contract: `config/design_manual_governance.json`.
- Calculation inventory: `manuals/upstreamdrift/calculation-registry.json`.
- Offline gate: `python3 -m scripts.check_design_manual_governance`.
- Generated LaTeX, PDF, DOCX, and HTML are non-editable artifacts. UP-D8 must
  record semantic, page-render, accessibility, digest, and human-approval
  evidence before any public projection.
- Existing user guides, ADRs, research publications, and their source trees are
  separate governed products, not alternate mutable manual authorities.
