# ADR-0042: Engineering Design Manual Authority and Release Boundary

- Status: Accepted
- Date: 2026-08-25
- Decision Makers: UpstreamDrift maintainers and documentation owners
- Related Issues/PRs: #9064, #9066

## Context

UpstreamDrift contains source code, user guides, ADRs, long-form scientific
articles, generated publications, and several renderer pathways. None currently
defines one repository-wide, calculation-level design manual that traces theory,
contracts, implementation, tests, uncertainty, user surfaces, and artifacts.
Treating generated PDF, DOCX, HTML, LaTeX, or an existing research product as a
second editable authority would make drift and unsupported promotion likely.

The cross-repository program owns common calculation-registry and publication-
projection contracts. UpstreamDrift must consume those contracts without
copying their schemas or transferring scientific authority out of this repo.

## Decision

`manuals/upstreamdrift` is the sole editable QMD authority for the engineering
design manual. Existing documentation remains separately governed source or
reference material, not a mutable mirror of this manual. Generated LaTeX, PDF,
DOCX, and HTML are non-editable release artifacts.

`config/design_manual_governance.json` binds the owning epic, source path,
program contract identifiers, calculation registry, impacted paths, artifact
formats, required release evidence, Git policy, and agent context. The offline
`scripts.check_design_manual_governance` gate validates the exact contract and
keeps publication blocked until inventory, content, freshness, toolchain,
semantic, visual, accessibility, digest, and human-approval evidence exists.

Unknown, external, provisional, unavailable, or model-conditioned inputs and
claims retain those classifications. An exemption must be structured, owned,
narrow, justified, reviewed, and expiring; it cannot grant scientific or
publication authority.

## Alternatives Considered

1. Edit generated LaTeX, PDF, DOCX, or HTML beside QMD. Rejected because it
   creates multiple mutable authorities.
2. Treat the proximal-to-distal publication or user manual as the design manual.
   Rejected because they have different scopes and release contracts.
3. Copy the program-owned schemas into UpstreamDrift. Rejected because duplicate
   contracts would drift and violate cross-repository ownership.
4. Allow publication from a structurally successful render. Rejected because
   rendering does not qualify equations, evidence, semantic parity, or pages.

## Consequences

- Positive: one editable authority, explicit ownership, deterministic failures,
  and a versioned path from calculation inventory to approved artifacts.
- Negative: all formats remain blocked until later subepics supply the required
  source, toolchain, semantic, visual, and approval evidence.
- Follow-ups: UP-D1 inventories calculations; UP-D2 types their pathways; UP-D3
  qualifies the renderer; UP-D4 through UP-D8 complete content, freshness, QA,
  and publication; UP-D9 sustains context and ownership.

## Validation

Contract tests mutate permissive fields, unsafe paths, empty approved
registries, and generated artifacts and require deterministic rejection. CI and
pre-commit run `python3 -m scripts.check_design_manual_governance`. Exact-head
tests, formatting, typing, title-case, governance, and protected review remain
mandatory before merge.
