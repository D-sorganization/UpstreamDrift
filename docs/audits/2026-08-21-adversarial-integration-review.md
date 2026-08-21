# Adversarial Integration & Organization Review — 2026-08-21

Scope requested by the repository owner: assess whether UpstreamDrift forms a
coherent, fully integrated whole — launcher/tool integration completeness,
data provenance in the GUIs (which engine/swing/trial produced what), linkage
from GUI tabs to supporting calculation documentation, and overall repository
organization. Every confirmed finding is filed as a GitHub issue; this
document is the running index of that review.

Review branch: `claude/upstreamdrift-adversarial-review-x29y1c`.

## Method

Parallel deep-dive audits over five axes, each verified against the code
before filing:

1. Launcher tile ↔ tool integration completeness (registry vs. reality).
2. GUI data provenance — engine / swing / trial labeling on displayed and
   exported data.
3. Calculation documentation ("calc sheets") and their linkage from GUIs.
4. Repository organization coherence (duplicated/orphaned trees, entry
   points, docs sprawl, config sprawl).
5. Motion pipeline and simulation workflow end-to-end integration.

Pre-existing open issues checked for overlap before filing: #8763, #8776,
#8689, #8695 (duplication/shadow-package debt), #8766, #8735 (test debt).

## Findings and Filed Issues

_(Populated as the review proceeds; each row is filed as a GitHub issue.)_

| # | Issue | Area | Severity | Title |
|---|-------|------|----------|-------|

## Status

Review in progress. If usage runs out before completion, the filed issues
are the durable output; the table above is the index.
