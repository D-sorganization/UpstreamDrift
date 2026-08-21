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
| 1 | #8816 | Provenance | Critical | Cross-engine dashboard charts one aggregate robustness score replicated per engine |
| 2 | #8817 | Provenance | Critical | Cross-engine desktop GUI silently substitutes a 2-DOF stub, charts stay labeled as real engines |
| 3 | #8818 | Provenance | Critical | Ball Flight GUI wind/altitude/aero controls never read by the simulation |
| 4 | #8819 | Provenance | Critical | Swing→Flight Pipeline engine selector is cosmetic; engine_name never used |
| 5 | #8820 | Provenance | High | Unified dashboard exports carry no engine/model/run/timestamp identity |
| 6 | #8821 | Provenance | High | JSON/CSV exports get no provenance sidecar (binary formats do) |
| 7 | #8822 | Provenance | High | ProvenanceInfo has no engine field; callers omit model_path |
| 8 | #8823 | Provenance | High | ProvenanceValue UI provenance system built+tested but unused |
| 9 | #8824 | Provenance | High | Workspace project/session lineage subsystem has zero consumers |
| 10 | #8825 | Provenance | High | Launch Monitor plots stay stale across project changes |
| 11 | #8826 | Provenance | Medium | Launch Monitor data export not linked to its reproducibility manifest |
| 12 | #8827 | Provenance | Medium | Pose Studio engine status pill stale after silent mock downgrade |
| 13 | #8828 | Provenance | Medium | Shared plot titles/exports carry no identity; include_metadata ignored |
| 14 | #8829 | Provenance | Medium | Engine dashboards swallow model-load failures; engine name invisible embedded |

## Status

Review in progress. If usage runs out before completion, the filed issues
are the durable output; the table above is the index.
