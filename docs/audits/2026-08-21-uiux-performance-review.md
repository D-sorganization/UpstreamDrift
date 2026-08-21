# UI/UX, Performance, and Proximal–Distal Modeling Review — 2026-08-21

Second review wave requested by the repository owner, following the
integration/organization review (PR #8815, issues #8816–#8876). Three
axes this wave, each verified against the code before filing:

1. **UI/UX** — desktop launcher and shared UI infrastructure; tool GUIs
   and the React/Tauri web app. Responsiveness, error-state UX,
   feedback, consistency, accessibility, state persistence.
2. **Performance** — runtime hot paths (vectorization, per-frame
   allocation, plotting), startup cost, API latency, caching, IPC, and
   Rust-acceleration opportunities.
3. **Proximal–distal modeling** — critical technical review of the
   energy-transfer research program (epics #8557/#8668/#8684):
   assumptions, energy accounting, friction modeling, cross-engine
   parity, experimental design, claim governance.

Dedup notes: `.scratch/uiux-issues/` holds 28 pre-drafted web-app UI/UX
issues; at least drafts 01–03 were filed as #7415–#7417 (closed). New
findings were checked against those drafts and against open issues
before filing.

## Findings and Filed Issues

_(Populated as the review proceeds; each row is a filed GitHub issue.)_

| # | Issue | Area | Severity | Title |
|---|-------|------|----------|-------|

## Status

Review in progress. If usage runs out, the filed issues are the durable
output; the table above is the index.
