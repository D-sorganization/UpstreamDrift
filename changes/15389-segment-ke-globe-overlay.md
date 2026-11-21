---
issue: 15389
summary: "Segment kinetic-energy globes, per-frame KE time series API, and translational/rotational split"
dl_state: "in_review"
next_step: "CI on PR #15391."
branch: "cursor/segment-ke-globe-view-f14a"
---

Expose per-body translational and rotational kinetic energy, add an engine-agnostic
segment KE time-series contract with MuJoCo provider, and render COM-centered
globes with swing peak highlighting. UI and docs state that globe peaks are a
kinematic proxy, not inter-segment energy flow (Robertson-Winter follow-up #15390).