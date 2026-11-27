---
issue: 15522
summary: "Exercise native MuJoCo Advanced Golf Analysis with headless QTest and release simulation resources on window close"
dl_state: "in_review"
next_step: "Independent exact-head review and required CI; owner requires the PR to remain unarmed."
owner: "codex"
branch: "codex/15522-mujoco-advanced-qtest"
---

Three native QTest flows select the shipped upper-body golf model, pause/resume,
record/reset, edit a kinematic joint, and close the window. Real MuJoCo loading,
workers, integration and recording run; only GPU rendering is replaced.

RED: closing the window left the simulation running (1 failed, 2 passed).
GREEN: closing releases both simulation widgets and stops owned timers.
Screenshots show the existing desktop layout through 1280-wide and 390x844
viewport crops with a headless renderer; they do not qualify rendered physics
or claim responsive support. No merge, auto-merge, or TURNOVER changes.
