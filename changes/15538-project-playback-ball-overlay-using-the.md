---
issue: 15538
summary: "Project playback ball overlay using the view look-at (#15538)"
dl_state: "in_review"
next_step: "Review the PR; operator handles merge authorization."
branch: "codex/15538-overlay-lookat"
---

- Thread the active view `lookat_m` through `BallOverlay.paint` for aim-line and flight projection via `pinhole_for_view` and `bare.draw` (#15538).
- Add shifted-camera regression coverage in `tests/tools/model_explorer/test_playback_ball_overlay.py` (#15538).
