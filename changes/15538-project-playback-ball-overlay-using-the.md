---
issue: 15538
summary: "Project playback ball overlay using the view look-at (#15538)"
dl_state: "in_review"
next_step: "Independent acceptance at the new head; production wiring remains tracked by open #15538."
branch: "codex/15538-overlay-lookat"
---

- Thread the active view `lookat_m` through `BallOverlay.paint` for aim-line and flight projection via `pinhole_for_view` and `bare.draw` (#15538).
- Add shifted-camera regression coverage in `tests/tools/model_explorer/test_playback_ball_overlay.py` (#15538).
- Assert shifted-camera output differs from the default in both views; remove the unrelated straight-roll assert (#15542).
