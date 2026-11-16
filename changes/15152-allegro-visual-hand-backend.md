---
issue: 15152
summary: "Wonik Allegro 4-finger visual hand backend scaled to HandSize, curled to grip contact, with warm pad theme, both-hand fill, and 27-view roll-audited Gouraud evidence under #15150."
dl_state: "in_review"
next_step: "Cross-track integration once shared posing/render PRs land (#15154, #15164); little finger absent and flat palm block remain hardware CAD limits."
branch: "cursor/allegro-hand-backend-7b4a"
---

- Cycle 2 (2026-11-15, Refs #15150): Harmonized `tip_pad_rgba` (`[0.85, 0.69, 0.58, 1.0]`), added `both_frame_fill = 0.75` (`allegro_settings.py`, `hand_backend_allegro.json`), extracted `build_evidence_receipt(..., include_roll_audit=True)` with `pad_gaps_by_roll_m` across `-20°/0°/+20°` roll, added `test_allegro_cycle2_receipt_and_theme.py` (`24 passed, 625 deselected`), and regenerated 27 `1024 px` PNGs + `receipt.json`.
- Adopts the shared ADR-0057 hand_backends registry: Allegro registers through `register()`, returns the shared `HandVisualResult`, maps canonical `HandJointAngles` hinges to `ffj1`-style joints, and drops its private `types`/`registry` copies (raw mesh builder is now `build_allegro_hand_mesh`).
