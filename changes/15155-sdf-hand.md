---
issue: 15155
summary: "Sculpt implicit SDF palm/metacarpals with cupped grip arc, bridge thumb web, smooth Gouraud shading, and lateral-oblique evidence views."
dl_state: "in_review"
next_step: "Integrate shared posing.py (#15164) and render.py (#15154) when merged; minor wrist lobe and PIP crease polish in sdf_hand.json before Dieter sign-off."
title: "SDF Hand Sculpting, Controls, and Smooth Shading"
branch: "cursor/sdf-sculpted-hand-c70f"
---

- Cycle 2 (2026-11-15, Refs #15150): Added cupped metacarpal circumferential arc (`palm_wrap_arc_rad`, `palm_base_hover_m`, `palm_axial_drop_m`), bridged `thumb_web` across `thumb_mid -> bases["index"]`, tuned `palm_fill_radii_m` / `wrist_radii_m` / `crease_*` settings to close the web hole and remove palm saucer flare/wrist horn, and updated `_camera_for` oblique framing with `thumb_dir`; `29 passed` across `test_sdf_hand*.py` and 27 `1024 px` PNGs regenerated.
- Adopts the shared ADR-0057 hand_backends registry: `sdf_hand` registers through `register()` (only when the `sdf-hand` scikit-image extra is installed) and builds the shared `HandVisualResult`; the rich builder is now `build_sdf_hand`.
