---
issue: 15396
summary: "Segment kinetic-energy surface shading (cividis, fixed limits); demote KE globes to KE proxy debug overlay"
---

- Shade segment surfaces by stored kinetic energy (J) via shared `ScalarColorScale` / cividis with reference-swing fixed limits; MuJoCo scene and MeshCat bindings; visualization tab toggle and per-engine persistence (#15396).
- Demote KE globes to off-by-default "KE proxy" debug overlay without glow or translucency; align `globe_rgba_for_ke` zero/non-finite samples with colormap zero.