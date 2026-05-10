# docs(body-part-viz): ADR + user guide + asset-author guide

Depends on every other child issue (lands last).

## What

### ADR

`docs/adr/00<next>-body-part-viz-toolkit.md` covering:
- Context: motion-matching pipeline already exists; segments rendered as lines or cylinders only.
- Decision: ship a shared `body_part_viz` package with shapes / fitters / renderers contracts.
- Alternatives considered:
  1. Extend C3D Viewer's segment-tab geometry directly — rejected; not reusable.
  2. Per-tool implementation — rejected; violates DRY.
- Consequences: cross-tool consistency; mesh import; URDF visuals share the same source of truth.

### User guide

`docs/user_guide/body_part_viz/` with three pages:

1. `quickstart.md`: load a C3D, open Segments tab, swap a cylinder for a library shape, save.
2. `mesh_import.md`: how to bring in a custom STL / OBJ; sizing; rest-pose fitting.
3. `asset_author_guide.md`: how to add a new shape to the default library; the manifest; the procedural-generation script.

### API reference

`docs/api/body_part_viz.md` auto-generated from docstrings via the existing pdoc / sphinx infrastructure (whichever is in `pyproject.toml`).

## Acceptance criteria

- [ ] ADR with the next sequential number; mirrors `docs/adr/0006-multi-source-motion-targets.md` style.
- [ ] User-guide pages renderable as Markdown.
- [ ] API reference generated.
- [ ] AGENTS.md updated to point at `body_part_viz` as the canonical shape stack.

## Files touched

- New: `docs/adr/00<next>-body-part-viz-toolkit.md`
- New: `docs/user_guide/body_part_viz/{quickstart,mesh_import,asset_author_guide}.md`
- Edit: `AGENTS.md`
- Edit: `docs/motion_matching/README.md` (link to the new pages)
