# test(body-part-viz): comprehensive TDD coverage + golden snapshots

Depends on every other child issue.

## Why

Cross-cutting test pass: enforce ≥ 80% line / 70% branch coverage on the new `body_part_viz` package, plus golden image snapshots for the renderers.

## What

### Coverage target

```bash
python3 -m pytest tests/unit/body_part_viz/ tests/integration/body_part_viz/ \
  --cov=src/shared/python/body_part_viz --cov-branch --cov-report=term-missing
```

≥ 80% line, ≥ 70% branch.

### Golden image snapshots

`tests/integration/body_part_viz/test_renderer_snapshots.py`:

- Use matplotlib `savefig` to PNG at fixed DPI (100).
- Compare against committed reference PNGs in `tests/fixtures/body_part_viz/`.
- Tolerance: ~0.5% RMS pixel diff.

Snapshots:

- 3 default segments (line, cylinder, mesh) at frame 0.
- Same 3 segments at frame 50.
- Library-shape full body at address frame.

### Performance regression

`tests/integration/body_part_viz/test_perf_budget.py`:

- 26 cylinders × 16 facets × 654 frames; measure mean per-frame update time.
- Assert ≤ 16 ms (≥ 60 fps).
- 26 library meshes × ~200 verts each × 654 frames; assert ≤ 33 ms (≥ 30 fps).

### Cross-tool integration

`tests/integration/body_part_viz/test_cross_tool.py`:

- Build a SegmentVizSet, save JSON, load in C3D Viewer (offscreen), assert renderer instantiates correctly.
- Same set fed to `LiveViewController`; assert.
- Same set fed to URDF generator; assert URDF parses.

## Acceptance criteria

- [ ] Coverage targets met.
- [ ] Golden snapshots committed; CI uses Agg backend so they're stable.
- [ ] Performance regression test passing locally and in CI.
- [ ] Cross-tool integration test passing.

## Files touched

- New: `tests/integration/body_part_viz/test_renderer_snapshots.py`
- New: `tests/integration/body_part_viz/test_perf_budget.py`
- New: `tests/integration/body_part_viz/test_cross_tool.py`
- New: `tests/fixtures/body_part_viz/*.png` (committed; ~12 small PNGs)
