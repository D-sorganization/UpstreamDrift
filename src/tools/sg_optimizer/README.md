# sg_optimizer — Strokes Gained Optimizer (launcher tile)

Personalized golf strategy optimizer. Epic [#6269](https://github.com/D-sorganization/UpstreamDrift/issues/6269).

## Phase 1 (this PR)

Headless library + CLI only. No Qt imports.

```bash
python -m src.tools.sg_optimizer \
  --profile examples/sg_optimizer/profiles/scratch.yaml \
  --baseline data/sg_optimizer/baselines/pga_tour.yaml \
  --hole-spec examples/sg_optimizer/holes/par4_right_water.py \
  --conditions tournament
```

Outputs JSON with the optimal-from-tee action and expected strokes.

## Architecture

The shared library lives under [`src/shared/python/sg_optimizer/`](../../shared/python/sg_optimizer/). This tile is the future Phase-3 UI host — no business logic belongs here.

## Roadmap

- Phase 2 (#6271) — GeoJSON I/O + classic-holes library + full `TreeModel`.
- Phase 3 (#6272) — PyQt6 profile editor + conditions panel.
- Phase 4 (#6273) — Library browser + map widget + tracing.
- Phase 5 (#6274) — Strategy view + GPS real-time (stretch).
- Phase 6 (#6275) — AffineDrift integration (research).
