---
issue: 15395
summary: "Generalize segment tension/compression shading into quantity-agnostic scalar series and colour scale"
---

- Add `SegmentScalarSeries`, `ScalarColorScale`, and `ScalarColorLegend` in `body_part_viz/scalar_shading.py` with sequential/diverging maps, display transforms, saturation flags, and legend ticks in physical units (#15395).
- Refactor `ForceColorScale` into a thin adapter over the scalar core so existing JSON settings and cross-runtime force goldens stay byte-identical.