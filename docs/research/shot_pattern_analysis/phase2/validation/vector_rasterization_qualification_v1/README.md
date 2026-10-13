# Qualified Vector Rasterization Helper

The canonical `scripts.research.shot_pattern_analysis.vector_rasterization.render_vector` reuses the qualified inspector and requires explicit Poppler/Node/Sharp paths. It preserves native stdout/stderr/exit codes, input hashes, output identity, physical viewport and decoded pixel dimensions in `raster_diagnostics.json`. It never grants visual approval. SVG rendering uses a proportional fit inside the declared physical viewport at 144 pixels/inch; equal configured density does not establish PDF/SVG pixel equivalence.

## Tests and Integration

AGY supplied a 13-test candidate. Astra reproduced four independent defects (three configuration/evidence failures and PNG replacement between decode and hashing), repaired them test-first, and passed 18 tests plus native PDF/SVG rendering. Root integrated the helper, narrowed two broad exception catches required by repository Ruff policy, and reran all 18 tests and repository-config Ruff, formatting and Mypy successfully. Exact raw evidence is archived by author; the initial repository lint failure is retained. Test imports were adapted to the canonical module; optional dependencies and native tools skip explicitly when unavailable. POSIX shell fault fixtures are marked accordingly.

Set `P2_VECTOR_TOOLS` to a JSON object containing absolute `pdftoppm`, `node` and `sharp` paths, then run:

```bash
python -m pytest -o addopts='' -q tests/research/test_shot_pattern_vector_rasterization.py
python -m ruff check scripts/research/shot_pattern_analysis/vector_rasterization.py tests/research/test_shot_pattern_vector_rasterization.py
python -m mypy --python-version 3.12 --follow-imports=silent scripts/research/shot_pattern_analysis/vector_rasterization.py tests/research/test_shot_pattern_vector_rasterization.py
```

All four findings are recorded through AGY on [Issue #12160](https://github.com/D-sorganization/UpstreamDrift/issues/12160#issuecomment-6100685375), with an independent exact-body check. Component acceptance does not establish full figure collection, scientific, visual or publication acceptance. Producer sources and numerical outputs remain unchanged.
