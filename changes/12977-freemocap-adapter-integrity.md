---
issue: 12977
summary: "pipeline: FreeMoCap adapter deterministic CSV selection, NaN confidence handling, non-monotonic time rejection, and configurable min_confidence"
branch: "fix/12977-freemocap-adapter-integrity"
---

Hardened `FreeMoCapOutputAdapter` against silent data corruption and ambiguities: sorted matching landmark CSVs deterministically, raised `ValueError` when multiple matching CSVs exist unless an explicit `csv_path` is specified, treated `NaN` and non-finite confidences or coordinates as invisible (`visible=False`), enforced strictly monotonic timestamps across frames, made `min_confidence` configurable (validated in `[0.0, 1.0]`), and recorded `min_confidence` into session metadata.
