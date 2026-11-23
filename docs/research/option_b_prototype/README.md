# Fixed-Gain Option B Prototype

This research checkpoint implements the algebra and coefficient-space trust
check in `src/shared/python/impact_optimization/option_b_prototype.py`. The
editable [calculation reference](option_b_prototype.tex) is a research record,
not the engineering design manual. It does not implement an optimizer, a
solution bundle, biomechanical feasibility, impact, or a qualified swing.

Run the focused check from the repository root with Python 3.12, the pinned
Tools submodule initialized, and `QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg`:

```text
python -m pytest -q tests/unit/impact_optimization/test_option_b_prototype.py
```

The controller is sampled at times chosen by the caller. Its reported
`applied_nm` is ideal per-channel clipping of the **total** command, before
any downstream root masking or model-specific actuator transformation. It is
not evidence of actual applied torque. Gain selection, state channel mapping,
reference rate consistency, hold timing, and saturation parameters remain
caller responsibilities. Saturation flags do not constitute a feasibility
gate. Reusing the shared integrator in a synthetic unit-mass test establishes
only its local input/hold connection, not replayability or human performance.
To represent an explicitly unactuated coordinate, set its lower and upper
limits to zero; a nonzero command is then reported as saturated and clipped
to zero before the plant. Equal nonzero lower and upper limits are rejected.
