# Sampled Applied Effort Audit

Epic #15493, Slice 3 prerequisite. The engine-neutral Python API is
`src.shared.python.optimization.applied_effort_audit.audit_applied_effort`.
It consumes measured total applied torques after feedforward, PD feedback and
saturation, in a caller-declared consistent actuator order and sign frame.
Only rotational actuator torques in Nm are supported; times are seconds.

The standalone calculation reference is [reference.tex](reference.tex).
For each actuator, the receipt reports the minimum signed torque margin and
the slew limit minus the maximum absolute adjacent-sample secant rate. Bounds
are inclusive and can be asymmetric. No implicit tolerance is applied.

This is a sampled diagnostic, not full biomechanical feasibility or a wired
optimization loop. It cannot certify between-sample extrema, ZOH jump rates,
continuous derivatives, mechanical work, ROM, tracking, or flight acceptance.
Existing work accounting remains in `motion_matching.cost.compute_total_work`.
No duplicate physics or dynamics solver is introduced.

Reproduce with Python 3.11+ and the pinned Tools submodule initialized:

```powershell
python3 -m pytest tests/unit/optimization/test_applied_effort_audit.py -q --no-cov -o addopts=''
python3 -m ruff check src/shared/python/optimization/applied_effort_audit.py tests/unit/optimization/test_applied_effort_audit.py
python3 -m ruff format --check src/shared/python/optimization/applied_effort_audit.py tests/unit/optimization/test_applied_effort_audit.py
```

Tests first failed with the new module missing, after dependency initialization.
Analytical fixtures verify nonuniform steps, signed margins, feedback-driven
violations, exact limits, zero slew, malformed data and overflow rejection.
These are arithmetic contract checks; no matched-swing experiment was run.
The LaTeX source has not been compiled or visually reviewed.
