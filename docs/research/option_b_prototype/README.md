# Fixed-Gain Option B Prototype

This research checkpoint implements the algebra and coefficient-space trust
check in `src/shared/python/impact_optimization/option_b_prototype.py`. The
editable [calculation reference](option_b_prototype.tex) is a research record,
not the engineering design manual. It does not implement an optimizer, a
solution bundle, biomechanical feasibility, impact, or a qualified swing.

Run the focused check from the repository root with Python 3.12, the pinned
Tools submodule initialized, and `QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg`:

```text
python -m pytest -q tests/unit/impact_optimization/test_option_b_prototype.py tests/unit/impact_optimization/test_total_applied_zoh_audit.py
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

The separate `total_applied_zoh_audit.py` checks caller-declared total applied
rotational generalized input, not residuals or commands. It takes each
left-held torque and independently supplied start/end velocities, splitting
positive and negative work at per-joint zero crossings under a linear-velocity
interpolation assumption. It checks declared per-channel torque, finite
inter-update difference rate, quadratic effort and signed/positive work
limits without clipping inputs. The inter-update quantity is not continuous
slew: ZOH inputs jump, and the initial transition is unobserved. A single hold
with no observed update is indeterminate. Even a pass covers only the supplied
schedule and this quadrature model. Provenance, units, coordinate mapping and
limits are caller assertions; between-hold motion, actual actuator forces and
anatomical or physical feasibility are not certified.
Separately labeled `ideal_direct_observed` evidence supports exploratory
accounting of an observed ideal direct generalized input without claiming an
actuator model. It is assumption-limited: a no-violation result remains
indeterminate, while violations apply only to the declared model limits.
Work, effort and update-rate limits may be explicitly unavailable; metrics are
still reported, but missing limits prevent a no-violation result from passing.
