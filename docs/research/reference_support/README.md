# Offline Reference Support at Joins

`motion_matching.reference_support` reports a conservative set of input-row
indices that may contribute to the existing offline reference position,
velocity and acceleration samples at one query time. Its opt-in
`require_past_only()` rejects when any such row is later than the query.
It does not change the existing controller or assert that the offline
reference was intended to be causal.

GCV-20 explicitly splits **rate** differentiation and rate interpolation at
impact to avoid smearing a velocity step. Position interpolation remains
unsplit in the existing controller, and can therefore use the first
post-impact row before the impact time. This is a documented property of
that implementation, not by itself evidence of a defect. A seam analysis
that requires past-only inputs must reject or account for the dependence.

The support is a structural overapproximation; zero weights or cancellations
may make a listed row numerically irrelevant. Passing the guard does not
establish causality of earlier filtering, target generation, a reference hook,
or other controller inputs. It supplies no numerical tracking, ROM, torque,
or physical-feasibility evidence. See the LaTeX reference for stencils and
run `python -m pytest -q tests/unit/motion_matching/test_reference_support_15493.py`.
