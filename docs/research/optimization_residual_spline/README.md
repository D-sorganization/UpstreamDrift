# Bounded Feedforward Residual Spline

This separate research reference implements the parameterization part of
issue 15493, Slice 3. The editable calculation reference is
[residual_spline.tex](residual_spline.tex); it is separate from the engineering
design manual. No plant, contact, flight, reference trajectory, or PD gains
change here.

`scripts.research.optimization_residual_spline.sample_residual` accepts physical
sample times, distinct breakpoints, cubic control coefficients, per-joint
residual torque radii, and an explicitly declared minimum breakpoint spacing.
It returns torque and analytic slew from the same SciPy B-spline. Query times
may repeat or be unordered; output preserves their order. Inputs remain intact.

For B distinct breakpoints and J joints, coefficients have shape (B + 2, J).
Endpoint knots repeat four times and interior knots occur once. The result is
C2 across interior knots. Radii have shape (J,) in N m; zero freezes a joint's
residual. Breakpoints and minimum spacing use seconds. Slew uses N m/s.
The coefficient box bounds the residual throughout its closed support by the
B-spline convex-hull property. This is a conservative box, not a Euclidean
trust region, physiological limit, or proof of feasible total torque.

```python
import numpy as np
from scripts.research.optimization_residual_spline import sample_residual

torque, slew = sample_residual(
    times=np.linspace(0, 1, 101),
    breakpoints=np.array([0, 0.5, 1]),
    coefficients=np.zeros((5, 2)),
    torque_radius=np.array([2.0, 3.0]),
    minimum_spacing_s=0.1,
)
```

Invalid shapes, nonfinite data, too-close or unordered breakpoints,
out-of-support queries, and coefficients outside the declared box are rejected.
Proposals are never clipped or extrapolated. Nonrepresentable numerical output
is rejected. Minimum spacing is declared by the experiment; no universal
biomechanical default is asserted. Derivatives at the support endpoints are
one-sided. Extension beyond the support is not defined by this helper.

## Validation and Remaining Gates

Reproduce with:

```text
python3 -m pytest tests/unit/scripts/test_optimization_residual_spline.py -q -o addopts=''
python3 -m ruff check scripts/research/optimization_residual_spline.py tests/unit/scripts/test_optimization_residual_spline.py
python3 -m ruff format --check scripts/research/optimization_residual_spline.py tests/unit/scripts/test_optimization_residual_spline.py
```

Analytic cubic Bernstein fixtures pin torque and derivatives, including the
physical time scaling and endpoints. A finite-difference check crosses an
interior knot. Multispan samples check the bound and a frozen joint; the
continuous bound follows from nonnegative basis functions summing to one.
Malformed proposals and floating-point overflow have negative controls.
These tests use synthetic inputs, with no capture or admitted swing fixture.
They establish the parameterization only. Controller wiring, fixed-gain
sensitivity, applied-torque/slew/work audits, anatomical limits, contact and
flight convergence, and independent forward replay remain separate gates.
No physical improvement, replay acceptance, or clinical safety is claimed.
