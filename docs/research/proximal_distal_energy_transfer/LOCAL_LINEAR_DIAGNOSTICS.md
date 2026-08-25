# Local Linear Observability and Controllability Diagnostics

## Purpose and Scope

This audit supplies the first executed diagnostic downstream of the
[hybrid-system contract](HYBRID_SYSTEM_CONTRACT.md). It asks a narrow question:
at a declared state and control of the analytical double pendulum, does the
first-order continuous-time linearization have full state-observability rank
when both joint angles are measured, and full state-controllability rank when
both generalized torques are available?

The answer is conditional on the model, operating point, measurement set,
actuator set, nondimensional scaling, finite-difference steps, and SVD tolerance.
It is not structural or practical identifiability, a global nonlinear result,
participant evidence, or a coaching recommendation.

## Executable Definition

For continuous dynamics $\dot{x}=f(x,u)$ and output $y=h(x)$, the implementation
uses explicit-step central differences at $(x_0,u_0)$:

$$
A=\left.\frac{\partial f}{\partial x}\right|_{x_0,u_0},\qquad
B=\left.\frac{\partial f}{\partial u}\right|_{x_0,u_0},\qquad
C=\left.\frac{\partial h}{\partial x}\right|_{x_0}.
$$

The local linear observability and controllability matrices are

$$
\mathcal{O}=\begin{bmatrix}C\\CA\\\cdots\\CA^{n-1}\end{bmatrix},\qquad
\mathcal{C}=\begin{bmatrix}B&AB&\cdots&A^{n-1}B\end{bmatrix}.
$$

Those matrices are formed only after the declared transformation

$$
\bar A=T S_x^{-1} A S_x,\qquad
\bar B=T S_x^{-1} B S_u,\qquad
\bar C=S_y^{-1} C S_x,
$$

where $S_x$, $S_u$, and $S_y$ contain finite positive characteristic state,
control, and output scales and $T$ is a positive characteristic time. Raw
dimensional $A$, $B$, and $C$ remain in the report for traceability. Raw
continuous-time observability and controllability condition numbers are not
interpreted because their columns mix powers with different physical units.

The numerical rank threshold is
$\max(\epsilon_{\mathrm{abs}},\epsilon_{\mathrm{rel}}\sigma_1)$, with
$\epsilon_{\mathrm{abs}}=10^{-8}$ and $\epsilon_{\mathrm{rel}}=10^{-7}$.
The report retains every matrix, singular value, threshold, and retained
condition number. Rank is never inferred from a default library tolerance.

## Registered Synthetic Operating Points

The four states are sampled from the existing deterministic
`restrain_then_drive@0.100s` ODE rollout at 0%, 30%, 70%, and 100% of the
interpolated delivery-event time. They are trace-derived synthetic states, not
measured or asserted to reproduce a human swing. Each state is recomputed with
0.1x, 1x, and 10x the registered coordinate perturbation steps. Characteristic
angles use $\pi$ rad; rate and torque scales use bounded maxima from the entire
registered rollout; the nominal time is the registered delivery-event time.
Short-time/high-rate and long-time/low-rate alternatives expose scale choice.

| Operating Point               | Time (s) | Observability Rank | Smallest Retained $\sigma$ | Controllability Rank | Smallest Retained $\sigma$ | Nominal Controllability Condition Number |
| ----------------------------- | -------: | -----------------: | -------------------------: | -------------------: | -------------------------: | ---------------------------------------: |
| Initial State                 |    0.000 |                  4 |                      1.484 |                    4 |                      0.574 |                                     35.6 |
| Early Downswing               |    0.105 |                  4 |                      1.002 |                    4 |                      0.594 |                                     15.3 |
| Mid Downswing                 |    0.244 |                  4 |                      1.304 |                    4 |                      0.526 |                                     65.9 |
| Delivery Event Nearest Sample |    0.349 |                  4 |                      1.024 |                    4 |                      0.372 |                                    402.0 |

All 24 primary rank decisions are stable across the registered step
multipliers. The full-rank label does **not** mean uniform conditioning. The
delivery sample is the least well-conditioned of these four points under all
three registered scale scenarios, but its retained condition number ranges
from 92.1 to 2,119.3 across those scenarios. This scale sensitivity is itself
evidence against treating any one condition number as a physical invariant.

## Measurement and Actuator Countermodels

Both-angle, shoulder-angle-only, and wrist-relative-angle-only outputs retain
local rank four at all four registered points; the zero-output killswitch has
rank zero. Full, shoulder-only, and wrist-only generalized-torque allocations
also retain local rank four; the zero-input killswitch has rank zero. These are
properties of this coupled, fully specified analytical linearization. They do
not show that one sensor or actuator identifies parameters, supplies robust
authority, or represents a biological muscle or hand-force pathway.

## Falsification and Adverse Controls

The implementation must fail if any of the following occurs:

1. a manufactured double integrator is not observable and controllable under
   its exact position-output and input contracts;
2. the zero-input killswitch retains controllability rank or the zero-output
   killswitch retains observability rank;
3. a finite-difference step is non-finite, non-positive, or dimensionally
   incomplete;
4. equivalent physical systems expressed in different length units do not
   produce the same dimensionless matrices and singular values;
5. a registered operating-point rank changes across the 0.1x--10x step range
   without being retained as an adverse outcome; or
6. a report promotes local rank to structural identifiability, practical
   identifiability, or a global nonlinear conclusion.

These controls test the diagnostic implementation and local numerical
stability. They do not validate the double-pendulum model against human data.

## Reproduction

```powershell
python -m scripts.research.proximal_distal_energy.run_local_linear_diagnostics validate
python -m pytest -q tests/research/test_local_linear_diagnostics.py tests/research/test_local_linear_diagnostics_evidence.py
```

The machine-readable authority is
[`data/local_linear_diagnostics.json`](data/local_linear_diagnostics.json).
Subsequent #9027 slices must add structural and practical identifiability,
constraint-singularity and internal-force diagnostics, event sensitivity, and
finite-time stability without treating this local full-rank result as their
substitute.
