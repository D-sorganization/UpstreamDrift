# Trajectory-Varying Event-Conditioned Control Authority

## Purpose and Scope

This study asks whether the two declared torque channels can create local state
variation along the registered analytical downswing and at its geometric
delivery event. It does not ask which controller is best, whether a bounded
human actuator can realize a perturbation, or which movement strategy should be
taught.

## Exact Discrete Linearization

At every sample, both Jacobians differentiate the same registered RK4 step:

<!-- prettier-ignore -->
\[
\delta x_{k+1}=A_k\delta x_k+B_k\delta u_k.
\]

State perturbations are divided by the declared state scales. Torque
perturbations use equal 100 N m characteristic scales. For comparisons across
time steps, the discrete input matrix is divided by \(\sqrt{\Delta t_k}\).
This makes the quadratic discrete input norm equivalent to the energy of a
piecewise-constant continuous input. Per-sample matrices are retained
separately; conclusions from the two normalizations are not interchanged.

The trajectory-varying Gramian follows

<!-- prettier-ignore -->
\[
W_{k+1}=A_kW_kA_k^\mathsf T+B_kB_k^\mathsf T,\qquad W_0=0.
\]

Both channels, shoulder only, wrist only, and zero input are evaluated from the
same state trajectory. The full two-channel history equals the sum of the two
single-channel histories within \(5.2\times10^{-12}\), while the zero-input
history remains exactly zero.

## Event Conditioning

The unique positive crossing of
\(h(x)=\theta_s+\theta_w=0\) is refined inside its sampled bracket by repeatedly
executing the exact RK4 step. The final guard residual is
\(1.3\times10^{-13}\), and the transversality denominator is 35.0258 s\(^{-1}\).
The refined time differs from the earlier linearly interpolated event time by
\(1.6\times10^{-7}\) s.

Arrival variation is projected onto the event surface with

\[
P=I-\frac{fn^\mathsf T}{n^\mathsf Tf}.
\]

An explicit orthonormal basis \(Q\in\mathbb R^{4\times3}\) spans the guard
tangent space, and the reported event Gramian is
\(Q^\mathsf TPWP^\mathsf TQ\). Its dimension and local numerical rank are both
three. The missing guard-normal direction is a geometric consequence of
conditioning on event arrival; it is not labeled as actuator rank loss.

## Falsification Controls

Six nonlinear pulse rollouts, spanning three phases and both torque channels,
agree with propagated input sensitivities within \(4.2\times10^{-8}\). Halving
and doubling the input finite-difference step changes the event Gramian by less
than \(4.2\times10^{-10}\). Halving and doubling the integration step changes
it by less than \(1.3\times10^{-7}\) under the continuous-energy-equivalent
normalization. Equivalent degree and N mm coordinates preserve the scaled
matrices within \(3.4\times10^{-16}\).

Four matched phase windows compare the trajectory-varying calculation with a
frozen-local countermodel. Their relative Gramian differences range from 0.136
to 0.298. A frozen operating-point Gramian is therefore not substituted for
the registered trajectory-varying result.

## Interpretation Boundary

The full-state finite-window Gramian has local numerical rank four; the
event-tangent Gramian has rank three. These are scale- and tolerance-qualified
properties of one synthetic analytical trajectory. They do not establish
global nonlinear reachability, minimum required torque, bounded-control
feasibility, controller superiority, human strength, neural timing, passive
late-downswing torque, robustness to fatigue or noise, or a coaching
recommendation. Human interpretation remains unavailable until governed
participant kinematics and synchronized bilateral load data pass the declared
held-out tests.

## Reproduction

```powershell
python -m scripts.research.proximal_distal_energy.run_trajectory_control_authority validate
python -m scripts.research.proximal_distal_energy.make_trajectory_control_authority_figure
python -m pytest -n 0 -q tests/research/test_trajectory_control_authority.py tests/research/test_trajectory_control_authority_evidence.py
```

Portable summaries are in `data/trajectory_control_authority.json`; the exact
step matrices, histories, tangent basis, countermodels, refinement arrays, and
direct-pulse responses are in `data/trajectory_control_authority.npz`.
