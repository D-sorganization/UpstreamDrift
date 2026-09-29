# Double-Pendulum Parameter Identifiability

## Purpose and Evidential Boundary

This audit separates two questions that are often conflated:

1. can a declared motion record excite every coefficient that appears in the
   analytical inverse dynamics; and
2. can those coefficients uniquely recover the underlying physical parameter
   vector?

The first is a finite-record rank and conditioning question. The second is a
structural property of the physical-to-coefficient map. A full-rank trajectory
regressor cannot repair an exact many-to-one parameter map.

The eleven physical entries below are a reduced equation-of-motion
parameterization. They are not a claim that every entry is independently
editable in `GolfModelParams`, nor that a participant's anatomy or equipment
can be recovered from the registered synthetic trace.

## Exact Inverse-Dynamics Factorization

For relative wrist angle $q_2$, generalized velocity $\dot q$, and acceleration
$\ddot q$, the analytical double pendulum can be written

$$
\tau=Y(q,\dot q,\ddot q)\,\theta_b,
$$

with seven base coefficients

$$
\theta_b=
[\alpha,\beta,\delta,\gamma_1,\gamma_2,d_1,d_2]^\mathsf{T},
$$

where

$$
\begin{aligned}
\alpha &= I_1+I_2+m_2l_1^2, &
\beta &= m_2l_1r_2, &
\delta &= I_2,\\
\gamma_1 &= (m_1r_1+m_2l_1)g\cos\phi, &
\gamma_2 &= m_2r_2g\cos\phi.
\end{aligned}
$$

The implementation reconstructs the canonical ODE backend's mass-matrix,
Coriolis/centripetal, gravity, and damping terms to $10^{-12}$ relative and
absolute tolerance for registered adverse states. This is an equation-identity
test, not a fit to generated output.

## Structural Non-Uniqueness

The Jacobian from the eleven reduced physical entries

$$
[m_1,r_1,I_1,m_2,l_1,r_2,I_2,g,\phi,d_1,d_2]
$$

to the seven base coefficients has rank 7 and nullity 4 throughout the declared
positive parameter domain. The proof uses the seven columns associated with
$I_1$, $l_1$, $I_2$, $m_1$, $r_2$, $d_1$, and $d_2$. The magnitude of that
$7\times7$ minor is

$$
m_2^2 r_1 r_2\,[g\cos(\phi)]^2,
$$

which is strictly positive under the registered domain. This analytic witness,
not a condition number computed from columns with incompatible units, is the
structural-rank authority. Three finite, exact counterexample families are
retained rather than relying on rank alone:

- for any admissible $\lambda>0$,
  $m_1'=\lambda m_1$ and $r_1'=r_1/\lambda$ preserve the upper first
  moment;
- for any admissible $|\phi'|<\pi/2$,
  $g'=g\cos\phi/\cos\phi'$ preserves the projected gravity; and
- for $\lambda$ sufficiently close to one to retain positive physical
  entries,

  $$
  \begin{aligned}
  m_2'&=\lambda m_2, & r_2'&=r_2/\lambda,\\
  I_1'&=I_1-(\lambda-1)m_2l_1^2, &
  r_1'&=r_1-(\lambda-1)m_2l_1/m_1
  \end{aligned}
  $$

  jointly preserve the coupling inertia, proximal inertia, and both gravity
  coefficients.

The registered alternatives select one admissible member of each family. Each
is distinct from the default and changes no base coefficient at machine
precision. These invariances prove that the declared physical vector is
structurally non-identifiable under this model, even with an arbitrarily long
noiseless record governed by the same equations.

## Nondimensional Finite-Record Result

The existing `restrain_then_drive@0.100s` synthetic ODE rollout contributes
350 samples from 0 to 0.349 s, ending at the delivery-nearest sample. Rank is
never assigned to its dimensional regressor. The coefficient coordinates use
the absolute registered base-coefficient values, the torque coordinate uses
the largest registered generalized torque with a 1 N m floor, and

$$
\bar Y = Y\,\operatorname{diag}(s_\theta)/s_\tau.
$$

With the explicit mixed threshold
$\max(10^{-8},10^{-7}\bar\sigma_1)$, the stacked $700\times7$ dimensionless
regressor has rank 7 and retained condition number 180.853. The raw dimensional
condition number is retained only implicitly through the source matrix and is
not interpreted.

Full rank is not uniform information quality. The first 10% of the record also
has rank 7 but dimensionless condition number 167,192; the 30%, 70%, and 100%
cumulative records have condition numbers 562.943, 304.534, and 180.853.
Consequently, a binary rank label must not be reported as precise parameter
recovery. Noise models, repeated trials, priors, profile likelihoods, and
governed observations remain required for practical identifiability.

An equivalent-unit fixture rescales every coefficient coordinate and the
corresponding regressor column. It changes the dimensionless matrix by no more
than $4.45\times10^{-16}$ and preserves rank 7. Two additional positive scale
choices also preserve rank 7 while changing retained condition number. This is
the intended result: rank is stable for these declared choices, while condition
number remains scale-dependent and must always be reported with its scale
contract.

## Noise-Aware Best-Case Bound

A conditional Gaussian Fisher-information screen now asks what the same
regressor would imply if position, velocity, acceleration, model form, and
event alignment were exact and the only error were known, independent,
homoscedastic generalized-torque noise. These deliberately favorable
assumptions make the result a lower bound on coefficient uncertainty rather
than a practical-identifiability result.

For the full registered record, assumed torque-noise standard deviations of
0.1, 0.5, 1.0, and 2.0 N m produce worst coefficient-relative 95% half-widths
of 1.23%, 6.16%, 12.3%, and 24.7%, respectively. At 1.0 N m, the worst bound
belongs to the distal gravity coefficient and the largest absolute coefficient
correlation is 0.937. The first 10% window is much more adverse: despite rank
seven, its worst relative half-width is approximately 499 times the coefficient
magnitude. The corresponding 30%, 70%, and 100% values are approximately
0.750, 0.306, and 0.123 times the coefficient magnitude.

This screen falsifies the proposition that full numerical rank alone implies
useful finite-noise recovery. It does not include kinematic differentiation
noise, correlated sensor errors, event-time uncertainty, model discrepancy,
unknown noise scale, priors, repeated trials, participant variability, or
held-out prediction. Those omissions can only weaken the practical inference;
they cannot promote this best-case bound into participant parameter evidence.

## Falsifiers and Killswitch

The evidence fails closed if:

1. the analytical regressor does not reconstruct canonical inverse dynamics;
2. any exact counterexample changes a base coefficient;
3. the physical-map rank/nullity differs from 7/4 at the registered default;
4. the zero-motion regressor retains nonzero rank; or
5. the oracle-kinematics uncertainty lower bound is promoted to participant or
   practical-identifiability evidence;
6. an equivalent coefficient-unit conversion changes the dimensionless rank;
   or
7. finite-record rank is promoted to model-adequacy, mechanism, or coaching
   evidence.

The zero-position, zero-velocity, zero-acceleration killswitch has rank zero.
The machine-readable authority is
[`data/double_pendulum_identifiability.json`](data/double_pendulum_identifiability.json).

## Reproduction

```powershell
python -m scripts.research.proximal_distal_energy.run_double_pendulum_identifiability validate
python -m pytest -n 0 -q tests/research/test_double_pendulum_identifiability.py tests/research/test_double_pendulum_identifiability_evidence.py
```

Subsequent #9027 work must add measurement-chain and model-discrepancy-aware
practical identifiability, consolidate the existing planar closed-loop and
bilateral point-force nullspaces into tier-by-tier constraint/internal-force
reports, add singular-event and event-time sensitivities, and produce comparable
reports for eligible higher model tiers without silently transferring this
analytical result to them.
