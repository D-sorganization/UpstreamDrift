# Hand-Path Force Attribution: Source, Terminology, and Estimand Contract

## Purpose and Scope

This contract defines the quantities and claim boundaries for decomposing
force along the golf-club hand path into drift and control contributions. It
applies across the planar double-pendulum, one-arm, and closed-loop two-arm
model tiers. Its purpose is to make results comparable without treating a
measured or simulated contact force as a direct measure of muscular effort.

The contract governs pointwise matched-state analyses. A forward
torque-killswitch trajectory is a different experiment and must be labeled as
such because its state diverges from the observed trajectory after the switch.

## Primary Evidence and Exact Claim Boundary

The primary hand-path source is MacKenzie, McCourt, and Champoux (2020),
[_How Amateur Golfers Deliver Energy to the
Driver_](https://www.golfsciencejournal.org/article/12640-how-amateur-golfers-deliver-energy-to-the-driver)
[@mackenzie2020energy]. The study analyzed drives from 76 right-handed amateur
golfers with handicaps below 15. It represented the golfer's action on the club
as a net force applied at the mid-grip point and a net couple, inferred by
inverse dynamics.

Within that sample, average force along the hand path correlated with
clubhead speed at $r=0.96$ and predicted 92% of its between-player variance.
Linear work predicted 90% of the variance, angular work added 9%, and
gravitational work added no predictive ability. The reported sample means were
129 N for average hand-path force, 174 J for linear work, 39 J for angular
work, and 1.35 m for hand-path length.

Those results establish an association and a work-energy accounting identity.
They do not establish that hand-path force is muscular effort, that increasing
voluntary exertion necessarily increases clubhead speed, or that any stated
fraction of the force arose from drift. The authors explicitly identify
coordination, exertion, and force-generating capacity as possible contributors
without identifying their separate effects. Drift-control attribution is a new
model-based analysis motivated by, but not reported in, that study.

## Canonical Signs, Frames, and Samples

Every result must declare:

- the body on which each force or wrench acts;
- the coordinate frame, reference point, units, and positive directions;
- the hand-path point, which defaults to the mid-grip point $H$ for comparison
  with MacKenzie et al.;
- the analyzed event window and its start and end events; and
- the model tier, parameter set, solver, timestep, and source-data provenance.

Let ${\mathbf v}_H(t)$ be the velocity of $H$. For samples satisfying
$\lVert{\mathbf v}_H\rVert>v_\epsilon$, define the instantaneous path tangent

$$
{\mathbf e}_t(t)=\frac{{\mathbf v}_H(t)}{\lVert{\mathbf v}_H(t)\rVert}.
$$

The threshold $v_\epsilon$ must be reported. Samples below it are marked
undefined for directional projections; they are not silently set to zero.

## MacKenzie-Compatible Hand-Path Estimands

For the net golfer-on-club force ${\mathbf F}_H$, the signed force along the
hand path is

$$
F_{\parallel}(t)={\mathbf F}_H(t)\mathbin{\cdot}{\mathbf e}_t(t).
$$

For an analysis window $[t_a,t_b]$, define

$$
L_H=\int_{t_a}^{t_b}\lVert{\mathbf v}_H\rVert\,dt,
\qquad
W_{H,F}=\int_{t_a}^{t_b}{\mathbf F}_H\mathbin{\cdot}{\mathbf v}_H\,dt,
$$

and the path-averaged force

$$
\overline F_{\parallel}=\frac{W_{H,F}}{L_H}
=\frac{\int_{t_a}^{t_b}F_{\parallel}\lVert{\mathbf v}_H\rVert\,dt}{L_H}.
$$

This path-weighted, signed estimand is the quantity compatible with the 2020
study. A time average of $F_{\parallel}$, an average force magnitude, peak
force, and the sum of individual-hand magnitudes are different estimands and
must not reuse its name or evidence claim.

## Drift, Control, ZTCF, and ZVCF

For control-affine generalized dynamics evaluated at the same state,

$$
M(q)\ddot q+h(q,\dot q)=B(q)u,
$$

the pointwise acceleration split is

$$
\ddot q=\underbrace{-M^{-1}h}_{\ddot q_{\mathrm{drift}}}
+\underbrace{M^{-1}Bu}_{\ddot q_{\mathrm{control}}}.
$$

A **pointwise ZTCF sample** is the zero-applied-control acceleration and
reaction evaluation at one achieved state. Repeating that evaluation along the
achieved history produces a **stitched pointwise ZTCF trace**; it is not one
forward trajectory. The associated acceleration is the drift vector at each
sample. Its declared inventory may include gravity, Coriolis and centrifugal
effects, passive stiffness, damping, and prescribed-base effects. The inventory
must be fixed before comparing models.

A **forward** or **branched ZTCF trajectory** instead integrates the
zero-applied-control equations from one declared initial state. It shares the
achieved state only at the branch time and can support persistence questions
that the stitched pointwise trace cannot. Every use of _ZTCF_ in the article,
figures, data, and software is qualified as pointwise, stitched, forward, or
branched on first use.

The zero-velocity counterfactual (ZVCF) is an instantaneous evaluation at the
fixed configuration/internal state with $\dot q=0$ and $u=0$. Passive and
other autonomous plant terms remain. A computation that sets velocity to zero
while preserving $u$ is a **zero-velocity control-preserved evaluation**, not
ZVCF. Every figure, table, and dataset must distinguish these quantities.

For constrained or closed-loop models, accelerations and constraint reactions
must come from the same constrained solve. A reaction split
${\boldsymbol\lambda}={\boldsymbol\lambda}_{\mathrm{drift}}+
{\boldsymbol\lambda}_{\mathrm{control}}$ is admissible only at the same state,
with the same contact mode, constraint stabilization, passive parameters, and
prescribed motion. Reconstruction residuals must be reported. Subtracting two
diverged trajectories does not establish this pointwise identity.

If the net hand force has a verified split
${\mathbf F}_H={\mathbf F}_{H,d}+{\mathbf F}_{H,c}$, then

$$
F_{\parallel}=F_{\parallel,d}+F_{\parallel,c},
\qquad
P_{H,F}=P_{H,F,d}+P_{H,F,c},
$$

where every projection and power uses the actual matched-state hand tangent
and velocity. A component evaluated with its own counterfactual velocity is not
additive with the measured-state power.

## Impulse, Power, and Work

The following quantities are distinct and must be stored separately:

$$
{\mathbf J}_H=\int {\mathbf F}_H\,dt,
\qquad
J_{\parallel}=\int F_{\parallel}\,dt,
$$

$$
P_{H,F}={\mathbf F}_H\mathbin{\cdot}{\mathbf v}_H,
\qquad
W_{H,F}=\int P_{H,F}\,dt.
$$

Because the hand-path tangent rotates, $J_{\parallel}$ is not the magnitude or
projection of ${\mathbf J}_H$. Impulse is not an energy proxy. Work must be
calculated from force-velocity power or the equivalent force-displacement
integral.

For a wrench $({\mathbf F},{\mathbf M}_O)$ and twist
$({\mathbf v}_O,{\boldsymbol\omega})$ at the same reference point $O$,

$$
P_O={\mathbf F}\mathbin{\cdot}{\mathbf v}_O
+{\mathbf M}_O\mathbin{\cdot}{\boldsymbol\omega}.
$$

Force power, couple power, and total wrench power are all reported. Reference
transport must transform the wrench and twist together and preserve total
power within tolerance.

## Joint and Interface Estimands

The hand-path projection is specific to the declared grip point. At shoulders,
elbows, wrists, and other interfaces, the canonical output is the full
distal-body wrench, the compatible interface twist, and their power. A joint's
generalized torque and generalized speed may also be reported, but they are not
renamed as force along the hand path.

Each event window reports drift, control, and total values for:

- signed force and moment components;
- vector and selected scalar impulses;
- force power, couple power, and total power; and
- positive, negative, and net work.

Signed fractions may be negative or exceed 100% when drift and control oppose
one another. Ratios with a denominator below a declared tolerance are reported
as undefined. Results must also include the cancellation-stable share
$|X_d|/(|X_d|+|X_c|)$ and the reconstruction residual; raw signed quantities
remain authoritative.

## Two-Hand Identifiability

A net force and net couple at the grip do not uniquely identify lead- and
trail-hand forces. Individual contact wrenches may be claimed only when they
come from instrumented contacts or from a fully declared constrained model with
an explicit allocation or regularization rule.

Two-hand analyses report the resultant/common mode, the opposed or
differential mode, the moment created by separated contact forces, applied free
wrist torques, and the equivalent wrench about a declared grip point. Internal
null-space forces can increase physiological demand while producing zero net
club wrench or zero net club work. Net club mechanics therefore cannot identify
individual-hand effort.

## Effort and Preactivation Boundaries

Neither measured grip force nor a modeled reaction force directly identifies
muscular effort. An effort claim requires a declared physiological estimand,
such as actuator torque or stress, positive and negative actuator work,
activation, metabolic cost, electromyography, or an instrumented force measure.
The selected estimator and its limitations must accompany the claim.

A nonzero pointwise ZTCF reaction is not free energy and does not show that the
state was reached without prior effort. It shows only that the reaction is
present in the zero-applied-control solve at the achieved state. Stored energy,
momentum, passive impedance, and earlier active work may all contribute. Only a
forward or branched ZTCF can test how long a response persists.

The proposed late-downswing preactivation benefit remains a model-only,
falsifiable hypothesis until tested with activation dynamics,
electromechanical delay, torque-rate and torque-velocity limits, and matched
effort constraints. Rigid-body timing alone cannot establish a physiological
preactivation mechanism, and no model result is described as an observed human
adaptation without independent data.

## Required Reporting and Validation

Every model tier must provide:

1. total, drift, control, and separately defined ZVCF traces;
2. force-vector and wrench-power views with fixed signs and scales;
3. event-window impulse, power, work, and robust-share tables;
4. acceleration, reaction, power, and work reconstruction residuals;
5. timestep, event-boundary, filtering, and parameter sensitivity; and
6. a statement of which conclusions survive the next model tier.

Permitted language is **associated with**, **attributed within the declared
model**, or **consistent with**. The words **effort**, **caused**, **passive**,
and **optimal** require the corresponding estimator, intervention,
counterfactual inventory, or objective and constraints to be stated.
