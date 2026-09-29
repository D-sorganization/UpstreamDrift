"""Jacobian equation HTML content: ZTCF, Jacobian, Constraint Jacobian.

Extracted from equations_popup.py.
"""

from __future__ import annotations

from ._equations_popup_css import _CSS

_ZTCF_HTML = f"""
<html><head><style>{_CSS}</style></head><body>

<h1>ZTCF Transfer Matrix — Endpoint Forces</h1>

<h2>1. What Is ZTCF?</h2>
<p>
<b>ZTCF</b> stands for <b>Zero-Torque Constraint Force</b> and is the name
of the transfer matrix that maps generalized torques (joint motor commands)
to the resulting Cartesian forces at the endpoint (club tip for the golfer
model).  We denote this matrix as <span class="eq-inline">T</span>:
</p>
<div class="eq">
F<sub>endpoint</sub> = T · τ
</div>
<p>
where <span class="eq-inline">F<sub>endpoint</sub></span> is the force vector at
the end-effector in Cartesian coordinates (typically 2D or 3D) and
<span class="eq-inline">τ</span> is the vector of joint torques.
</p>

<h2>2. Derivation from Jacobian</h2>
<p>
The relationship between joint velocities and endpoint velocities is given
by the geometric Jacobian <span class="eq-inline">J(q)</span>:
</p>
<div class="eq">
v<sub>endpoint</sub> = J(q) · q̇
</div>
<p>
where <span class="eq-inline">J</span> is the <b>m × n</b> Jacobian
(m = endpoint DOF, n = joint DOF).
</p>
<p>
Virtual work at the endpoint must equal virtual work at the joints:
</p>
<div class="eq">
F<sub>endpoint</sub>ᵀ δx = τᵀ δq
</div>
<p>
Using <span class="eq-inline">δx = J δq</span>:
</p>
<div class="eq">
F<sub>endpoint</sub>ᵀ J δq = τᵀ δq
</div>
<p>
This gives the force-torque dual relationship:
</p>
<div class="eq">
F<sub>endpoint</sub> = Jᵀ λ  (where λ are constraint multipliers)
</div>

<h2>3. ZTCF Formula</h2>
<p>
The ZTCF transfer matrix is derived by solving for the endpoint force
that results from applying joint torques under the assumption of
zero external load:
</p>
<div class="eq">
T = (J M⁺ Jᵀ)⁻¹ J M⁺
</div>
<table class="params">
<tr><td>J</td><td>Jacobian matrix (m × n), maps joint velocities to endpoint velocity</td></tr>
<tr><td>M⁺</td><td>Moore-Penrose pseudoinverse of the mass matrix</td></tr>
<tr><td>T</td><td>ZTCF transfer matrix (m × n), maps joint torques to endpoint forces</td></tr>
</table>

<h2>4. Physical Interpretation</h2>
<p>
If you apply a torque vector <span class="eq-inline">τ</span> at the joints
and no external load acts on the endpoint, the resulting endpoint force is:
</p>
<div class="eq">
F = T · τ
</div>
<p>
This force is the "reaction force" that appears at the endpoint due to the
inertias and configuration of the mechanism.  In a golf swing context, it
represents the force that the hands must exert (or resist) to keep the club
accelerating under the applied motor torques.
</p>
<p>
The term <b>"zero-torque"</b> refers to the fact that no external torque
is applied at the endpoint—only the forces arising from the internal
dynamics of the mechanism.
</p>

<h2>5. Jacobian for the Golfer Model</h2>
<p>
For the golfer upper-body model, the endpoint is the <b>club tip</b>
(the contact point with the ball).  The Jacobian J is computed by
differentiating the forward kinematics of the club tip position with
respect to all 8 generalized coordinates:
</p>
<div class="eq">
p<sub>club_tip</sub>(q) = f(q)
</div>
<div class="eq">
J(q) = ∂f/∂q
</div>
<p>
The Jacobian is typically <b>2 × 8</b> (club tip position in 2D) or
<b>3 × 8</b> (in 3D, including height).  It encodes how each joint's
motion contributes to moving the club tip.
</p>

<h2>6. Singularity and Invertibility</h2>
<p>
The ZTCF matrix <span class="eq-inline">T = (J M⁺ Jᵀ)⁻¹ J M⁺</span>
can become singular (non-invertible) at certain configurations where:
</p>
<ul>
<li><b>Jacobian singularity:</b> The Jacobian J loses rank, typically when
the endpoint reaches a workspace boundary or kinematic singular point.</li>
<li><b>Zero-determinant of J M⁺ Jᵀ:</b> The effective "operational-space mass"
becomes zero, meaning no endpoint force can be generated for a given
joint torque.</li>
</ul>
<p>
In such cases, the inverse fails to exist and the simulator returns
<span class="eq-inline">T = None</span>, indicating that endpoint forces
cannot be reliably computed at that instant.  This is typically brief
(single time-step) and resolved as the configuration moves away from
the singularity.
</p>

<h2>7. Relationship to Impedance Control</h2>
<p>
In robot control, the ZTCF matrix is closely related to operational-space
impedance:
</p>
<div class="eq">
M<sub>op</sub> = (J M⁺ Jᵀ)⁻¹
</div>
<p>
This is the <b>operational-space mass</b> or <b>effective inertia</b> at
the endpoint.  The ZTCF matrix relates to it by:
</p>
<div class="eq">
T = M<sub>op</sub> · J M⁺
</div>
<p>
Configurations where M<sub>op</sub> has small eigenvalues are those where
the endpoint feels "light" or difficult to control—characteristic of
kinematic singularities or near-parallel segments.
</p>

</body></html>
"""

# ---------------------------------------------------------------------------
# Jacobian content
# ---------------------------------------------------------------------------

_JACOBIAN_HTML = f"""
<html><head><style>{_CSS}</style></head><body>

<h1>Geometric Jacobian — Velocity Mapping</h1>

<h2>1. What Is the Jacobian?</h2>
<p>
The <b>geometric Jacobian</b> <span class="eq-inline">J(q)</span> is the
matrix that maps joint velocities to Cartesian (task-space) velocities
at a particular endpoint:
</p>
<div class="eq">
v<sub>endpoint</sub> = J(q) · q̇
</div>
<p>
It encodes the <em>instantaneous kinematic relationship</em> between
how fast each joint moves and how fast the endpoint (e.g. club tip,
wrist, hub) translates in 2D space.  The Jacobian depends on the
current configuration <span class="eq-inline">q</span> and changes at
every time step.
</p>

<h2>2. Double Pendulum (2R) Jacobian</h2>
<p>
For the 2-DOF pendulum with endpoint at the tip
(position <span class="eq-inline">p = (x, y)</span>):
</p>
<div class="eq">
<span class="matrix">
     ┌                                                               ┐
J =  │  L₁ cos θ₁ + L₂ cos(θ₁+φ)      L₂ cos(θ₁+φ)               │
     │  L₁ sin θ₁ + L₂ sin(θ₁+φ)      L₂ sin(θ₁+φ)               │
     └                                                               ┘
</span>
</div>
<p>
This is a 2×2 matrix.  Each column represents how fast the tip moves
(in x and y) per unit angular velocity of that joint.
</p>

<h3>2.1 Physical Interpretation of Columns</h3>
<table class="params">
<tr><td>Column 1</td><td>Tip velocity contribution from shoulder rotation (θ̇₁).
Includes both the direct arm swing and the coupled wrist effect.</td></tr>
<tr><td>Column 2</td><td>Tip velocity contribution from wrist rotation (φ̇).
Only involves the distal segment — shorter lever arm.</td></tr>
</table>

<div class="note">
<b>Key insight:</b> The Jacobian becomes singular when the two segments
are aligned (φ = 0 or φ = π).  At singularity, the mechanism cannot
generate velocity in one Cartesian direction regardless of joint speeds.
This corresponds to a "locked" configuration.
</div>

<h2>3. Triple Pendulum (3R) Jacobian</h2>
<p>
For the 3-DOF pendulum, the Jacobian at the tip is 2×3:
</p>
<div class="eq">
<span class="matrix">
     ┌                                                                        ┐
J =  │  ∂x<sub>tip</sub>/∂θ₁   ∂x<sub>tip</sub>/∂φ₁   ∂x<sub>tip</sub>/∂φ₂  │
     │  ∂y<sub>tip</sub>/∂θ₁   ∂y<sub>tip</sub>/∂φ₁   ∂y<sub>tip</sub>/∂φ₂  │
     └                                                                        ┘
</span>
</div>
<p>
The 3R mechanism is <b>redundant</b> for 2D positioning — 3 joints
controlling 2 task-space DOFs.  This means there is a 1-dimensional
<em>null space</em>: joint velocity combinations that move the joints
without affecting the endpoint.  Physically, this is an internal
reconfiguration motion.
</p>

<h2>4. Golfer Model (8-DOF) Jacobian</h2>
<p>
The golfer upper-body model uses 8 generalized coordinates.  The
Jacobian for any endpoint (hub, right elbow, left wrist, club tip,
etc.) is <b>2×8</b>:
</p>
<div class="eq">
J<sub>endpoint</sub>(q) = ∂p<sub>endpoint</sub>/∂q  ∈ ℝ<sup>2×8</sup>
</div>
<p>
These are computed analytically via the chain rule applied to the
forward kinematics.  For each of the 7 key joints
(hub, RS, RE, RH, LS, LE, LH) and the club tip, the simulator
provides individual 2×8 Jacobians.
</p>

<h3>4.1 Analytical vs. Numerical Jacobians</h3>
<p>
The simulator computes Jacobians two ways:
</p>
<table class="params">
<tr><td>Analytical</td><td>Derived symbolically from the FK chain rule.
Exact to machine precision. Used for real-time display and ZTCF.</td></tr>
<tr><td>Numerical</td><td>Finite-difference approximation: J<sub>ij</sub> ≈
(f(q+ε eⱼ) − f(q)) / ε with ε = 10⁻⁷. Used for validation.</td></tr>
</table>

<h2>5. Manipulability Ellipsoid</h2>
<p>
The Jacobian defines the <b>velocity manipulability ellipsoid</b>
at each endpoint.  For unit joint velocity (‖q̇‖ = 1), the set of
achievable endpoint velocities forms an ellipse:
</p>
<div class="eq">
v<sub>max</sub> = σ₁(J),  v<sub>min</sub> = σ₂(J)
</div>
<p>
where σ₁ ≥ σ₂ are the singular values of J.  The
<b>manipulability index</b> is:
</p>
<div class="eq">
w(q) = σ₁ · σ₂ = √det(J Jᵀ)
</div>
<p>
A large manipulability index means the endpoint can move easily in
all directions.  The index drops to zero at kinematic singularities.
</p>

<h2>6. Force Ellipsoid (Dual)</h2>
<p>
The force ellipsoid is the dual of the velocity ellipsoid.  For unit
endpoint force, the required joint torques are bounded by:
</p>
<div class="eq">
τ = Jᵀ F<sub>endpoint</sub>
</div>
<p>
The force ellipsoid semi-axes are <span class="eq-inline">1/σᵢ</span> —
directions where the mechanism moves fast (high σ) are directions where
it is weak at applying force, and vice versa.  This is a fundamental
trade-off in mechanism design.
</p>

<div class="note">
<b>Golf application:</b> At impact, a large force ellipsoid in the
swing direction means the golfer can effectively transfer torque into
ball speed.  The Jacobian structure at impact determines how efficiently
the kinetic chain delivers force to the club.
</div>

</body></html>
"""

# ---------------------------------------------------------------------------
# Constraint Jacobian content
# ---------------------------------------------------------------------------

_CONSTRAINT_JACOBIAN_HTML = f"""
<html><head><style>{_CSS}</style></head><body>

<h1>Constraint Jacobian — Closed-Loop Kinematics</h1>

<h2>1. What Is the Constraint Jacobian?</h2>
<p>
The <b>constraint Jacobian</b> <span class="eq-inline">Φ<sub>q</sub></span>
is the matrix of partial derivatives of the holonomic constraint
equations with respect to the generalized coordinates:
</p>
<div class="eq">
Φ<sub>q</sub>(q) = ∂Φ/∂q  ∈ ℝ<sup>c×n</sup>
</div>
<p>
where <span class="eq-inline">c</span> is the number of scalar constraints
and <span class="eq-inline">n</span> is the number of generalized coordinates.
This matrix appears in the constrained equations of motion and is essential
for maintaining the closed kinematic loop in the golfer model.
</p>

<h2>2. Why Is It Needed?</h2>

<h3>2.1 Open vs. Closed Kinematic Chains</h3>
<p>
The double and triple pendulum models are <b>open chains</b> — each joint
connects only to its neighbors.  No constraint Jacobian is needed because
the equations of motion are naturally unconstrained.
</p>
<p>
The golfer model is a <b>closed chain</b> — both arms connect to the same
club, forming a loop.  The constraint equations enforce that both hands
grip the club at specified positions:
</p>
<div class="eq">
<span class="matrix">
Φ(q) = ┌ p<sub>R,hand</sub>(q) − p<sub>club,grip_R</sub>(q) ┐
        │                                          │  = 0
        └ p<sub>L,hand</sub>(q) − p<sub>club,grip_L</sub>(q) ┘
</span>
</div>
<p>
Each of the two vector equations (right hand = right grip, left hand = left
grip) provides 2 scalar constraints (x and y), giving <b>4 constraints total</b>.
</p>

<h2>3. Golfer Constraint Jacobian Structure</h2>
<p>
For the 8-DOF golfer with 4 constraints, the constraint Jacobian is a
<b>4×8 matrix</b>:
</p>
<div class="eq">
<span class="matrix">
           ┌  ∂Φ₁/∂q₁  ∂Φ₁/∂q₂  ...  ∂Φ₁/∂q₈ ┐
Φ<sub>q</sub> =  │  ∂Φ₂/∂q₁  ∂Φ₂/∂q₂  ...  ∂Φ₂/∂q₈ │   ∈ ℝ<sup>4×8</sup>
           │  ∂Φ₃/∂q₁  ∂Φ₃/∂q₂  ...  ∂Φ₃/∂q₈ │
           └  ∂Φ₄/∂q₁  ∂Φ₄/∂q₂  ...  ∂Φ₄/∂q₈ ┘
</span>
</div>
<p>
Each row encodes how a particular constraint error changes when each
joint angle changes.  The matrix is computed by differentiating the
forward kinematics of both arm endpoints and the club grip points.
</p>

<h3>3.1 Sparsity Pattern</h3>
<p>
The constraint Jacobian has a characteristic <b>block-sparse</b> pattern
reflecting the kinematic tree topology:
</p>
<table class="params">
<tr><td>Rows 1–2</td><td>Right-hand constraint: nonzero entries only in columns
for hub, right-arm joints, and club angle</td></tr>
<tr><td>Rows 3–4</td><td>Left-hand constraint: nonzero entries only in columns
for hub, left-arm joints, and club angle</td></tr>
</table>
<p>
Joints from one arm chain do not appear in the other arm's constraint rows
(except the hub and club which are shared).
</p>

<h2>4. Role in Constrained Dynamics</h2>
<p>
The constraint Jacobian appears in three critical places:
</p>

<h3>4.1 KKT System (Equations of Motion)</h3>
<div class="eq">
<span class="matrix">
┌  M    Φ<sub>q</sub>ᵀ ┐ ┌ q̈ ┐   ┌ τ − C − G                            ┐
│            │ │     │ = │                                        │
└  Φ<sub>q</sub>   0   ┘ └ λ  ┘   └ −γ − 2α Φ̇ − β² Φ     ┘
</span>
</div>
<p>
The constraint Jacobian transposed (<span class="eq-inline">Φ<sub>q</sub>ᵀ</span>)
maps Lagrange multipliers λ to constraint forces in joint space.
The constraint Jacobian itself (<span class="eq-inline">Φ<sub>q</sub></span>)
projects accelerations onto the constraint manifold.
</p>

<h3>4.2 Constraint Force Computation</h3>
<div class="eq">
τ<sub>constraint</sub> = Φ<sub>q</sub>ᵀ λ
</div>
<p>
The constraint forces τ<sub>constraint</sub> are the "internal" forces
that the mechanism must generate to maintain the closed loop.  They
represent the grip forces transmitted through the hands to the club.
</p>

<h3>4.3 Velocity-Level Constraint</h3>
<div class="eq">
Φ<sub>q</sub> · q̇ = 0
</div>
<p>
At the velocity level, the constraint Jacobian enforces that the
joint velocities remain consistent with the closed loop — the hands
cannot slide relative to the grip.
</p>

<h2>5. Rank and Singularity</h2>
<p>
For the system to be well-posed, the constraint Jacobian must have
<b>full row rank</b> (rank = c = 4).  If Φ<sub>q</sub> becomes rank-deficient:
</p>
<ul>
<li>The KKT system becomes singular (no unique solution)</li>
<li>Some constraint forces become indeterminate</li>
<li>The mechanism is at a <b>constraint singularity</b></li>
</ul>
<p>
This can occur when the arms are fully extended or in certain degenerate
configurations.  The simulator detects this via SVD and logs a warning.
</p>

<h2>6. Analytical vs. Numerical Computation</h2>
<table class="params">
<tr><td>Analytical</td><td>Derived by differentiating the FK equations
symbolically.  Each entry is an explicit function of q.
Computed via <code>analytical_constraint_jacobian(q, p)</code>.</td></tr>
<tr><td>Numerical</td><td>Central finite differences: Φ<sub>q,ij</sub> ≈
(Φᵢ(q+εeⱼ) − Φᵢ(q−εeⱼ)) / 2ε.
Computed via <code>numerical_constraint_jacobian(q, p)</code>.</td></tr>
</table>
<p>
Both methods are provided for cross-validation.  The analytical version
is used in production for speed; numerical for testing correctness.
</p>

<h2>7. Relationship to Effective DOF</h2>
<p>
The effective degrees of freedom of the constrained system are:
</p>
<div class="eq">
n<sub>eff</sub> = n − rank(Φ<sub>q</sub>) = 8 − 4 = 4
</div>
<p>
The 4 effective DOFs correspond to motions that respect the closed-loop
constraint (e.g., hub rotation, arm swing, wrist cock, club rotation
about the grip axis).  The null space of Φ<sub>q</sub> defines these
feasible motion directions.
</p>

<div class="note">
<b>Physical meaning:</b> The constraint Jacobian is the mathematical
expression of "the hands stay on the club."  Every row says: "if you
move the joints this way, the grip error changes by this amount."
Zero rows would mean a constraint that is satisfied regardless of
configuration (a degenerate constraint).  Full-rank Φ<sub>q</sub> means
all 4 constraints are actively restricting motion.
</div>

</body></html>
"""

__all__ = ["_ZTCF_HTML", "_JACOBIAN_HTML", "_CONSTRAINT_JACOBIAN_HTML"]
