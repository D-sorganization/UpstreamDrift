"""Reference equation topics for the pendulum equations popup."""

from __future__ import annotations

from .equations_popup_styles import EQUATIONS_POPUP_CSS

MASS_MATRIX_HTML = f"""
<html><head><style>{EQUATIONS_POPUP_CSS}</style></head><body>

<h1>Mass (Inertia) Matrix — Derivation &amp; Interpretation</h1>

<h2>1. What Is the Mass Matrix?</h2>
<p>
The <b>mass matrix</b> <span class="eq-inline">M(q)</span> is the
configuration-dependent inertia tensor of the mechanism in generalized
coordinates.  It maps joint accelerations to the generalized forces
required to produce them:
</p>
<div class="eq">
τ = M(q) · q̈ + C(q, q̇) + G(q)
</div>
<p>
Physically, <span class="eq-inline">M(q)</span> tells you "how heavy does
the system feel" when you try to accelerate a particular joint.
It is always <b>symmetric</b> (M = Mᵀ) and <b>positive definite</b>
(all eigenvalues > 0), guaranteeing a unique solution for
<span class="eq-inline">q̈ = M⁻¹(τ − C − G)</span>.
</p>

<h2>2. Double Pendulum (2R) Mass Matrix</h2>
<p>
For a two-segment pendulum with generalized coordinates
<span class="eq-inline">q = [θ₁, φ]ᵀ</span> where θ₁ is the absolute
shoulder angle and φ is the wrist angle relative to the arm:
</p>
<div class="eq">
<span class="matrix">
     ┌                                                            ┐
M =  │  (m₁+mₑ)L₁² + mₑL₂² + 2mₑL₁L₂cos φ    mₑL₂² + mₑL₁L₂cos φ  │
     │  mₑL₂² + mₑL₁L₂cos φ                     mₑL₂²                  │
     └                                                            ┘
</span>
</div>
<p>where <span class="eq-inline">mₑ = m₂ + m<sub>club</sub></span>
is the effective distal mass (shaft + clubhead).</p>

<h3>2.1 Parameter Definitions</h3>
<table class="params">
<tr><td>m₁</td><td>Mass of segment 1 (arms), kg</td></tr>
<tr><td>m₂</td><td>Mass of segment 2 (shaft), kg</td></tr>
<tr><td>m<sub>club</sub></td><td>Clubhead point mass at tip of shaft, kg</td></tr>
<tr><td>mₑ</td><td>Effective mass: m₂ + m<sub>club</sub></td></tr>
<tr><td>L₁</td><td>Length of segment 1, m</td></tr>
<tr><td>L₂</td><td>Length of segment 2, m</td></tr>
<tr><td>φ</td><td>Relative wrist angle, rad</td></tr>
</table>

<h3>2.2 Physical Interpretation of Each Entry</h3>
<ul>
<li><b>M₁₁</b> — Effective inertia at the shoulder when the wrist is locked.
Includes contributions from both segments plus the coupling term
<span class="eq-inline">2mₑL₁L₂cos φ</span> that depends on the relative
configuration.</li>
<li><b>M₂₂</b> — Effective inertia at the wrist when the shoulder is locked.
This is simply <span class="eq-inline">mₑL₂²</span>, the inertia of the
distal segment about the wrist.</li>
<li><b>M₁₂ = M₂₁</b> — <b>Inertial coupling</b>. This is the key term for
energy transfer in golf swings. When joint 1 accelerates, the off-diagonal
term creates a "free" torque at joint 2 (and vice versa).  This is how the
proximal-to-distal kinetic chain works.</li>
</ul>

<div class="note">
<b>Key insight:</b> The coupling term is proportional to
<span class="eq-inline">cos φ</span>. Maximum coupling occurs when the
segments are aligned (φ = 0). Coupling vanishes at φ = ±90°.
In a golf swing, the delayed wrist release keeps the segments aligned
for maximum coupling transfer at impact.
</div>

<h3>2.3 Derivation Sketch</h3>
<p>Start from the kinetic energy in Cartesian coordinates, then transform
to generalized coordinates using the Jacobian:</p>
<div class="eq">
T = ½ q̇ᵀ M(q) q̇ = ½ Σᵢ mᵢ vᵢᵀvᵢ
</div>
<p>The velocity of each point mass is obtained via the geometric Jacobian
<span class="eq-inline">vᵢ = Jᵢ(q) q̇</span>, so:</p>
<div class="eq">
M(q) = Σᵢ mᵢ Jᵢᵀ Jᵢ
</div>

<h2>3. Triple Pendulum (3R) Mass Matrix</h2>
<p>
For three segments with coordinates <span class="eq-inline">q = [θ₁, φ₁, φ₂]ᵀ</span>,
the mass matrix is 3×3 symmetric:
</p>
<div class="eq">
<span class="matrix">
      ┌                    ┐
M  =  │  M₁₁   M₁₂   M₁₃ │
      │  M₁₂   M₂₂   M₂₃ │
      │  M₁₃   M₂₃   M₃₃ │
      └                    ┘
</span>
</div>
<p>Each entry follows the same pattern as the 2R case but with
additional terms for the third segment.  The coupling structure means
that accelerating the hub (joint 1) creates inertial torques at
both the arm (joint 2) and club (joint 3) simultaneously.</p>

<div class="note">
In the golf context: hub = sternum-to-shoulder (~0.15 m),
arm = shoulder-to-wrist (~0.60 m), club = wrist-to-clubhead (~1.10 m).
The short first segment means M₁₁ is dominated by the coupled terms.
</div>

<h2>4. Golfer Model (8-DOF) Mass Matrix</h2>
<p>
The golfer upper-body model uses 8 generalized coordinates for a closed
kinematic loop (left arm + right arm + shared club).  The mass matrix is
8×8 but has block structure reflecting the kinematic tree:
</p>
<div class="eq">
<span class="matrix">
      ┌                                              ┐
M  =  │  M<sub>hub</sub>    M<sub>hub,R</sub>   M<sub>hub,L</sub>   M<sub>hub,club</sub>  │
      │  M<sub>hub,R</sub>ᵀ  M<sub>R</sub>      0           M<sub>R,club</sub>   │
      │  M<sub>hub,L</sub>ᵀ  0           M<sub>L</sub>      M<sub>L,club</sub>   │
      │  M<sub>hub,club</sub>ᵀ M<sub>R,club</sub>ᵀ M<sub>L,club</sub>ᵀ M<sub>club</sub>     │
      └                                              ┘
</span>
</div>
<p>
The closed loop introduces 4 holonomic constraints
<span class="eq-inline">Φ(q) = 0</span> that enforce the two hands
meeting at the grip.  These are handled via <b>Baumgarte stabilization</b>
(see Equations of Motion section).
</p>

</body></html>
"""

# ---------------------------------------------------------------------------
# Equations of Motion content
# ---------------------------------------------------------------------------

EOM_HTML = f"""
<html><head><style>{EQUATIONS_POPUP_CSS}</style></head><body>

<h1>Equations of Motion — Complete Derivation</h1>

<h2>1. Lagrangian Formulation</h2>
<p>
The equations of motion are derived from the Euler-Lagrange equations.
The Lagrangian is <span class="eq-inline">L = T − V</span> where T is
kinetic energy and V is potential energy.  For a driven system with
dissipation:
</p>
<div class="eq">
d/dt (∂L/∂q̇ᵢ) − ∂L/∂qᵢ = τᵢ + τ<sub>friction,i</sub> + τ<sub>limits,i</sub>
</div>
<p>This yields the standard manipulator equation:</p>
<div class="eq">
M(q) · q̈ = τ<sub>drive</sub> + τ<sub>friction</sub> + τ<sub>limits</sub> − C(q, q̇) − G(q)
</div>

<h2>2. Individual Terms</h2>

<h3>2.1 Mass Matrix M(q)</h3>
<p>See the Mass Matrix tab for full derivation. Configuration-dependent,
symmetric positive definite. Computed analytically for each model.</p>

<h3>2.2 Coriolis &amp; Centrifugal Vector C(q, q̇)</h3>
<p>For the 2R pendulum:</p>
<div class="eq">
<span class="matrix">
      ┌                                        ┐
C  =  │  −h (2 θ̇₁ φ̇ + φ̇²)           │
      │   h θ̇₁²                           │
      └                                        ┘

where  h = mₑ L₁ L₂ sin φ
</span>
</div>
<p>
<b>Physical meaning:</b> These terms arise from the velocity-dependent
"fictitious forces" in the rotating reference frames.  The first
component contains both the Coriolis term (∝ θ̇₁ φ̇) and the centrifugal
term (∝ φ̇²).
</p>

<h3>2.3 Gravity Vector G(q)</h3>
<div class="eq">
<span class="matrix">
      ┌                                                      ┐
G  =  │  (m₁ + mₑ) g L₁ sin θ₁  +  mₑ g L₂ sin(θ₁ + φ)  │
      │  mₑ g L₂ sin(θ₁ + φ)                                │
      └                                                      ┘
</span>
</div>
<p>Gravity acts through the COM of each segment.  The absolute angle
of segment 2 is <span class="eq-inline">θ₁ + φ</span> (relative
coordinate convention).</p>

<h3>2.4 Driving Torque τ<sub>drive</sub></h3>
<p>User-specified as polynomial functions of time:</p>
<div class="eq">
τᵢ(t) = c₀ + c₁t + c₂t² + c₃t³ + ...
</div>
<p>Subject to saturation limits (torque clamping):</p>
<div class="eq">
τᵢ,clamped = clip(τᵢ, −τ<sub>max,i</sub>, +τ<sub>max,i</sub>)
</div>
<div class="note">
The absolute-value clamp ensures symmetric limits (±τ<sub>max</sub>)
even if the user enters a negative limit value.  This prevents the
common error of only limiting positive torque.
</div>

<h3>2.5 Friction Torque τ<sub>friction</sub></h3>
<div class="eq">
τ<sub>friction,i</sub> = −bᵢ q̇ᵢ  −  μᵢ sign(q̇ᵢ)
</div>
<table class="params">
<tr><td>bᵢ</td><td>Viscous damping coefficient (N·m·s/rad)</td></tr>
<tr><td>μᵢ</td><td>Coulomb friction magnitude (N·m)</td></tr>
</table>

<h3>2.6 Joint Limit Penalty τ<sub>limits</sub></h3>
<p>Smooth Hermite smoothstep barrier at joint angle limits:</p>
<div class="eq">
τ<sub>lim</sub>(q) = s(d) · [ K·d + B·max(0, −q̇) ]

where d = q<sub>min</sub> − q  (penetration depth)
      s(x) = 3x² − 2x³  (smoothstep blend, x ∈ [0,1])
</div>
<table class="params">
<tr><td>K</td><td>Penalty stiffness (N·m/rad)</td></tr>
<tr><td>B</td><td>Penalty damping (N·m·s/rad)</td></tr>
</table>

<h2>3. Numerical Integration</h2>
<p>The ODE system is solved using adaptive Runge-Kutta methods:</p>
<table class="params">
<tr><td>Double</td><td>RK45 (Dormand-Prince, 5th order), rtol=1e-8, atol=1e-10</td></tr>
<tr><td>Triple</td><td>DOP853 (8th order), rtol=1e-6, atol=1e-8</td></tr>
<tr><td>Golfer</td><td>RK45 with constraint projection at each step</td></tr>
</table>

<h2>4. Energy Conservation</h2>
<div class="eq">
E = T + V = ½ q̇ᵀ M(q) q̇  +  V(q)

dE/dt = q̇ᵀ τ<sub>drive</sub>  +  q̇ᵀ τ<sub>friction</sub>
</div>
<p>
When τ<sub>drive</sub> = 0 and friction = 0, total energy is conserved
(E = const).  This is used as a sanity check for the integrator.
</p>

<h2>5. Golfer Model — Constrained Dynamics (KKT System)</h2>
<p>
The 8-DOF golfer model has 4 holonomic constraints
<span class="eq-inline">Φ(q) = 0</span> enforcing that both hands
grip the same club.  The constrained EOM are:
</p>
<div class="eq">
<span class="matrix">
┌  M    Φ<sub>q</sub>ᵀ ┐ ┌ q̈ ┐   ┌ τ − C − G                                ┐
│            │ │     │ = │                                            │
└  Φ<sub>q</sub>   0   ┘ └ λ  ┘   └ −γ(q,q̇) − 2α Φ̇(q,q̇) − β² Φ(q) ┘
</span>
</div>

<h3>5.1 Constraint Equations</h3>
<p>The 4 constraints enforce that the right-hand endpoint and left-hand
endpoint coincide at the club grip:</p>
<div class="eq">
Φ(q) = p<sub>R,wrist</sub>(q) − p<sub>L,wrist</sub>(q) = 0   (2 eqs: x, y)
Φ(q) = p<sub>R,wrist</sub>(q) − p<sub>club,base</sub>(q) = 0  (2 eqs: x, y)
</div>

<h3>5.2 Baumgarte Stabilization</h3>
<p>To prevent numerical constraint drift, the acceleration-level
constraint is augmented with position and velocity feedback:</p>
<div class="eq">
Φ<sub>q</sub> q̈ = −γ(q,q̇) − 2α Φ̇ − β² Φ
</div>
<table class="params">
<tr><td>α</td><td>Velocity feedback gain (typical: 5–20)</td></tr>
<tr><td>β</td><td>Position feedback gain (typical: 5–20)</td></tr>
<tr><td>Φ<sub>q</sub></td><td>Constraint Jacobian (4×8 matrix)</td></tr>
<tr><td>γ</td><td>Constraint bias: γ = −Φ̇<sub>q</sub> q̇ (velocity-level RHS)</td></tr>
<tr><td>λ</td><td>Lagrange multipliers (constraint forces)</td></tr>
</table>

<div class="note">
<b>Why Baumgarte?</b> Direct constraint enforcement at the acceleration
level allows constraint violations to accumulate (drift).  Baumgarte
feedback drives violations to zero exponentially, like a PD controller
on the constraint error.  Higher α, β = faster stabilization but can
cause stiffness.
</div>

<h2>6. Impulse, Work, and Power</h2>

<h3>6.1 Angular Power</h3>
<div class="eq">
P<sub>angular,i</sub>(t) = τᵢ(t) · ω<sub>i</sub>(t)
</div>
<p>Power delivered by joint torque τ at angular velocity ω.
Positive = energy flowing from proximal to distal.</p>

<h3>6.2 Linear Power</h3>
<div class="eq">
P<sub>linear,i</sub>(t) = F<sub>i</sub>(t) · v<sub>i</sub>(t)
</div>
<p>Power delivered by net joint force at the joint's linear velocity.</p>

<h3>6.3 Angular Impulse</h3>
<div class="eq">
J<sub>angular,i</sub>(t) = ∫₀ᵗ τᵢ(s) ds
</div>

<h3>6.4 Angular Work</h3>
<div class="eq">
W<sub>angular,i</sub>(t) = ∫₀ᵗ τᵢ(s) · ωᵢ(s) ds = ∫₀ᵗ P<sub>angular,i</sub>(s) ds
</div>

<h3>6.5 Linear Impulse</h3>
<div class="eq">
J<sub>linear,i</sub>(t) = ∫₀ᵗ F<sub>i</sub>(s) ds
</div>

<h3>6.6 Joint Moments</h3>
<p>At each joint, three moment quantities are computed (proximal on distal):</p>
<div class="eq">
M<sub>applied</sub> = τ<sub>joint</sub>                    (motor/muscle torque)
M<sub>force</sub>   = r × F<sub>net</sub>                  (moment of net force)
M<sub>total</sub>   = M<sub>applied</sub> + M<sub>force</sub>  (total moment)
</div>
<p>where <span class="eq-inline">r</span> is the position vector from the
joint to the distal segment's center of mass.</p>

</body></html>
"""

# ---------------------------------------------------------------------------
# Delta Matrix content
# ---------------------------------------------------------------------------

DELTA_HTML = f"""
<html><head><style>{EQUATIONS_POPUP_CSS}</style></head><body>

<h1>Delta Matrix (M⁺) — Inverse Dynamics</h1>

<h2>1. What Is the Delta Matrix?</h2>
<p>
The <b>delta matrix</b> <span class="eq-inline">D</span> is the mapping
from generalized torques to joint accelerations in the feasible subspace.
For unconstrained systems it is simply the inverse of the mass matrix.
For constrained systems (like the golfer model with holonomic constraints),
it is the Moore-Penrose pseudoinverse of the mass matrix:
</p>
<div class="eq">
D = M⁺
</div>
<p>
The pseudoinverse is necessary because the constrained system has a
rank-deficient mass matrix—accelerations in certain directions are
forbidden by the constraints.  The Delta matrix maps torques to the
accelerations that can actually occur:
</p>
<div class="eq">
q̈ = M⁺ · (τ − C − G)
</div>

<h2>2. Moore-Penrose Pseudoinverse Properties</h2>
<p>
The Moore-Penrose pseudoinverse M⁺ is the unique matrix satisfying
all four Penrose conditions:
</p>
<div class="eq">
<span class="matrix">
M · M⁺ · M = M
M⁺ · M · M⁺ = M⁺
(M · M⁺)ᵀ = M · M⁺
(M⁺ · M)ᵀ = M⁺ · M
</span>
</div>
<p>
These conditions guarantee that:
</p>
<ul>
<li>The pseudoinverse is unique</li>
<li>If M is square and invertible, then M⁺ = M⁻¹</li>
<li>For rank-deficient M, M⁺ projects onto the row space of M
(and the column space, symmetrically)</li>
<li>Applying M⁺ twice gives back M⁺</li>
</ul>

<h2>3. When Do We Need the Pseudoinverse?</h2>

<h3>3.1 Double and Triple Pendulums</h3>
<p>
For the unconstrained double pendulum (2R) and triple pendulum (3R),
the mass matrix M is:
</p>
<ul>
<li><b>Square:</b> 2×2 or 3×3</li>
<li><b>Full rank:</b> All eigenvalues strictly positive</li>
<li><b>Symmetric positive definite:</b> Guarantees M⁻¹ exists</li>
</ul>
<p>
In these cases, the Delta matrix is simply the standard inverse:
</p>
<div class="eq">
D = M⁻¹  (Double and Triple Pendulums)
</div>

<h3>3.2 Golfer Model (8-DOF)</h3>
<p>
The golfer upper-body model uses 8 generalized coordinates
<span class="eq-inline">q = [θ<sub>hub</sub>, θ<sub>R,shoulder</sub>, φ<sub>R,wrist</sub>,
θ<sub>L,shoulder</sub>, φ<sub>L,wrist</sub>, θ<sub>club,lead</sub>,
θ<sub>club,trail</sub>, θ<sub>club,roll</sub>]ᵀ</span>
with 4 holonomic constraints enforcing that both hands grip the club.
</p>
<p>
The mass matrix M is 8×8, but the 4 constraints eliminate 4 degrees of
freedom, leaving only <b>rank 6</b>.  This is because only 4 independent
accelerations can be achieved in the constrained space—the other 4 directions
would violate the grip constraint.
</p>
<p>
Computing the pseudoinverse M⁺ and projecting the control torques gives the
accelerations that satisfy the constraints:
</p>
<div class="eq">
D = M⁺  (Golfer Model with 4 constraints)
</div>
<div class="note">
<b>Rank deficiency:</b> rank(M) = 8 − 4 = 6 for the 8-DOF golfer model.
The 4 null directions correspond to accelerations that would require
the hands to "slide" relative to the club grip.
</div>

<h2>4. Physical Interpretation</h2>
<p>
The Delta matrix tells you how to apply torques to get desired accelerations:
</p>
<ul>
<li><b>In unconstrained systems (2R, 3R):</b> Each torque τᵢ is divided by
the effective inertia (M⁻¹ᵢⱼ) to get the corresponding acceleration.
The off-diagonal terms capture inertial coupling.</li>
<li><b>In constrained systems (golfer):</b> The pseudoinverse automatically
enforces that accelerations remain in the feasible subspace.  Torques that
would create constraint-violating accelerations are projected into the
constrained manifold.</li>
</ul>

<h2>5. Computation</h2>
<p>
For numerical stability, the pseudoinverse is computed using singular value
decomposition (SVD):
</p>
<div class="eq">
M = U Σ Vᵀ
</div>
<p>
where U and V are orthogonal and Σ is diagonal with singular values σᵢ.
The pseudoinverse is:
</p>
<div class="eq">
M⁺ = V Σ⁺ Uᵀ
</div>
<p>
where Σ⁺ is diagonal with entries 1/σᵢ if σᵢ &gt; ε (threshold), else 0.
This avoids dividing by near-zero singular values.
</p>

</body></html>
"""

# ---------------------------------------------------------------------------
# ZTCF Matrix content
# ---------------------------------------------------------------------------

ZTCF_HTML = f"""
<html><head><style>{EQUATIONS_POPUP_CSS}</style></head><body>

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
