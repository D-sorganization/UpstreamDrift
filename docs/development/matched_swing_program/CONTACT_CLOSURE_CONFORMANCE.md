---
conformance_version: "1.0.0"
tolerances:
  static_equilibrium_force_n: 1.0e-6
  normal_force_parity_n: 1.0e-9
  friction_force_parity_n: 1.0e-9
  penetration_continuity_m: 1.0e-12
  zero_penetration_force_n: 0.0
  rebound_tensile_force_n: 0.0
  closure_rank_expected: 6
  closure_position_tolerance_m: 1.0e-4
  closure_orientation_tolerance_rad: 1.0e-4
  contact_sphere_radius_min_m: 0.001
  energy_dissipation_nonnegative: true
---

# Contact and Grip Closure Conformance Specification (MS-72)

## Scope and Governing Requirements

This specification formalizes the shared contact law and dual-grip kinematic closure
contracts across all six physics engines in the Matched Swing Program (epic #10363, MS-72 #10352).
Engines' native contact solvers evaluate contact and closed-chain constraints at differing
points within numerical integration steps (e.g. MuJoCo via `qfrc_applied`, Drake via external force
inputs, Pinocchio via Delassus constraint dynamics, and MJX via compliant approximations).
To compare full-body models like for like across engines, every engine consumes or evaluates
this versioned physical formulation.

Tolerances in this document govern merge gates and are loaded directly from the document's
YAML frontmatter by test harnesses, preserving the Dependency Inversion and Demeter principles.

## Normal Contact Formulation (Hunt-Crossley Form)

Ground contact is modeled via rigid ground plane collision with foot contact spheres.
Let $n \in \mathbb{R}^3$ denote the ground plane unit normal ($\|n\|_2 = 1$) and $h \in \mathbb{R}$
the signed plane height. For a contact sphere of radius $r > 0$ with center $p \in \mathbb{R}^3$
and velocity $v \in \mathbb{R}^3$:

1. **Signed Distance and Penetration**:
   $$d = \max(0, -(n^\top p - h - r))$$
   The penetration rate (positive when deepening) is:
   $$\dot{d} = -n^\top v$$

2. **Normal Force Magnitude**:
   $$f_n = \max\left(0, k \cdot d \cdot (1 + c \cdot \dot{d})\right)$$
   where:

   - $k > 0$ is normal stiffness in $\text{N}/\text{m}$ (`stiffness_n_m`)
   - $c \ge 0$ is dissipation in $\text{s}/\text{m}$ (`dissipation_s_m`)
   - The outer $\max(0, \cdot)$ prevents tensile (attractive) forces during high-velocity rebound.

3. **Normal Force Vector**:
   $$F_n = f_n \cdot n$$

## Tangential Friction Formulation (Regularized Coulomb-Viscous)

Friction opposes tangential slip velocity and transitions smoothly between static, dynamic,
and viscous regimes:

1. **Tangential Slip Velocity**:
   $$v_t = v - (n^\top v) \cdot n, \quad s = \|v_t\|_2$$

2. **Velocity-Dependent Friction Coefficient**:
   $$\mu(s) = \mu_d + (\mu_s - \mu_d) \exp\left(-\left(\frac{s}{v_{\text{trans}}}\right)^2\right) + \mu_v \cdot s$$
   where:

   - $\mu_s \ge \mu_d \ge 0$ are static and dynamic friction coefficients
   - $\mu_v \ge 0$ is the viscous friction coefficient ($\text{s}/\text{m}$)
   - $v_{\text{trans}} > 0$ is the transition velocity threshold ($\text{m}/\text{s}$)

3. **Regularized Friction Force Vector**:
   For $s > 0$ and $f_n > 0$:
   $$F_t = -\frac{v_t}{s} \cdot \mu(s) \cdot f_n \cdot \tanh\left(\frac{s}{v_{\text{trans}}}\right)$$
   For $s = 0$ or $f_n = 0$, $F_t = 0$.

4. **Thermodynamic Dissipation**:
   The contact force $F = F_n + F_t$ satisfies:
   $$P_{\text{contact}} = F^\top v \le 0$$
   ensuring the contact interaction never adds spurious mechanical energy to the system.

## Kinematic Weld Closure Formulation (Dual-Grip Constraint)

The golf grip forms a closed kinematic loop between the pelvis, torso, lead arm, club, and trail arm.
The relative kinematic closure is modeled as a 6-DOF spatial weld constraint between the lead-hand
grip frame $T_{\text{lead}} \in \text{SE}(3)$ and trail-hand grip frame $T_{\text{trail}} \in \text{SE}(3)$:

1. **Spatial Constraint Residual**:
   $$\Phi(q) = \begin{bmatrix} p_{\text{lead}}(q) - p_{\text{trail}}(q) \\ \text{Log}\left(R_{\text{lead}}(q)^\top R_{\text{trail}}(q)\right) \end{bmatrix} = 0 \in \mathbb{R}^6$$

2. **Constraint Jacobian**:
   $$J_{\text{closure}}(q) = \frac{\partial \Phi}{\partial q} = \begin{bmatrix} J_{v,\text{lead}} - J_{v,\text{trail}} \\ J_{\omega,\text{lead}} - J_{\omega,\text{trail}} \end{bmatrix} \in \mathbb{R}^{6 \times n_v}$$

3. **Rank Condition**:
   At valid kinematic configurations, $\text{rank}(J_{\text{closure}}) = 6$, removing exactly 6 relative
   degrees of freedom between the two hands and preventing kinematic singularity or constraint redundancy.

## Cross-Engine Divergence Governance

Engine-specific implementations must register any physical or numerical departures in
`tests/integration/cross_engine/divergence_registry.yaml`. Every registered divergence requires:

- Unique identifier (`id`)
- Conformance check name (`check`)
- Discrepancy metric and unit (`metric`)
- Involved engines (`engines`)
- Approved maximum tolerance threshold (`tolerance`)
- Architectural rationale explaining the origin and justification (`rationale`)

Registration alone does not constitute passing a release gate; affected parity claims must
explicitly reference the registered divergence and obtain approval under MS-100 (#10374).
