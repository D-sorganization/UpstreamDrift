"""Comprehensive glossary data for biomechanics education system.

This module provides 500+ terms organized by category with multi-level
definitions for beginner, intermediate, advanced, and expert users.

Categories:
- dynamics: Forces, torques, energy
- kinematics: Motion, position, velocity
- anatomy: Bones, joints, muscles
- golf: Swing mechanics, equipment
- simulation: Physics engines, numerical methods
- data: File formats, measurement
- validation: Quality assurance, testing
- math: Mathematical concepts
- signal: Signal processing
- injury: Safety, biomechanical limits
"""

from __future__ import annotations

from src.shared.python.ai.types import ExpertiseLevel

# Type alias for glossary entry data
GlossaryData = dict[str, dict]


def get_dynamics_terms() -> GlossaryData:
    """Get dynamics-related glossary terms."""
    return {
        "inverse_dynamics": {
            "term": "Inverse Dynamics",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A way to figure out what forces caused a movement. "
                    "By watching how someone moved, we can calculate the "
                    "muscle forces they must have used."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "A computational method that calculates joint torques from "
                    "measured kinematics. Given positions, velocities, and accelerations, "
                    "inverse dynamics solves for the net torques at each joint."
                ),
                ExpertiseLevel.ADVANCED: (
                    "The solution of τ = M(q)q̈ + C(q,q̇)q̇ + g(q) for joint torques τ, "
                    "given measured generalized coordinates q and their derivatives."
                ),
                ExpertiseLevel.EXPERT: (
                    "Recursive Newton-Euler formulation: outward kinematics sweep "
                    "followed by inward dynamics sweep. O(n) complexity for n bodies."
                ),
            },
            "formula": "τ = M(q)q̈ + C(q,q̇) + g(q)",
            "units": "N·m",
            "related_terms": ["forward_dynamics", "joint_torque", "equations_of_motion"],
        },
        "forward_dynamics": {
            "term": "Forward Dynamics",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Predicting how something will move when you apply forces. "
                    "Like pushing a swing - you know how hard you pushed, "
                    "and forward dynamics tells you how the swing will move."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Calculating motion from applied forces. Given joint torques, "
                    "forward dynamics computes the resulting accelerations."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Solving q̈ = M(q)⁻¹ [τ - C(q,q̇)q̇ - g(q)] for accelerations. "
                    "Requires mass matrix inversion via Cholesky decomposition."
                ),
            },
            "formula": "q̈ = M(q)⁻¹ [τ - C(q,q̇) - g(q)]",
            "units": "rad/s²",
            "related_terms": ["inverse_dynamics", "simulation", "equations_of_motion"],
        },
        "joint_torque": {
            "term": "Joint Torque",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The rotational force at a joint, like your elbow or knee. "
                    "When you curl a weight, your bicep creates torque at your elbow."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "The net rotational force acting about a joint axis. "
                    "Represents the sum of all muscle, ligament, and contact forces."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Generalized force corresponding to a rotational degree of freedom. "
                    "In inverse dynamics, the torque required to produce observed motion."
                ),
            },
            "units": "N·m",
            "related_terms": ["inverse_dynamics", "muscle_force", "moment_of_inertia"],
        },
        "equations_of_motion": {
            "term": "Equations of Motion",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Mathematical rules that describe how things move. "
                    "They connect forces to motion."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Differential equations relating forces, masses, and accelerations. "
                    "For multibody systems: M(q)q̈ + C(q,q̇)q̇ + g(q) = τ."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Lagrangian or Newton-Euler derived ODEs. The mass matrix M(q) is "
                    "configuration-dependent. Coriolis C and gravity g terms complete the dynamics."
                ),
            },
            "formula": "M(q)q̈ + C(q,q̇)q̇ + g(q) = τ",
            "related_terms": ["inverse_dynamics", "forward_dynamics", "lagrangian"],
        },
        "lagrangian": {
            "term": "Lagrangian",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A mathematical formula that helps us derive equations of motion. "
                    "It's based on the difference between kinetic and potential energy."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "L = T - V, where T is kinetic energy and V is potential energy. "
                    "The Euler-Lagrange equations give the equations of motion."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Scalar function L(q, q̇) = T(q, q̇) - V(q). Equations of motion: "
                    "d/dt(∂L/∂q̇) - ∂L/∂q = τ. Basis for analytical mechanics."
                ),
            },
            "formula": "L = T - V",
            "related_terms": ["equations_of_motion", "kinetic_energy", "potential_energy"],
        },
        "kinetic_energy": {
            "term": "Kinetic Energy",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Energy of motion. The faster something moves, the more kinetic energy it has."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Energy due to motion: KE = ½mv² for translation, ½Iω² for rotation. "
                    "Total kinetic energy is the sum of all body segments."
                ),
                ExpertiseLevel.ADVANCED: (
                    "T = ½q̇ᵀM(q)q̇ in generalized coordinates. Configuration-dependent "
                    "mass matrix M(q) captures coupled inertias."
                ),
            },
            "formula": "T = ½q̇ᵀM(q)q̇",
            "units": "J (Joules)",
            "related_terms": ["potential_energy", "lagrangian", "mass_matrix"],
        },
        "potential_energy": {
            "term": "Potential Energy",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Stored energy due to position or configuration. "
                    "A raised arm has gravitational potential energy."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Energy stored in configuration: V = mgh for gravity, "
                    "½kx² for springs. Converted to kinetic energy during motion."
                ),
                ExpertiseLevel.ADVANCED: (
                    "V(q) is configuration-dependent. Gradient ∂V/∂q gives generalized forces. "
                    "Includes gravitational, elastic, and constraint potentials."
                ),
            },
            "formula": "V = mgh (gravitational)",
            "units": "J (Joules)",
            "related_terms": ["kinetic_energy", "lagrangian", "gravity_vector"],
        },
        "mass_matrix": {
            "term": "Mass Matrix",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A table of numbers describing how mass is distributed in a system. "
                    "It affects how the system responds to forces."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Symmetric positive-definite matrix M(q) relating accelerations to forces. "
                    "Encodes inertial properties of all body segments."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Configuration-dependent n×n matrix where n is DOF. "
                    "Computed via composite rigid body algorithm. Cholesky factorization for solving."
                ),
            },
            "related_terms": ["equations_of_motion", "moment_of_inertia", "generalized_coordinates"],
        },
        "coriolis_forces": {
            "term": "Coriolis Forces",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Forces that appear when things rotate. They make moving objects curve."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Velocity-dependent forces in rotating systems. In multibody dynamics, "
                    "the C(q,q̇)q̇ term includes Coriolis and centrifugal effects."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Christoffel symbols of the first kind define the Coriolis matrix. "
                    "C(q,q̇) satisfies skew-symmetry: q̇ᵀ(Ṁ - 2C)q̇ = 0."
                ),
            },
            "related_terms": ["equations_of_motion", "centrifugal_force", "mass_matrix"],
        },
        "centrifugal_force": {
            "term": "Centrifugal Force",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The outward push you feel on a merry-go-round. "
                    "It's an apparent force due to rotation."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Apparent outward force in rotating reference frames. "
                    "Magnitude: mω²r where r is distance from rotation axis."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Inertial force in non-inertial frame. Combined with Coriolis in "
                    "the C(q,q̇)q̇ term of equations of motion."
                ),
            },
            "formula": "F = mω²r",
            "units": "N",
            "related_terms": ["coriolis_forces", "angular_velocity", "inertial_frame"],
        },
        "gravity_vector": {
            "term": "Gravity Vector",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The direction and strength of gravity pulling on each body part."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "The g(q) term in equations of motion. Represents gravitational forces "
                    "on all segments in generalized coordinates."
                ),
                ExpertiseLevel.ADVANCED: (
                    "g(q) = ∂V_grav/∂q where V_grav is gravitational potential. "
                    "Computed via recursive algorithm traversing kinematic tree."
                ),
            },
            "units": "N (forces) or N·m (torques)",
            "related_terms": ["potential_energy", "equations_of_motion", "center_of_mass"],
        },
        "moment_of_inertia": {
            "term": "Moment of Inertia",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How hard it is to spin something. A spread-out mass is harder to spin "
                    "than a compact one."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Rotational analog of mass. I = Σmᵢrᵢ² for point masses. "
                    "Determines angular acceleration from applied torque."
                ),
                ExpertiseLevel.ADVANCED: (
                    "3×3 inertia tensor about body frame. Principal axes diagonalize the tensor. "
                    "Parallel axis theorem for off-center rotations."
                ),
            },
            "formula": "τ = Iα",
            "units": "kg·m²",
            "related_terms": ["joint_torque", "angular_acceleration", "mass_matrix"],
        },
        "angular_momentum": {
            "term": "Angular Momentum",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The 'spin' a rotating object has. Hard to stop once it's spinning."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "L = Iω for rotation, or r × p for translation. "
                    "Conserved in absence of external torques."
                ),
                ExpertiseLevel.ADVANCED: (
                    "L = Σ(Iᵢωᵢ + rᵢ × mᵢvᵢ) for multibody system. "
                    "Transfer between segments enables kinetic chain."
                ),
            },
            "formula": "L = Iω",
            "units": "kg·m²/s",
            "related_terms": ["moment_of_inertia", "angular_velocity", "kinetic_chain"],
        },
        "work": {
            "term": "Work",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Energy transferred when a force moves something. "
                    "Pushing a car transfers work to the car."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "W = F·d for constant force, or integral of F·ds along path. "
                    "For rotation: W = τ·θ."
                ),
                ExpertiseLevel.ADVANCED: (
                    "W = ∫τᵀq̇ dt for generalized coordinates. "
                    "Joint work indicates energy contribution of each DOF."
                ),
            },
            "formula": "W = F·d or W = τ·θ",
            "units": "J (Joules)",
            "related_terms": ["power", "energy", "force"],
        },
        "power": {
            "term": "Power",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How fast work is done. More power means doing work faster."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Rate of energy transfer: P = dW/dt = F·v for translation, τ·ω for rotation."
                ),
                ExpertiseLevel.ADVANCED: (
                    "P = τᵀq̇ in generalized coordinates. Joint power flow analysis "
                    "reveals energy generation and absorption patterns."
                ),
            },
            "formula": "P = τ·ω",
            "units": "W (Watts)",
            "related_terms": ["work", "joint_torque", "angular_velocity"],
        },
        "impulse": {
            "term": "Impulse",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A sudden force applied briefly. Like hitting a golf ball."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "J = ∫F dt = Δp. Change in momentum equals impulse. "
                    "Critical for impact analysis."
                ),
                ExpertiseLevel.ADVANCED: (
                    "For constrained impacts: impulse computed from velocity jump conditions "
                    "and coefficient of restitution."
                ),
            },
            "formula": "J = ∫F dt = Δp",
            "units": "N·s",
            "related_terms": ["momentum", "impact", "force"],
        },
        "momentum": {
            "term": "Momentum",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Mass times velocity. Heavy, fast things have lots of momentum."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Linear momentum p = mv. Conserved in isolated systems. "
                    "Rate of change equals net force."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Generalized momentum pᵢ = ∂L/∂q̇ᵢ. Hamiltonian formulation uses "
                    "(q, p) as state variables."
                ),
            },
            "formula": "p = mv",
            "units": "kg·m/s",
            "related_terms": ["impulse", "angular_momentum", "kinetic_energy"],
        },
        "contact_force": {
            "term": "Contact Force",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Force between touching objects. Your feet push on the ground, "
                    "the ground pushes back."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Force at contact points including normal and friction components. "
                    "Modeled as constraint forces or compliant contacts."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Contact dynamics via LCP (Linear Complementarity Problem) or "
                    "compliant penalty methods. Coulomb friction cone constraints."
                ),
            },
            "related_terms": ["ground_reaction_force", "friction", "constraint_force"],
        },
        "friction": {
            "term": "Friction",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Resistance to sliding between surfaces. Helps you grip the ground."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Tangential contact force opposing motion. Static (μs) and dynamic (μd) "
                    "coefficients. Coulomb model: Ff ≤ μFn."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Friction cone constraint in contact dynamics. Pyramid approximation "
                    "for linear complementarity. Stribeck effect at low velocities."
                ),
            },
            "formula": "Ff ≤ μFn",
            "related_terms": ["contact_force", "ground_reaction_force", "slip"],
        },
        "constraint_force": {
            "term": "Constraint Force",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Forces that keep things connected or on a path. "
                    "Like a hinge keeping a door attached."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Internal forces maintaining kinematic constraints. "
                    "Do no work on constrained motion. Computed as Lagrange multipliers."
                ),
                ExpertiseLevel.ADVANCED: (
                    "λ in: M(q)q̈ + C(q,q̇)q̇ + g(q) = τ + Jᵀλ. "
                    "Baumgarte stabilization prevents constraint drift."
                ),
            },
            "related_terms": ["joint", "lagrange_multiplier", "holonomic_constraint"],
        },
        "damping": {
            "term": "Damping",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Resistance that slows things down. Like friction in joints."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Velocity-dependent dissipative force. Viscous damping: F = -bv. "
                    "Removes energy from the system."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Rayleigh damping: D = αM + βK. Critical damping ratio ζ = c/(2√(km)). "
                    "Affects natural frequency response."
                ),
            },
            "formula": "F = -bv (viscous)",
            "units": "N·s/m",
            "related_terms": ["stiffness", "natural_frequency", "oscillation"],
        },
        "stiffness": {
            "term": "Stiffness",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How hard it is to deform something. Stiffer things resist bending more."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Force per unit displacement: k = F/x. For joints, rotational stiffness: "
                    "τ = kθ. Units: N/m or N·m/rad."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Stiffness matrix K = ∂²V/∂q². Positive definiteness ensures stability. "
                    "Impedance control uses programmable stiffness."
                ),
            },
            "formula": "k = F/x",
            "units": "N/m or N·m/rad",
            "related_terms": ["damping", "compliance", "impedance"],
        },
    }


def get_kinematics_terms() -> GlossaryData:
    """Get kinematics-related glossary terms."""
    return {
        "kinematics": {
            "term": "Kinematics",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The study of motion without worrying about forces. "
                    "It's about describing HOW things move."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "The branch of mechanics describing motion through degrees of "
                    "freedom without considering forces."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Configuration space analysis: generalized coordinates q(t), "
                    "velocities q̇(t), and accelerations q̈(t)."
                ),
            },
            "related_terms": ["dynamics", "motion_capture", "joint_angles"],
        },
        "position": {
            "term": "Position",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Where something is in space. Can be described by x, y, z coordinates."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Location in 3D space or configuration space. For rigid bodies, "
                    "includes both translation (3 DOF) and orientation (3 DOF)."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Element of SE(3) for rigid body, or q ∈ Q for configuration space. "
                    "Position kinematics: x = f(q) mapping joints to Cartesian."
                ),
            },
            "units": "m (meters)",
            "related_terms": ["velocity", "acceleration", "configuration_space"],
        },
        "velocity": {
            "term": "Velocity",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How fast something is moving and in what direction. Speed with direction."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Rate of change of position: v = dx/dt. For rotation: angular velocity ω. "
                    "Vector quantity with magnitude and direction."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Tangent vector to configuration manifold. Jacobian relates: "
                    "ẋ = J(q)q̇. Body vs spatial velocity representations."
                ),
            },
            "formula": "v = dx/dt",
            "units": "m/s",
            "related_terms": ["position", "acceleration", "jacobian"],
        },
        "acceleration": {
            "term": "Acceleration",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How quickly velocity changes. Speeding up or slowing down."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Rate of change of velocity: a = dv/dt = d²x/dt². "
                    "Key input for inverse dynamics calculations."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Second derivative: q̈. In Cartesian: ẍ = J(q)q̈ + J̇(q,q̇)q̇. "
                    "Often computed by numerical differentiation with filtering."
                ),
            },
            "formula": "a = dv/dt",
            "units": "m/s²",
            "related_terms": ["velocity", "inverse_dynamics", "differentiation"],
        },
        "angular_velocity": {
            "term": "Angular Velocity",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How fast something is spinning. Measured in degrees or radians per second."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Rate of rotation: ω = dθ/dt. Vector pointing along rotation axis. "
                    "Magnitude is rotation speed."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Body angular velocity ω^b vs spatial ω^s. Related by: ω^s = R·ω^b. "
                    "Skew-symmetric matrix [ω]× for cross products."
                ),
            },
            "formula": "ω = dθ/dt",
            "units": "rad/s",
            "related_terms": ["angular_acceleration", "rotation_matrix", "joint_angles"],
        },
        "angular_acceleration": {
            "term": "Angular Acceleration",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How quickly rotation speed changes. Spinning faster or slower."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Rate of change of angular velocity: α = dω/dt. "
                    "Related to torque by: τ = Iα."
                ),
                ExpertiseLevel.ADVANCED: (
                    "α = dω/dt in body frame. Time derivative of ω requires careful handling "
                    "of rotating frames."
                ),
            },
            "formula": "α = dω/dt",
            "units": "rad/s²",
            "related_terms": ["angular_velocity", "joint_torque", "moment_of_inertia"],
        },
        "joint_angles": {
            "term": "Joint Angles",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The angles at your joints - how bent your elbow or knee is."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Rotation angles between connected segments. Often expressed in "
                    "anatomical planes: flexion/extension, ab/adduction, rotation."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Generalized coordinates q for revolute joints. Euler angles or "
                    "quaternions for 3-DOF joints. Gimbal lock considerations."
                ),
            },
            "units": "degrees or radians",
            "related_terms": ["kinematics", "euler_angles", "quaternion"],
        },
        "euler_angles": {
            "term": "Euler Angles",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Three angles that describe any 3D rotation. Like roll, pitch, and yaw."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Sequence of three rotations (e.g., ZYX, XYZ) representing orientation. "
                    "Convention matters - different sequences give different angles."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Parameterization of SO(3). Gimbal lock at specific configurations. "
                    "Singularity-free alternatives: quaternions, rotation vectors."
                ),
            },
            "related_terms": ["rotation_matrix", "quaternion", "gimbal_lock"],
        },
        "quaternion": {
            "term": "Quaternion",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A special way to represent 3D rotations using four numbers. "
                    "Avoids some problems that angles have."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "4D extension of complex numbers for rotation: q = [w, x, y, z]. "
                    "Singularity-free and efficient for composition."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Unit quaternion q ∈ S³ double-covers SO(3). Composition: q₁⊗q₂. "
                    "SLERP for smooth interpolation."
                ),
            },
            "related_terms": ["euler_angles", "rotation_matrix", "slerp"],
        },
        "rotation_matrix": {
            "term": "Rotation Matrix",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A 3×3 table of numbers that describes how to rotate something."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "R ∈ SO(3): orthogonal matrix with det(R) = 1. Transforms vectors: "
                    "v' = Rv. Nine numbers, three constraints (6 DOF)."
                ),
                ExpertiseLevel.ADVANCED: (
                    "R = exp([ω]×θ) via Rodrigues formula. Lie group SO(3) with Lie algebra so(3). "
                    "Computational cost higher than quaternions."
                ),
            },
            "related_terms": ["quaternion", "euler_angles", "transformation_matrix"],
        },
        "jacobian": {
            "term": "Jacobian",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A mathematical tool that relates joint movements to end-effector movements."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Matrix relating joint velocities to end-effector velocities: ẋ = J(q)q̇. "
                    "Used for inverse kinematics and force mapping."
                ),
                ExpertiseLevel.ADVANCED: (
                    "J = ∂x/∂q. Singularities where rank(J) < min(m,n). "
                    "Pseudoinverse J† for redundant systems."
                ),
            },
            "formula": "ẋ = J(q)q̇",
            "related_terms": ["inverse_kinematics", "manipulability", "singularity"],
        },
        "forward_kinematics": {
            "term": "Forward Kinematics",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Calculating where body parts end up given the joint angles."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Computing end-effector pose from joint angles: x = f(q). "
                    "Product of transformation matrices along kinematic chain."
                ),
                ExpertiseLevel.ADVANCED: (
                    "SE(3) transformations via DH parameters or spatial algebra. "
                    "Product of exponentials: T = Πexp(ξᵢθᵢ)."
                ),
            },
            "related_terms": ["inverse_kinematics", "transformation_matrix", "kinematic_chain"],
        },
        "inverse_kinematics": {
            "term": "Inverse Kinematics",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Finding what joint angles are needed to reach a target position."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Computing joint angles from desired end-effector pose: q = f⁻¹(x). "
                    "Often has multiple solutions or no solution."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Numerical: Jacobian pseudoinverse with damped least squares. "
                    "Analytical: closed-form for specific kinematic structures."
                ),
            },
            "related_terms": ["forward_kinematics", "jacobian", "redundancy"],
        },
        "degrees_of_freedom": {
            "term": "Degrees of Freedom (DOF)",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The number of independent ways something can move. Your elbow has 1 DOF."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Number of independent coordinates needed to specify configuration. "
                    "A free rigid body has 6 DOF (3 translation + 3 rotation)."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Dimension of configuration manifold Q. Grübler formula for mechanisms: "
                    "DOF = 6(n-1) - Σ(6-fᵢ) where fᵢ is joint DOF."
                ),
            },
            "related_terms": ["generalized_coordinates", "constraint", "joint"],
        },
        "generalized_coordinates": {
            "term": "Generalized Coordinates",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A minimal set of numbers that fully describe a system's position."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Independent variables q that uniquely define system configuration. "
                    "Typically joint angles for serial chains."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Coordinates on configuration manifold Q. Constrained systems: "
                    "minimal coordinates eliminate constraints."
                ),
            },
            "related_terms": ["degrees_of_freedom", "configuration_space", "joint_angles"],
        },
        "configuration_space": {
            "term": "Configuration Space",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "All possible positions a system can be in."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "The space Q of all possible configurations. Each point represents "
                    "a unique system pose."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Manifold Q with dimension = DOF. For n-link robot: Q = (S¹)ⁿ or "
                    "product of SO(3) for spherical joints."
                ),
            },
            "related_terms": ["generalized_coordinates", "degrees_of_freedom", "workspace"],
        },
        "workspace": {
            "term": "Workspace",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "All the positions an arm or tool can reach."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Set of all reachable end-effector positions/orientations. "
                    "Limited by joint ranges and kinematic structure."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Image of forward kinematics map: W = f(Q). Reachable vs dexterous workspace. "
                    "Singularity boundaries define workspace limits."
                ),
            },
            "related_terms": ["forward_kinematics", "range_of_motion", "singularity"],
        },
        "range_of_motion": {
            "term": "Range of Motion (ROM)",
            "category": "kinematics",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How far a joint can move. Like how far you can bend your knee."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Angular or linear limits of joint movement. Varies by individual, "
                    "affected by age, flexibility, injury."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Joint limits qₘᵢₙ ≤ q ≤ qₘₐₓ as inequality constraints. "
                    "Active vs passive ROM. Barrier methods for limit avoidance."
                ),
            },
            "units": "degrees",
            "related_terms": ["joint_angles", "flexibility", "workspace"],
        },
    }


def get_anatomy_terms() -> GlossaryData:
    """Get anatomy-related glossary terms."""
    return {
        "pelvis": {
            "term": "Pelvis",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The bony structure at the base of your spine that connects "
                    "your legs to your torso."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Basin-shaped bone structure comprising ilium, ischium, and pubis. "
                    "Key for hip joint and trunk rotation in golf."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Reference segment for biomechanical analysis. Contains ASIS/PSIS "
                    "landmarks for marker placement. Rotates ~45° in backswing."
                ),
            },
            "related_terms": ["hip_joint", "lumbar_spine", "x_factor"],
        },
        "thorax": {
            "term": "Thorax",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Your chest and upper back, including the ribcage."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "The upper trunk containing the ribcage and thoracic spine. "
                    "Rotates separately from pelvis to create X-factor."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Modeled as rigid segment attached to lumbar spine. "
                    "T8 and acromion markers define orientation. ~90° rotation at top of backswing."
                ),
            },
            "related_terms": ["pelvis", "x_factor", "spine"],
        },
        "spine": {
            "term": "Spine",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The column of bones running down your back, from neck to pelvis."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Vertebral column with cervical (7), thoracic (12), lumbar (5), "
                    "sacral, and coccygeal regions. Primary axis for trunk rotation."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Often modeled as 2-3 segments in biomechanics. Intervertebral joints "
                    "provide limited ROM per level, cumulative flexibility."
                ),
            },
            "related_terms": ["lumbar_spine", "thorax", "vertebra"],
        },
        "lumbar_spine": {
            "term": "Lumbar Spine",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The lower back - the part of your spine between your ribs and pelvis."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Five vertebrae (L1-L5) providing trunk flexion/extension "
                    "and lateral bend. Limited rotation capacity."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Primary sagittal plane motion. Each level: ~15° flexion, ~5° extension. "
                    "Rotation mostly from facet joints. Key injury site in golf."
                ),
            },
            "related_terms": ["spine", "pelvis", "intervertebral_disc"],
        },
        "hip_joint": {
            "term": "Hip Joint",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Ball-and-socket joint connecting your leg to your pelvis."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Synovial ball-and-socket joint with 3 DOF. Femoral head articulates "
                    "with acetabulum. Major power source in golf swing."
                ),
                ExpertiseLevel.ADVANCED: (
                    "ROM: flexion ~120°, extension ~30°, internal/external rotation ~45° each. "
                    "Hip-spine separation crucial for X-factor."
                ),
            },
            "related_terms": ["pelvis", "femur", "ball_and_socket"],
        },
        "shoulder_joint": {
            "term": "Shoulder Joint",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Where your arm connects to your body. Very flexible joint."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Glenohumeral joint: ball-and-socket with greatest ROM in body. "
                    "Stabilized by rotator cuff muscles."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Shoulder complex includes GH, AC, SC joints and scapulothoracic interface. "
                    "5 DOF effective motion. High injury risk in golf."
                ),
            },
            "related_terms": ["humerus", "scapula", "rotator_cuff"],
        },
        "elbow_joint": {
            "term": "Elbow Joint",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The hinge in the middle of your arm."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Hinge joint (1 DOF flexion/extension) plus radioulnar joint "
                    "for forearm rotation (pronation/supination)."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Humeroradial and humeroulnar articulations. ROM: 0-145° flexion. "
                    "Pronation/supination ~75° each. Golfer's elbow common injury."
                ),
            },
            "related_terms": ["humerus", "radius", "ulna"],
        },
        "wrist_joint": {
            "term": "Wrist Joint",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Where your hand connects to your forearm."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Complex of radiocarpal and midcarpal joints. 2 DOF: "
                    "flexion/extension and radial/ulnar deviation."
                ),
                ExpertiseLevel.ADVANCED: (
                    "8 carpal bones in proximal and distal rows. Wrist hinge critical for "
                    "club head speed. Peak velocity ~1000°/s at impact."
                ),
            },
            "related_terms": ["radius", "carpal_bones", "grip"],
        },
        "scapula": {
            "term": "Scapula",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Your shoulder blade - the flat bone on your upper back."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Triangular bone connecting arm to trunk. Glides on thorax, "
                    "providing additional shoulder mobility."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Scapulothoracic rhythm: 2:1 GH to ST motion ratio. "
                    "Protraction/retraction, elevation/depression, rotation."
                ),
            },
            "related_terms": ["shoulder_joint", "clavicle", "thorax"],
        },
        "femur": {
            "term": "Femur",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Your thigh bone - the largest bone in your body."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Long bone from hip to knee. Head articulates with pelvis, "
                    "condyles with tibia. Major moment arm for hip muscles."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Length ~25% of height. Neck-shaft angle ~125°. "
                    "Anteversion ~15°. Key segment for ground reaction force transfer."
                ),
            },
            "related_terms": ["hip_joint", "knee_joint", "quadriceps"],
        },
        "humerus": {
            "term": "Humerus",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Your upper arm bone, from shoulder to elbow."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Long bone connecting shoulder to elbow. Head articulates with scapula, "
                    "condyles with radius and ulna."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Length ~19% of height. Attachment for rotator cuff, deltoid, "
                    "biceps, triceps. Humeral rotation key in swing."
                ),
            },
            "related_terms": ["shoulder_joint", "elbow_joint", "biceps"],
        },
        "muscle": {
            "term": "Muscle",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Tissue that contracts to produce movement."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Contractile tissue generating force and motion. Types: skeletal, "
                    "cardiac, smooth. Skeletal muscles are under voluntary control."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Force-length-velocity properties. Hill-type models: F = f(l)·f(v)·a. "
                    "Pennation angle affects force transmission."
                ),
            },
            "related_terms": ["muscle_force", "tendon", "activation"],
        },
        "tendon": {
            "term": "Tendon",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Tough tissue connecting muscles to bones."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Fibrous connective tissue transmitting muscle force to bone. "
                    "Stores elastic energy during stretch."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Viscoelastic behavior: stress-strain curve with toe region. "
                    "Series elastic element in Hill models. Strain rate dependent."
                ),
            },
            "related_terms": ["muscle", "ligament", "elastic_energy"],
        },
        "ligament": {
            "term": "Ligament",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Tissue connecting bones to other bones at joints."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Fibrous tissue stabilizing joints. Limits range of motion. "
                    "Can be injured by excessive force or range."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Collagen fiber bundles with nonlinear stress-strain properties. "
                    "Contribute to passive joint stiffness. Proprioceptive feedback."
                ),
            },
            "related_terms": ["joint", "tendon", "sprain"],
        },
        "cartilage": {
            "term": "Cartilage",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Smooth, slippery tissue covering the ends of bones in joints."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Avascular connective tissue providing low-friction articulation. "
                    "Types: hyaline, fibrocartilage, elastic."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Biphasic poroelastic material: solid matrix + fluid. "
                    "Load-dependent friction coefficient. Limited regeneration capacity."
                ),
            },
            "related_terms": ["joint", "arthritis", "meniscus"],
        },
        "center_of_mass": {
            "term": "Center of Mass (CoM)",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The balance point of your body or any object."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Point where total mass can be considered concentrated. "
                    "Whole-body CoM changes with posture."
                ),
                ExpertiseLevel.ADVANCED: (
                    "CoM = Σmᵢrᵢ/Σmᵢ. Segment CoM from regression equations. "
                    "External forces act as if applied at CoM for translation."
                ),
            },
            "related_terms": ["moment_of_inertia", "balance", "segment"],
        },
        "segment": {
            "term": "Body Segment",
            "category": "anatomy",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A body part treated as a single unit, like your forearm or thigh."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Rigid body approximation of limb section. Has mass, length, "
                    "center of mass, and moment of inertia."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Inertial properties from regression (Dempster, de Leva) or imaging. "
                    "Kinematic chain formed by connected segments."
                ),
            },
            "related_terms": ["center_of_mass", "moment_of_inertia", "kinematic_chain"],
        },
    }


def get_golf_terms() -> GlossaryData:
    """Get golf-specific glossary terms."""
    return {
        "kinetic_chain": {
            "term": "Kinetic Chain",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How energy flows through your body during a golf swing. "
                    "Power starts from your legs and moves to the club."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "The sequential activation and energy transfer from proximal "
                    "to distal segments. In golf: legs → trunk → arms → club."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Proximal-to-distal sequence for angular momentum transfer. "
                    "Peak velocities occur sequentially. Timing disruption reduces speed ~20%."
                ),
            },
            "related_terms": ["x_factor", "ground_reaction_force", "club_head_speed"],
        },
        "x_factor": {
            "term": "X-Factor",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The twist between your hips and shoulders at the top of the swing."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Difference between thorax and pelvis rotation at top of backswing. "
                    "Associated with increased club head speed."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Angular separation in transverse plane. Elite PGA: 45-55°. "
                    "X-Factor Stretch (additional separation into downswing) correlates r=0.7 with ball speed."
                ),
            },
            "units": "degrees",
            "related_terms": ["kinetic_chain", "thorax", "pelvis"],
        },
        "club_head_speed": {
            "term": "Club Head Speed",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How fast the golf club is moving when it hits the ball."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Velocity of club head at impact. Primary determinant of ball speed. "
                    "Tour average: ~112 mph for drivers."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Resultant of angular velocities: v = Σωᵢ × rᵢ. "
                    "Wrist uncocking contributes ~50% of total. Smash factor relates to ball speed."
                ),
            },
            "units": "mph or m/s",
            "related_terms": ["ball_speed", "smash_factor", "kinetic_chain"],
        },
        "ball_speed": {
            "term": "Ball Speed",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How fast the ball leaves the club face after impact."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Initial velocity of ball post-impact. Determined by club head speed, "
                    "impact efficiency, and club face properties."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Ball speed = CHS × smash factor. Coefficient of restitution limits. "
                    "USGA max COR = 0.83. Tour driver avg: ~167 mph."
                ),
            },
            "units": "mph or m/s",
            "related_terms": ["club_head_speed", "smash_factor", "launch_angle"],
        },
        "smash_factor": {
            "term": "Smash Factor",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How efficiently energy transfers from club to ball. Higher is better."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Ratio of ball speed to club head speed. Driver max ~1.5. "
                    "Indicates strike quality."
                ),
                ExpertiseLevel.ADVANCED: (
                    "SF = vball/vclub. Related to COR but includes gear effect and contact location. "
                    "Center strike maximizes SF."
                ),
            },
            "formula": "SF = ball_speed / club_head_speed",
            "related_terms": ["ball_speed", "club_head_speed", "impact"],
        },
        "launch_angle": {
            "term": "Launch Angle",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The angle the ball takes off at after impact."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Initial trajectory angle relative to horizontal. Optimal depends on "
                    "club head speed: faster swings need lower launch."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Determined by dynamic loft, attack angle, and face angle. "
                    "Optimal for max carry: LA ≈ 0.5 × spin loft for given ball speed."
                ),
            },
            "units": "degrees",
            "related_terms": ["ball_speed", "spin_rate", "attack_angle"],
        },
        "spin_rate": {
            "term": "Spin Rate",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How fast the ball is spinning after you hit it."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Ball rotation rate in RPM. Affects trajectory curvature and landing behavior. "
                    "Backspin creates lift."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Function of spin loft and friction. Magnus force: F ∝ ω × v. "
                    "Optimal driver spin ~2000-2500 rpm for max distance."
                ),
            },
            "units": "rpm",
            "related_terms": ["launch_angle", "carry_distance", "spin_axis"],
        },
        "attack_angle": {
            "term": "Attack Angle",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Whether you're hitting up or down on the ball at impact."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Vertical club head path angle at impact. Positive = ascending (hit up), "
                    "negative = descending (hit down)."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Driver: +2 to +5° optimal for distance. Irons: -3 to -6° for compression. "
                    "Affects dynamic loft and spin loft."
                ),
            },
            "units": "degrees",
            "related_terms": ["launch_angle", "club_path", "dynamic_loft"],
        },
        "club_path": {
            "term": "Club Path",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The direction the club head is moving through impact."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Horizontal direction of club head at impact relative to target line. "
                    "In-to-out is positive, out-to-in is negative."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Combined with face angle determines initial direction and curve. "
                    "New ball flight laws: ~75% face, ~25% path for direction."
                ),
            },
            "units": "degrees",
            "related_terms": ["face_angle", "attack_angle", "swing_plane"],
        },
        "face_angle": {
            "term": "Face Angle",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Where the club face is pointing at impact."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Club face orientation relative to target line. Open (pointing right for RH) "
                    "or closed. Primary determinant of start direction."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Face-to-path differential creates spin axis tilt (curve). "
                    "D-plane concept: face + path + attack = complete impact."
                ),
            },
            "units": "degrees",
            "related_terms": ["club_path", "spin_axis", "d_plane"],
        },
        "swing_plane": {
            "term": "Swing Plane",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The tilted circle your club travels around during the swing."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Inclined plane containing club head path. Single-plane vs two-plane swings. "
                    "Determined by posture and arm structure."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Ben Hogan's pane of glass concept. Functional plane angles vary: "
                    "backswing vs downswing. Video analysis: 2D projection of 3D motion."
                ),
            },
            "units": "degrees (from horizontal)",
            "related_terms": ["club_path", "posture", "downswing"],
        },
        "backswing": {
            "term": "Backswing",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The part of the swing where you take the club back away from the ball."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Phase from address to top of swing. Stores elastic energy in muscles "
                    "and creates X-factor stretch."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Weight shift, hip rotation (~45°), shoulder rotation (~90°). "
                    "Wrist cock sets club. SSC (stretch-shortening cycle) potentiation."
                ),
            },
            "related_terms": ["downswing", "x_factor", "wrist_cock"],
        },
        "downswing": {
            "term": "Downswing",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Swinging the club back down toward the ball."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "From top to impact. Initiated by lower body. Kinetic chain unwinds "
                    "proximal to distal."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Hip rotation leads shoulder rotation by 20-40ms. Peak angular velocities: "
                    "pelvis ~400°/s, thorax ~600°/s, arm ~1000°/s, club ~2000°/s."
                ),
            },
            "related_terms": ["backswing", "impact", "kinetic_chain"],
        },
        "impact": {
            "term": "Impact",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The moment the club hits the ball."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Club-ball collision lasting ~0.5ms. All launch conditions determined here. "
                    "Peak forces ~8000-10000 N."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Hertzian contact model inadequate - viscoelastic ball deformation. "
                    "Coefficient of restitution is velocity-dependent."
                ),
            },
            "related_terms": ["club_head_speed", "ball_speed", "smash_factor"],
        },
        "follow_through": {
            "term": "Follow-Through",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "What happens after you hit the ball - finishing your swing."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Post-impact deceleration phase. Controlled by antagonist muscles. "
                    "Indicator of swing balance."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Eccentric muscle action for deceleration. Peak eccentric loads in lead hip, "
                    "trail shoulder. Injury mechanism."
                ),
            },
            "related_terms": ["impact", "deceleration", "balance"],
        },
        "ground_reaction_force": {
            "term": "Ground Reaction Force",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The push-back from the ground when you push against it."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Force from ground measured by force plates. Components: vertical, "
                    "anterior-posterior, medio-lateral."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Vertical Fz: 1.2-1.5 BW at impact. Horizontal forces for rotation. "
                    "Center of pressure trajectory indicates weight shift."
                ),
            },
            "units": "N or BW",
            "related_terms": ["force_plate", "center_of_pressure", "weight_shift"],
        },
        "weight_shift": {
            "term": "Weight Shift",
            "category": "golf",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Moving your weight from one foot to the other during the swing."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Transfer of ground reaction force between feet. Trail foot in backswing, "
                    "lead foot at impact."
                ),
                ExpertiseLevel.ADVANCED: (
                    "COP trajectory analysis. At impact: 80-90% on lead foot. "
                    "Premature shift ('early extension') reduces power."
                ),
            },
            "related_terms": ["ground_reaction_force", "center_of_pressure", "balance"],
        },
    }


def get_simulation_terms() -> GlossaryData:
    """Get simulation and physics engine terms."""
    return {
        "physics_engine": {
            "term": "Physics Engine",
            "category": "simulation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Software that simulates how things move and interact."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Library implementing rigid body dynamics, contacts, and constraints."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Numerical solver for constrained multibody dynamics. "
                    "Time-stepping via symplectic Euler or semi-implicit RK."
                ),
            },
            "related_terms": ["mujoco", "drake", "pinocchio"],
        },
        "mujoco": {
            "term": "MuJoCo",
            "category": "simulation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A physics engine especially good at handling contacts."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Multi-Joint dynamics with Contact. Optimized for robotics and biomechanics."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Recursive dynamics with complementarity contact. PGS solver. "
                    "GPU-accelerated with MJX."
                ),
            },
            "see_also": ["https://mujoco.org"],
            "related_terms": ["physics_engine", "drake", "pinocchio"],
        },
        "drake": {
            "term": "Drake",
            "category": "simulation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A physics engine good at optimization and mathematical rigor."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "MIT-developed toolkit for dynamics, control, and optimization."
                ),
                ExpertiseLevel.ADVANCED: (
                    "AutoDiff-enabled multibody dynamics. Direct collocation and SNOPT."
                ),
            },
            "see_also": ["https://drake.mit.edu"],
            "related_terms": ["physics_engine", "mujoco", "pinocchio"],
        },
        "pinocchio": {
            "term": "Pinocchio",
            "category": "simulation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A lightweight, fast physics engine for articulated bodies."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Efficient rigid body dynamics with analytical derivatives."
                ),
                ExpertiseLevel.ADVANCED: (
                    "O(n) spatial algebra. CasADi/Ceres for automatic differentiation."
                ),
            },
            "see_also": ["https://github.com/stack-of-tasks/pinocchio"],
            "related_terms": ["physics_engine", "mujoco", "drake"],
        },
        "simulation": {
            "term": "Simulation",
            "category": "simulation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Running a virtual version of a physical system on a computer."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Numerical integration of equations of motion over time. "
                    "Step size affects accuracy and stability."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Forward dynamics integration. Explicit vs implicit methods. "
                    "Constraint stabilization for DAE systems."
                ),
            },
            "related_terms": ["forward_dynamics", "time_step", "integration"],
        },
        "time_step": {
            "term": "Time Step",
            "category": "simulation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How much time passes between each calculation in a simulation."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Δt in numerical integration. Smaller = more accurate but slower. "
                    "Typically 0.1-5 ms for biomechanics."
                ),
                ExpertiseLevel.ADVANCED: (
                    "CFL condition limits explicit methods. Stiff systems need implicit integration. "
                    "Adaptive stepping for efficiency."
                ),
            },
            "units": "seconds",
            "related_terms": ["simulation", "integration", "stability"],
        },
        "integration": {
            "term": "Numerical Integration",
            "category": "simulation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Computing positions and velocities step by step from accelerations."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Methods: Euler, Runge-Kutta, Verlet. Trade-off between accuracy, "
                    "stability, and computational cost."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Symplectic integrators preserve energy. Semi-implicit Euler: "
                    "v_{n+1} = v_n + hf(q_n), q_{n+1} = q_n + hv_{n+1}."
                ),
            },
            "related_terms": ["simulation", "time_step", "euler_method"],
        },
        "stability": {
            "term": "Numerical Stability",
            "category": "simulation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Whether a simulation stays well-behaved or explodes."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Bounded error growth over time. Unstable: small errors grow exponentially. "
                    "Affected by time step and stiffness."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Lyapunov stability for numerical schemes. A-stability for stiff ODEs. "
                    "Constraint drift in DAE systems."
                ),
            },
            "related_terms": ["time_step", "integration", "stiffness"],
        },
        "urdf": {
            "term": "URDF",
            "category": "simulation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A file format describing robot structure - joints and links."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Unified Robot Description Format. XML-based. Defines kinematic tree, "
                    "inertias, visuals, and collisions."
                ),
                ExpertiseLevel.ADVANCED: (
                    "ROS standard. Limitations: no closed chains, limited joint types. "
                    "MJCF and SDF offer more features."
                ),
            },
            "related_terms": ["mjcf", "sdf", "kinematic_tree"],
        },
        "mjcf": {
            "term": "MJCF",
            "category": "simulation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "MuJoCo's file format for describing physical models."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "MuJoCo Model Format. XML-based. Supports tendons, actuators, sensors, "
                    "and advanced contact models."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Compiler transforms to runtime model. Equality constraints for closed chains. "
                    "Muscle and tendon modeling support."
                ),
            },
            "related_terms": ["mujoco", "urdf", "sdf"],
        },
    }


def get_data_terms() -> GlossaryData:
    """Get data and measurement terms."""
    return {
        "c3d_file": {
            "term": "C3D File",
            "category": "data",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A file format for storing motion capture data."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Standard binary format for 3D biomechanics. Stores markers, "
                    "analog data, and metadata."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Header (512 bytes), parameter section, data section. "
                    "Up to 65535 frames, multiple point/analog channels."
                ),
            },
            "related_terms": ["motion_capture", "marker", "force_plate"],
        },
        "motion_capture": {
            "term": "Motion Capture",
            "category": "data",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Recording body movement using cameras and markers."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "System for tracking 3D marker positions. Types: optical (passive/active), "
                    "inertial (IMU), markerless."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Sub-millimeter accuracy with calibrated systems. Typical rates: 100-500 Hz. "
                    "Noise increases with distance and occlusion."
                ),
            },
            "related_terms": ["marker", "c3d_file", "imu"],
        },
        "marker": {
            "term": "Marker",
            "category": "data",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Reflective balls placed on the body for motion capture."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Retro-reflective spheres (10-25mm) at anatomical landmarks. "
                    "Tracked by infrared cameras."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Placement affects soft tissue artifact. Anatomical vs technical markers. "
                    "Cluster methods reduce artifact."
                ),
            },
            "related_terms": ["motion_capture", "anatomical_landmark", "soft_tissue_artifact"],
        },
        "force_plate": {
            "term": "Force Plate",
            "category": "data",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A platform that measures the forces your feet apply to the ground."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Instrument measuring 3D forces and moments. Typical output: Fx, Fy, Fz, "
                    "Mx, My, Mz, plus center of pressure."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Piezoelectric or strain gauge. Sample rates 1000+ Hz. "
                    "COP accuracy ±1mm. Calibration matrix transforms raw data."
                ),
            },
            "related_terms": ["ground_reaction_force", "center_of_pressure", "analog_data"],
        },
        "center_of_pressure": {
            "term": "Center of Pressure (COP)",
            "category": "data",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "The average point where your weight pushes on the ground."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Point of application of ground reaction force. "
                    "COP = (My/Fz, -Mx/Fz) from force plate data."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Instantaneous, not averaged. Undefined when Fz ≈ 0. "
                    "COP excursion indicates stability. Sway analysis for balance."
                ),
            },
            "formula": "COP_x = -My/Fz, COP_y = Mx/Fz",
            "units": "meters",
            "related_terms": ["force_plate", "ground_reaction_force", "balance"],
        },
        "imu": {
            "term": "IMU (Inertial Measurement Unit)",
            "category": "data",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A sensor that measures acceleration and rotation."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Contains accelerometer and gyroscope. 6 DOF (9 with magnetometer). "
                    "Portable alternative to optical mocap."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Sensor fusion via Kalman filter. Drift accumulation limits accuracy. "
                    "ZUPT (zero velocity update) for correction."
                ),
            },
            "related_terms": ["accelerometer", "gyroscope", "sensor_fusion"],
        },
        "emg": {
            "term": "EMG (Electromyography)",
            "category": "data",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Measuring the electrical activity of muscles."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Surface or intramuscular electrodes detect motor unit action potentials. "
                    "Indicates muscle activation timing and relative intensity."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Signal processing: rectification, filtering (20-500 Hz), RMS envelope. "
                    "Amplitude normalization to MVC. Onset detection algorithms."
                ),
            },
            "related_terms": ["muscle_activation", "mvc", "signal_processing"],
        },
        "sampling_rate": {
            "term": "Sampling Rate",
            "category": "data",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How many times per second data is recorded."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Samples per second (Hz). Nyquist: must be >2× highest frequency. "
                    "Motion capture: 100-500 Hz. Force plates: 1000+ Hz."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Anti-aliasing filter at Nyquist frequency. Oversampling for signal quality. "
                    "Resampling requires interpolation."
                ),
            },
            "units": "Hz",
            "related_terms": ["nyquist", "filtering", "interpolation"],
        },
        "filtering": {
            "term": "Filtering",
            "category": "data",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Smoothing data to remove noise and unwanted signals."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Low-pass filter removes high-frequency noise. Butterworth common in biomechanics. "
                    "Cutoff frequency selection is critical."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Zero-lag via forward-backward filtering (filtfilt). "
                    "Residual analysis or cross-validation for cutoff selection. "
                    "Filter order affects roll-off."
                ),
            },
            "related_terms": ["butterworth", "cutoff_frequency", "signal_processing"],
        },
        "differentiation": {
            "term": "Numerical Differentiation",
            "category": "data",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Calculating velocity from position, or acceleration from velocity."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Approximating derivatives from discrete data. Amplifies high-frequency noise. "
                    "Requires filtering before differentiation."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Central difference: (x_{i+1} - x_{i-1})/(2Δt). "
                    "Savitzky-Golay combines smoothing and differentiation. "
                    "Spline fitting alternative."
                ),
            },
            "related_terms": ["filtering", "velocity", "acceleration"],
        },
    }


def get_validation_terms() -> GlossaryData:
    """Get validation and quality terms."""
    return {
        "cross_engine_validation": {
            "term": "Cross-Engine Validation",
            "category": "validation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Checking results are correct by running on multiple physics engines."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Comparing results across MuJoCo, Drake, and Pinocchio. "
                    "Agreement indicates reliability."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Tolerance thresholds: τ ± 2%, KE ± 0.5%, position ± 1e-6. "
                    "Discrepancies often indicate contact model differences."
                ),
            },
            "related_terms": ["physics_engine", "tolerance", "validation"],
        },
        "drift_control_decomposition": {
            "term": "Drift-Control Decomposition",
            "category": "validation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "Understanding which part of motion is passive vs actively controlled."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Separating motion into drift (gravity, inertia) and control (muscles). "
                    "DCR indicates active control requirement."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Affine decomposition: q̈ = drift(q, q̇) + B(q)u. "
                    "DCR = ||control|| / (||drift|| + ||control||)."
                ),
            },
            "formula": "DCR = ||control|| / (||drift|| + ||control||)",
            "related_terms": ["inverse_dynamics", "energy", "muscle_contribution"],
        },
        "tolerance": {
            "term": "Tolerance",
            "category": "validation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How much difference is acceptable between expected and actual results."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Acceptable deviation from reference. Absolute (fixed value) or "
                    "relative (percentage). Context-dependent."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Propagated uncertainty from input errors. Combined via quadrature. "
                    "Sensitivity analysis for critical parameters."
                ),
            },
            "related_terms": ["validation", "uncertainty", "error"],
        },
        "rmse": {
            "term": "RMSE (Root Mean Square Error)",
            "category": "validation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A measure of how different two signals are on average."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "√(Σ(x_pred - x_true)²/n). Units same as measured quantity. "
                    "Common validation metric."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Sensitive to outliers. Normalize by range (NRMSE) or mean for comparison. "
                    "Not dimensionless like correlation."
                ),
            },
            "formula": "RMSE = √(Σ(x_pred - x_true)²/n)",
            "related_terms": ["mae", "correlation", "validation"],
        },
        "correlation": {
            "term": "Correlation",
            "category": "validation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How well two things move together - when one goes up, does the other?"
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Pearson r: linear relationship strength. r = 1 perfect positive, "
                    "-1 perfect negative, 0 no relationship."
                ),
                ExpertiseLevel.ADVANCED: (
                    "r² = coefficient of determination. Spearman for non-linear. "
                    "Cross-correlation for time-shifted relationships."
                ),
            },
            "formula": "r = Σ(x-x̄)(y-ȳ) / √(Σ(x-x̄)²Σ(y-ȳ)²)",
            "related_terms": ["rmse", "validation", "r_squared"],
        },
        "bland_altman": {
            "term": "Bland-Altman Plot",
            "category": "validation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "A graph showing agreement between two measurement methods."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Difference vs mean plot. Shows bias (mean difference) and "
                    "limits of agreement (±1.96 SD)."
                ),
                ExpertiseLevel.ADVANCED: (
                    "Proportional bias: correlation between difference and mean. "
                    "LOA interpretation requires clinical context."
                ),
            },
            "related_terms": ["validation", "agreement", "bias"],
        },
        "uncertainty": {
            "term": "Uncertainty",
            "category": "validation",
            "definitions": {
                ExpertiseLevel.BEGINNER: (
                    "How unsure we are about a measurement or calculation."
                ),
                ExpertiseLevel.INTERMEDIATE: (
                    "Range of possible values. Types: random (Type A) and systematic (Type B). "
                    "Reported as ± value or confidence interval."
                ),
                ExpertiseLevel.ADVANCED: (
                    "GUM framework. Propagation via Taylor series or Monte Carlo. "
                    "Sensitivity analysis identifies dominant sources."
                ),
            },
            "related_terms": ["tolerance", "error", "sensitivity_analysis"],
        },
    }


def get_math_terms() -> GlossaryData:
    """Get mathematical concepts terms."""
    return {
        "matrix": {
            "term": "Matrix",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A rectangular array of numbers arranged in rows and columns.",
                ExpertiseLevel.INTERMEDIATE: "2D array enabling linear transformations. Multiplication order matters.",
                ExpertiseLevel.ADVANCED: "Element of vector space M(m×n). Used for rotations, projections, system solving.",
            },
            "related_terms": ["vector", "linear_algebra", "transformation"],
        },
        "vector": {
            "term": "Vector",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A quantity with both magnitude and direction, like velocity.",
                ExpertiseLevel.INTERMEDIATE: "Ordered list of numbers representing direction and magnitude.",
                ExpertiseLevel.ADVANCED: "Element of vector space. Basis-dependent representation. Covariant vs contravariant.",
            },
            "related_terms": ["matrix", "scalar", "dot_product"],
        },
        "scalar": {
            "term": "Scalar",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A single number, like mass or temperature.",
                ExpertiseLevel.INTERMEDIATE: "Magnitude without direction. Invariant under coordinate transforms.",
                ExpertiseLevel.ADVANCED: "Rank-0 tensor. Field element in vector space definition.",
            },
            "related_terms": ["vector", "tensor", "magnitude"],
        },
        "tensor": {
            "term": "Tensor",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A generalization of vectors and matrices to higher dimensions.",
                ExpertiseLevel.INTERMEDIATE: "Multi-dimensional array with transformation rules. Inertia tensor is rank-2.",
                ExpertiseLevel.ADVANCED: "Multilinear map. Type (r,s) has r contravariant, s covariant indices.",
            },
            "related_terms": ["matrix", "inertia_tensor", "coordinate_transform"],
        },
        "dot_product": {
            "term": "Dot Product",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Multiplying vectors to get a single number. Used for projection.",
                ExpertiseLevel.INTERMEDIATE: "a·b = |a||b|cos(θ). Measures alignment between vectors.",
                ExpertiseLevel.ADVANCED: "Inner product in Euclidean space. Σaᵢbᵢ. Induces norm.",
            },
            "formula": "a·b = Σaᵢbᵢ = |a||b|cos(θ)",
            "related_terms": ["cross_product", "vector", "projection"],
        },
        "cross_product": {
            "term": "Cross Product",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Multiplying vectors to get a perpendicular vector. Used for torques.",
                ExpertiseLevel.INTERMEDIATE: "a×b = |a||b|sin(θ)n̂. Result perpendicular to both inputs.",
                ExpertiseLevel.ADVANCED: "Antisymmetric operation in R³. a×b = [a]×b via skew-symmetric matrix.",
            },
            "formula": "a×b = |a||b|sin(θ)n̂",
            "related_terms": ["dot_product", "torque", "angular_momentum"],
        },
        "derivative": {
            "term": "Derivative",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Rate of change. How fast something is changing.",
                ExpertiseLevel.INTERMEDIATE: "f'(x) = lim(Δx→0) [f(x+Δx)-f(x)]/Δx. Slope of tangent line.",
                ExpertiseLevel.ADVANCED: "Linear approximation. Partial derivatives for multivariate. Chain rule for composition.",
            },
            "formula": "f'(x) = df/dx",
            "related_terms": ["integral", "differentiation", "gradient"],
        },
        "integral": {
            "term": "Integral",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Accumulation or area under a curve.",
                ExpertiseLevel.INTERMEDIATE: "Inverse of derivative. ∫f(x)dx. Definite integral gives area.",
                ExpertiseLevel.ADVANCED: "Riemann or Lebesgue integration. Path integrals for work. Volume integrals for mass.",
            },
            "formula": "∫f(x)dx",
            "related_terms": ["derivative", "area", "work"],
        },
        "gradient": {
            "term": "Gradient",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Direction of steepest increase for a quantity.",
                ExpertiseLevel.INTERMEDIATE: "∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z). Points uphill, magnitude is slope.",
                ExpertiseLevel.ADVANCED: "Covector field. Gradient descent follows -∇f. Used in optimization.",
            },
            "formula": "∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)",
            "related_terms": ["derivative", "optimization", "potential_energy"],
        },
        "eigenvalue": {
            "term": "Eigenvalue",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Special scaling factors for a matrix.",
                ExpertiseLevel.INTERMEDIATE: "Av = λv. Eigenvectors only scale, don't rotate. λ is eigenvalue.",
                ExpertiseLevel.ADVANCED: "Roots of characteristic polynomial det(A-λI)=0. Real for symmetric matrices.",
            },
            "formula": "Av = λv",
            "related_terms": ["eigenvector", "matrix", "principal_axes"],
        },
        "eigenvector": {
            "term": "Eigenvector",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Special directions that only stretch under a transformation.",
                ExpertiseLevel.INTERMEDIATE: "Vector v such that Av = λv. Principal axes of inertia are eigenvectors.",
                ExpertiseLevel.ADVANCED: "Basis for eigenspace. Orthogonal for symmetric matrices. Degenerate when eigenvalues repeated.",
            },
            "related_terms": ["eigenvalue", "matrix", "principal_axes"],
        },
        "linear_algebra": {
            "term": "Linear Algebra",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Mathematics of vectors and matrices.",
                ExpertiseLevel.INTERMEDIATE: "Study of linear transformations and vector spaces. Foundation for dynamics.",
                ExpertiseLevel.ADVANCED: "Abstract: vector spaces, linear maps, dual spaces. Computational: matrix algorithms.",
            },
            "related_terms": ["matrix", "vector", "transformation"],
        },
        "optimization": {
            "term": "Optimization",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Finding the best solution among many possibilities.",
                ExpertiseLevel.INTERMEDIATE: "Minimizing/maximizing objective function subject to constraints.",
                ExpertiseLevel.ADVANCED: "Convex vs non-convex. Gradient descent, Newton methods, SQP for constrained.",
            },
            "related_terms": ["gradient", "objective_function", "constraint"],
        },
        "least_squares": {
            "term": "Least Squares",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Finding the best fit line through data points.",
                ExpertiseLevel.INTERMEDIATE: "Minimize Σ(residual)². Solution: x = (AᵀA)⁻¹Aᵀb.",
                ExpertiseLevel.ADVANCED: "QR factorization for numerical stability. Weighted, regularized variants.",
            },
            "formula": "min ||Ax - b||²",
            "related_terms": ["regression", "fitting", "optimization"],
        },
        "interpolation": {
            "term": "Interpolation",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Estimating values between known data points.",
                ExpertiseLevel.INTERMEDIATE: "Linear, polynomial, spline methods. Cubic spline common in biomechanics.",
                ExpertiseLevel.ADVANCED: "Spline: piecewise polynomial with continuity constraints. C² for cubic splines.",
            },
            "related_terms": ["spline", "extrapolation", "fitting"],
        },
        "spline": {
            "term": "Spline",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A smooth curve passing through data points.",
                ExpertiseLevel.INTERMEDIATE: "Piecewise polynomial with continuity at knots. Cubic spline most common.",
                ExpertiseLevel.ADVANCED: "B-spline basis. Natural, clamped, or periodic boundary conditions.",
            },
            "related_terms": ["interpolation", "smoothing", "curve_fitting"],
        },
        "fourier_transform": {
            "term": "Fourier Transform",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Breaking a signal into its frequency components.",
                ExpertiseLevel.INTERMEDIATE: "Transforms time domain to frequency domain. FFT for efficient computation.",
                ExpertiseLevel.ADVANCED: "F(ω) = ∫f(t)e^(-iωt)dt. DFT for discrete signals. Parseval's theorem relates energy.",
            },
            "formula": "F(ω) = ∫f(t)e^(-iωt)dt",
            "related_terms": ["frequency", "spectral_analysis", "filtering"],
        },
        "ode": {
            "term": "Ordinary Differential Equation (ODE)",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "An equation involving rates of change.",
                ExpertiseLevel.INTERMEDIATE: "Equation with derivatives: dy/dt = f(t,y). Solution is a function.",
                ExpertiseLevel.ADVANCED: "Order = highest derivative. Existence/uniqueness via Lipschitz. Stiff ODEs need implicit methods.",
            },
            "related_terms": ["integration", "simulation", "initial_condition"],
        },
        "dae": {
            "term": "Differential Algebraic Equation (DAE)",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "ODEs with additional constraint equations.",
                ExpertiseLevel.INTERMEDIATE: "F(t, y, ẏ) = 0 with some components purely algebraic.",
                ExpertiseLevel.ADVANCED: "Index = differentiation steps to get ODE. Constrained mechanics is index-3.",
            },
            "related_terms": ["ode", "constraint", "equations_of_motion"],
        },
        "numerical_methods": {
            "term": "Numerical Methods",
            "category": "math",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Using computers to solve math problems approximately.",
                ExpertiseLevel.INTERMEDIATE: "Algorithms for solving equations, integration, optimization numerically.",
                ExpertiseLevel.ADVANCED: "Error analysis, stability, convergence. Trade-offs: accuracy vs speed.",
            },
            "related_terms": ["integration", "optimization", "simulation"],
        },
    }


def get_signal_terms() -> GlossaryData:
    """Get signal processing terms."""
    return {
        "signal_processing": {
            "term": "Signal Processing",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Cleaning up and analyzing recorded data.",
                ExpertiseLevel.INTERMEDIATE: "Filtering, transforming, and extracting features from time-series data.",
                ExpertiseLevel.ADVANCED: "Digital filter design, spectral analysis, wavelet transforms.",
            },
            "related_terms": ["filtering", "fourier_transform", "sampling"],
        },
        "butterworth": {
            "term": "Butterworth Filter",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A common type of smoothing filter.",
                ExpertiseLevel.INTERMEDIATE: "Maximally flat passband. Order determines roll-off steepness.",
                ExpertiseLevel.ADVANCED: "Poles on circle in s-plane. 2nd order: -40 dB/decade. Zero-phase via filtfilt.",
            },
            "related_terms": ["filtering", "cutoff_frequency", "low_pass"],
        },
        "cutoff_frequency": {
            "term": "Cutoff Frequency",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The frequency above which signals are removed.",
                ExpertiseLevel.INTERMEDIATE: "-3 dB point where power is halved. Selection critical for preserving signal.",
                ExpertiseLevel.ADVANCED: "Residual analysis or cross-validation for optimal selection. Task-dependent.",
            },
            "units": "Hz",
            "related_terms": ["butterworth", "filtering", "nyquist"],
        },
        "nyquist": {
            "term": "Nyquist Frequency",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Half the sampling rate - the highest frequency you can measure.",
                ExpertiseLevel.INTERMEDIATE: "fNyquist = fs/2. Aliasing occurs above this. Anti-aliasing filter required.",
                ExpertiseLevel.ADVANCED: "Nyquist-Shannon theorem: fs > 2fmax for perfect reconstruction.",
            },
            "formula": "fNyquist = fs/2",
            "units": "Hz",
            "related_terms": ["sampling_rate", "aliasing", "anti_aliasing"],
        },
        "aliasing": {
            "term": "Aliasing",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "When high frequencies appear as low frequencies due to slow sampling.",
                ExpertiseLevel.INTERMEDIATE: "Frequency folding when sampling below Nyquist. Irreversible distortion.",
                ExpertiseLevel.ADVANCED: "Prevention: analog anti-aliasing filter before ADC. Digital: oversampling + decimation.",
            },
            "related_terms": ["nyquist", "sampling_rate", "anti_aliasing"],
        },
        "low_pass": {
            "term": "Low-Pass Filter",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Removes high-frequency noise, keeping the smooth signal.",
                ExpertiseLevel.INTERMEDIATE: "Passes frequencies below cutoff, attenuates above. Most common in biomechanics.",
                ExpertiseLevel.ADVANCED: "IIR (Butterworth, Chebyshev) or FIR designs. Phase distortion considerations.",
            },
            "related_terms": ["high_pass", "band_pass", "butterworth"],
        },
        "high_pass": {
            "term": "High-Pass Filter",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Removes slow drifts, keeping rapid changes.",
                ExpertiseLevel.INTERMEDIATE: "Passes frequencies above cutoff. Removes baseline wander and DC offset.",
                ExpertiseLevel.ADVANCED: "Used for EMG (20 Hz cutoff) and accelerometer data (removes gravity).",
            },
            "related_terms": ["low_pass", "band_pass", "dc_offset"],
        },
        "band_pass": {
            "term": "Band-Pass Filter",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Keeps only frequencies in a specific range.",
                ExpertiseLevel.INTERMEDIATE: "Combination of low-pass and high-pass. Isolates frequency band of interest.",
                ExpertiseLevel.ADVANCED: "Bandwidth affects filter sharpness. Order determines transition steepness.",
            },
            "related_terms": ["low_pass", "high_pass", "bandwidth"],
        },
        "noise": {
            "term": "Noise",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Unwanted random variations in measurements.",
                ExpertiseLevel.INTERMEDIATE: "Random component obscuring true signal. Sources: electrical, environmental, quantization.",
                ExpertiseLevel.ADVANCED: "White noise: flat spectrum. Pink: 1/f. SNR (signal-to-noise ratio) quantifies quality.",
            },
            "related_terms": ["filtering", "snr", "artifact"],
        },
        "snr": {
            "term": "Signal-to-Noise Ratio (SNR)",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How much stronger the real signal is compared to noise.",
                ExpertiseLevel.INTERMEDIATE: "SNR = Psignal/Pnoise. Often in dB: 10log₁₀(SNR). Higher is better.",
                ExpertiseLevel.ADVANCED: "Peak SNR (PSNR) for images. Effective bits = (SNR_dB - 1.76)/6.02.",
            },
            "formula": "SNR = Psignal/Pnoise",
            "units": "dB",
            "related_terms": ["noise", "filtering", "quality"],
        },
        "artifact": {
            "term": "Artifact",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Errors in data from the measurement process.",
                ExpertiseLevel.INTERMEDIATE: "Non-physiological signals. Motion artifact in EMG, marker dropout in mocap.",
                ExpertiseLevel.ADVANCED: "Detection via threshold, template matching. Removal: interpolation, reconstruction.",
            },
            "related_terms": ["noise", "soft_tissue_artifact", "marker"],
        },
        "soft_tissue_artifact": {
            "term": "Soft Tissue Artifact (STA)",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Marker movement caused by skin and muscle motion.",
                ExpertiseLevel.INTERMEDIATE: "Markers don't perfectly track bone. Can be 1-3 cm at thigh.",
                ExpertiseLevel.ADVANCED: "Frequency content overlaps movement signal. Cluster methods, global optimization to reduce.",
            },
            "related_terms": ["marker", "motion_capture", "artifact"],
        },
        "spectral_analysis": {
            "term": "Spectral Analysis",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Looking at the frequency content of a signal.",
                ExpertiseLevel.INTERMEDIATE: "Power spectral density shows energy at each frequency. FFT-based.",
                ExpertiseLevel.ADVANCED: "Welch method for noise reduction. Window selection affects resolution. Coherence for relationship.",
            },
            "related_terms": ["fourier_transform", "power_spectrum", "frequency"],
        },
        "wavelet": {
            "term": "Wavelet Transform",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Analyzing signals at different scales and times.",
                ExpertiseLevel.INTERMEDIATE: "Time-frequency analysis. Localized in both domains unlike FFT.",
                ExpertiseLevel.ADVANCED: "Mother wavelet selection. CWT for analysis, DWT for compression. Denoising via thresholding.",
            },
            "related_terms": ["fourier_transform", "spectral_analysis", "denoising"],
        },
        "resampling": {
            "term": "Resampling",
            "category": "signal",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Changing the number of samples in a signal.",
                ExpertiseLevel.INTERMEDIATE: "Upsampling (interpolation) or downsampling (decimation). Filter to prevent aliasing.",
                ExpertiseLevel.ADVANCED: "Polyphase implementation efficient. Anti-aliasing filter order affects quality.",
            },
            "related_terms": ["sampling_rate", "interpolation", "aliasing"],
        },
    }


def get_injury_terms() -> GlossaryData:
    """Get injury and safety terms."""
    return {
        "injury": {
            "term": "Injury",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Damage to body tissues from excessive force or motion.",
                ExpertiseLevel.INTERMEDIATE: "Tissue failure from mechanical overload. Acute (sudden) or overuse (cumulative).",
                ExpertiseLevel.ADVANCED: "Load exceeds tissue tolerance. Dose-response relationship. Risk factors: magnitude, rate, repetition.",
            },
            "related_terms": ["strain", "sprain", "overuse"],
        },
        "strain": {
            "term": "Muscle Strain",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A pulled or torn muscle.",
                ExpertiseLevel.INTERMEDIATE: "Muscle or tendon tear from excessive stretch or contraction. Grades I-III.",
                ExpertiseLevel.ADVANCED: "Eccentric loading most common cause. Musculotendinous junction vulnerable. Hamstring, groin common sites.",
            },
            "related_terms": ["sprain", "muscle", "tendon"],
        },
        "sprain": {
            "term": "Ligament Sprain",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A stretched or torn ligament.",
                ExpertiseLevel.INTERMEDIATE: "Ligament injury from excessive joint motion. Grades I (stretch) to III (complete tear).",
                ExpertiseLevel.ADVANCED: "Nonlinear stress-strain means sudden failure. Healing limited by vascularity. ACL, ankle common.",
            },
            "related_terms": ["strain", "ligament", "joint"],
        },
        "overuse": {
            "term": "Overuse Injury",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Injury from too much repetitive activity.",
                ExpertiseLevel.INTERMEDIATE: "Cumulative microtrauma exceeds tissue repair capacity. Tennis elbow, stress fractures.",
                ExpertiseLevel.ADVANCED: "Dose-response: load × repetition × inadequate recovery. Tissue remodeling can't keep pace.",
            },
            "related_terms": ["injury", "fatigue", "stress_fracture"],
        },
        "stress_fracture": {
            "term": "Stress Fracture",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A crack in bone from repetitive loading.",
                ExpertiseLevel.INTERMEDIATE: "Bone failure from accumulated microdamage. Common in runners, dancers.",
                ExpertiseLevel.ADVANCED: "Bone remodeling imbalance. BMU (basic multicellular unit) kinetics. Risk factors: training load, nutrition, biomechanics.",
            },
            "related_terms": ["overuse", "bone", "loading"],
        },
        "fatigue": {
            "term": "Muscle Fatigue",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Muscles getting tired and weaker during activity.",
                ExpertiseLevel.INTERMEDIATE: "Decline in force-generating capacity. Central (neural) and peripheral (muscular) components.",
                ExpertiseLevel.ADVANCED: "Metabolic: PCr depletion, H+ accumulation. Neuromuscular: reduced Ca²⁺ release, cross-bridge cycling.",
            },
            "related_terms": ["muscle", "endurance", "recovery"],
        },
        "low_back_pain": {
            "term": "Low Back Pain",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Pain in the lower back, common in golfers.",
                ExpertiseLevel.INTERMEDIATE: "Multifactorial: disc, facet, muscle. Golf swing creates high lumbar loads.",
                ExpertiseLevel.ADVANCED: "Peak L4-L5 compression ~8× body weight. Lateral shear significant. Reverse spine angle risky.",
            },
            "related_terms": ["lumbar_spine", "intervertebral_disc", "golf"],
        },
        "golfers_elbow": {
            "term": "Golfer's Elbow",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Pain on the inside of the elbow from golf.",
                ExpertiseLevel.INTERMEDIATE: "Medial epicondylitis. Overuse of wrist flexors and pronators.",
                ExpertiseLevel.ADVANCED: "Tendinopathy of flexor-pronator mass origin. Actually degenerative, not inflammatory. Eccentric loading for rehab.",
            },
            "related_terms": ["elbow_joint", "overuse", "tendon"],
        },
        "rotator_cuff_injury": {
            "term": "Rotator Cuff Injury",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Damage to the muscles that stabilize your shoulder.",
                ExpertiseLevel.INTERMEDIATE: "Strain or tear of supraspinatus, infraspinatus, teres minor, or subscapularis.",
                ExpertiseLevel.ADVANCED: "Supraspinatus most vulnerable (watershed zone). Impingement vs tensile failure. Age-related degeneration common.",
            },
            "related_terms": ["shoulder_joint", "strain", "impingement"],
        },
        "risk_factor": {
            "term": "Injury Risk Factor",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Something that increases your chance of getting hurt.",
                ExpertiseLevel.INTERMEDIATE: "Modifiable (training, technique) or non-modifiable (age, anatomy). Interact multiplicatively.",
                ExpertiseLevel.ADVANCED: "Relative risk, odds ratio. Prospective studies establish causation. Screening for prevention.",
            },
            "related_terms": ["injury", "prevention", "biomechanics"],
        },
        "tissue_tolerance": {
            "term": "Tissue Tolerance",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How much load a body part can handle before damage.",
                ExpertiseLevel.INTERMEDIATE: "Maximum load before failure. Varies with tissue type, health, adaptation.",
                ExpertiseLevel.ADVANCED: "Probabilistic: load-tolerance overlap. Rate-dependent for viscoelastic tissues. Fatigue reduces tolerance.",
            },
            "related_terms": ["injury", "loading", "adaptation"],
        },
        "loading": {
            "term": "Mechanical Loading",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Forces applied to body tissues.",
                ExpertiseLevel.INTERMEDIATE: "Types: compression, tension, shear, bending, torsion. Tissues adapt to loading.",
                ExpertiseLevel.ADVANCED: "Wolff's law for bone, Davis's law for soft tissue. Load-adaptation curve. Optimal loading zone.",
            },
            "related_terms": ["stress", "strain_mechanics", "adaptation"],
        },
        "stress_mechanics": {
            "term": "Mechanical Stress",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Force per unit area inside a material.",
                ExpertiseLevel.INTERMEDIATE: "σ = F/A. Normal (perpendicular) and shear (parallel) components.",
                ExpertiseLevel.ADVANCED: "Stress tensor: 9 components (6 independent). Principal stresses eliminate shear. von Mises for yield.",
            },
            "formula": "σ = F/A",
            "units": "Pa (N/m²)",
            "related_terms": ["strain_mechanics", "loading", "failure"],
        },
        "strain_mechanics": {
            "term": "Mechanical Strain",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How much something stretches or deforms.",
                ExpertiseLevel.INTERMEDIATE: "ε = ΔL/L. Ratio of deformation to original length. Dimensionless.",
                ExpertiseLevel.ADVANCED: "Engineering vs true strain. Strain tensor for 3D. Rate of strain affects tissue response.",
            },
            "formula": "ε = ΔL/L",
            "related_terms": ["stress_mechanics", "loading", "viscoelastic"],
        },
        "viscoelastic": {
            "term": "Viscoelasticity",
            "category": "injury",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Materials that behave like both solids and liquids.",
                ExpertiseLevel.INTERMEDIATE: "Time and rate-dependent response. Creep, stress relaxation, hysteresis.",
                ExpertiseLevel.ADVANCED: "Maxwell, Kelvin-Voigt, standard linear models. Tendons, ligaments are viscoelastic.",
            },
            "related_terms": ["tendon", "ligament", "creep"],
        },
    }


def get_muscle_terms() -> GlossaryData:
    """Get muscle physiology terms."""
    return {
        "muscle_force": {
            "term": "Muscle Force",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The pulling force a muscle generates.",
                ExpertiseLevel.INTERMEDIATE: "Net force from activated muscle fibers. Function of length, velocity, activation.",
                ExpertiseLevel.ADVANCED: "F = f(l)·f(v)·a·Fmax. Pennation angle affects effective force. EMG-force relationship.",
            },
            "units": "N",
            "related_terms": ["muscle", "activation", "hill_model"],
        },
        "muscle_activation": {
            "term": "Muscle Activation",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How 'on' a muscle is - from relaxed to fully contracted.",
                ExpertiseLevel.INTERMEDIATE: "Neural drive to muscle, 0-1 scale. First-order dynamics from excitation to activation.",
                ExpertiseLevel.ADVANCED: "a(t) = integral of (u-a)/τ. Time constants ~10-40 ms. Calcium transients underlying.",
            },
            "related_terms": ["emg", "muscle_force", "neural_control"],
        },
        "hill_model": {
            "term": "Hill Muscle Model",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A mathematical model of how muscles work.",
                ExpertiseLevel.INTERMEDIATE: "Three-element model: contractile, series elastic, parallel elastic. Force-length-velocity relations.",
                ExpertiseLevel.ADVANCED: "CE: F(l,v,a). SEE: tendon nonlinear elasticity. PEE: passive muscle stiffness. Activation dynamics added.",
            },
            "related_terms": ["muscle_force", "force_length", "force_velocity"],
        },
        "force_length": {
            "term": "Force-Length Relationship",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How muscle force changes with muscle length.",
                ExpertiseLevel.INTERMEDIATE: "Peak force at optimal length. Drops at shorter/longer lengths. Due to actin-myosin overlap.",
                ExpertiseLevel.ADVANCED: "Active F-L curve: gaussian-like. Passive F-L: exponential above slack length. Total = active + passive.",
            },
            "related_terms": ["muscle_force", "sarcomere", "optimal_length"],
        },
        "force_velocity": {
            "term": "Force-Velocity Relationship",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Muscles are weaker when moving fast.",
                ExpertiseLevel.INTERMEDIATE: "Concentric: force decreases with velocity. Eccentric: force increases slightly then plateaus.",
                ExpertiseLevel.ADVANCED: "Hill equation: (F+a)(v+b)=b(F₀+a). Eccentric ~1.5× isometric max. Cross-bridge cycling kinetics.",
            },
            "formula": "(F+a)(v+b) = b(F₀+a)",
            "related_terms": ["muscle_force", "concentric", "eccentric"],
        },
        "concentric": {
            "term": "Concentric Contraction",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Muscle shortening while generating force.",
                ExpertiseLevel.INTERMEDIATE: "Muscle length decreases under tension. Produces positive work. Biceps during curl up.",
                ExpertiseLevel.ADVANCED: "Power = F·v. Limited by force-velocity: max power ~1/3 max velocity. Metabolic cost high.",
            },
            "related_terms": ["eccentric", "isometric", "muscle_force"],
        },
        "eccentric": {
            "term": "Eccentric Contraction",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Muscle lengthening while generating force.",
                ExpertiseLevel.INTERMEDIATE: "Muscle resists external force while lengthening. Absorbs energy. Biceps lowering weight.",
                ExpertiseLevel.ADVANCED: "Force > isometric max. Lower metabolic cost per force. Higher injury risk. DOMS mechanism.",
            },
            "related_terms": ["concentric", "isometric", "muscle_force"],
        },
        "isometric": {
            "term": "Isometric Contraction",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Muscle generating force without changing length.",
                ExpertiseLevel.INTERMEDIATE: "Static: no external movement. Wall push. Force = muscle tension.",
                ExpertiseLevel.ADVANCED: "Zero work (F·d=0). Series elastic element stretches. MVC testing. Tetanic force reference.",
            },
            "related_terms": ["concentric", "eccentric", "mvc"],
        },
        "mvc": {
            "term": "Maximum Voluntary Contraction (MVC)",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The strongest contraction you can produce.",
                ExpertiseLevel.INTERMEDIATE: "Peak voluntary isometric force. Used to normalize EMG. Test: maximal effort against resistance.",
                ExpertiseLevel.ADVANCED: "Joint angle specific. Superimposed twitch for voluntary activation. Central vs peripheral limits.",
            },
            "related_terms": ["isometric", "emg", "muscle_force"],
        },
        "motor_unit": {
            "term": "Motor Unit",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A nerve cell plus all the muscle fibers it controls.",
                ExpertiseLevel.INTERMEDIATE: "α-motor neuron + muscle fibers. All-or-none: entire unit activates together.",
                ExpertiseLevel.ADVANCED: "Size principle: small units recruited first. Rate coding modulates force. Fiber type determines properties.",
            },
            "related_terms": ["muscle", "neural_control", "recruitment"],
        },
        "recruitment": {
            "term": "Motor Unit Recruitment",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Activating more motor units to increase force.",
                ExpertiseLevel.INTERMEDIATE: "Size principle: small (slow) units first, large (fast) last. Orderly recruitment.",
                ExpertiseLevel.ADVANCED: "Henneman's size principle. Exceptions: rapid movements, training. Surface EMG reflects recruitment.",
            },
            "related_terms": ["motor_unit", "muscle_force", "rate_coding"],
        },
        "rate_coding": {
            "term": "Rate Coding",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Changing force by changing how fast nerves fire.",
                ExpertiseLevel.INTERMEDIATE: "Higher firing rate = more force. Range: ~8-50 Hz. Complements recruitment.",
                ExpertiseLevel.ADVANCED: "Fusion frequency: tetanic plateau. Doublet discharges for rapid force. Fatigue decreases rate.",
            },
            "related_terms": ["motor_unit", "recruitment", "muscle_force"],
        },
        "fiber_type": {
            "term": "Muscle Fiber Type",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Different types of muscle cells with different properties.",
                ExpertiseLevel.INTERMEDIATE: "Type I (slow, oxidative) vs Type II (fast, glycolytic). IIa intermediate.",
                ExpertiseLevel.ADVANCED: "Myosin heavy chain isoforms determine. Fiber type composition varies by muscle and individual. Trainable to degree.",
            },
            "related_terms": ["motor_unit", "fatigue", "power"],
        },
        "pennation_angle": {
            "term": "Pennation Angle",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The angle at which muscle fibers attach to the tendon.",
                ExpertiseLevel.INTERMEDIATE: "Pennate muscles have fibers at angle. More fibers fit, but force reduced by cos(θ).",
                ExpertiseLevel.ADVANCED: "PCSA = V/(L·cos(θ)). Dynamic pennation: angle increases with contraction. Gear ratio effect.",
            },
            "related_terms": ["muscle", "pcsa", "muscle_architecture"],
        },
        "pcsa": {
            "term": "Physiological Cross-Sectional Area (PCSA)",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A measure of muscle size related to strength.",
                ExpertiseLevel.INTERMEDIATE: "Total area of all fibers perpendicular to their direction. PCSA = V/(Lf·cos(θ)).",
                ExpertiseLevel.ADVANCED: "Max isometric force = PCSA × specific tension (~25-40 N/cm²). Accounts for pennation.",
            },
            "formula": "PCSA = muscle_volume / (fiber_length × cos(pennation))",
            "units": "cm²",
            "related_terms": ["pennation_angle", "muscle_force", "specific_tension"],
        },
        "co_contraction": {
            "term": "Co-Contraction",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "When opposing muscles activate at the same time.",
                ExpertiseLevel.INTERMEDIATE: "Agonist and antagonist simultaneous activation. Increases joint stiffness and stability.",
                ExpertiseLevel.ADVANCED: "Trade-off: stability vs metabolic cost. Higher in novices and under uncertainty. Index = antagonist/agonist.",
            },
            "related_terms": ["agonist", "antagonist", "joint_stiffness"],
        },
        "agonist": {
            "term": "Agonist Muscle",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The muscle that causes a movement.",
                ExpertiseLevel.INTERMEDIATE: "Primary mover for a joint action. Biceps for elbow flexion.",
                ExpertiseLevel.ADVANCED: "Context-dependent: changes with movement direction. Synergists assist agonist.",
            },
            "related_terms": ["antagonist", "synergist", "muscle"],
        },
        "antagonist": {
            "term": "Antagonist Muscle",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The muscle that opposes a movement.",
                ExpertiseLevel.INTERMEDIATE: "Opposes agonist action. Triceps for elbow flexion. Controls deceleration.",
                ExpertiseLevel.ADVANCED: "Reciprocal inhibition reduces antagonist activity. Except during co-contraction.",
            },
            "related_terms": ["agonist", "synergist", "co_contraction"],
        },
        "synergist": {
            "term": "Synergist Muscle",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Muscles that help the main muscle do its job.",
                ExpertiseLevel.INTERMEDIATE: "Assists agonist or stabilizes during movement. Brachialis assists biceps.",
                ExpertiseLevel.ADVANCED: "Motor redundancy: infinite muscle force combinations for net torque. Synergy analysis.",
            },
            "related_terms": ["agonist", "antagonist", "redundancy"],
        },
        "muscle_architecture": {
            "term": "Muscle Architecture",
            "category": "muscle",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How muscle fibers are arranged within a muscle.",
                ExpertiseLevel.INTERMEDIATE: "Fiber length, pennation, PCSA. Determines force and excursion capacity.",
                ExpertiseLevel.ADVANCED: "Parallel: long excursion (sartorius). Pennate: high force (gastrocnemius). Fiber type distribution.",
            },
            "related_terms": ["pennation_angle", "pcsa", "fiber_length"],
        },
    }


def get_equipment_terms() -> GlossaryData:
    """Get golf equipment terms."""
    return {
        "club": {
            "term": "Golf Club",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The tool used to hit the golf ball.",
                ExpertiseLevel.INTERMEDIATE: "Components: grip, shaft, head. Different clubs for different shots.",
                ExpertiseLevel.ADVANCED: "MOI, CG location, face properties affect performance. 14 club limit per bag.",
            },
            "related_terms": ["shaft", "club_head", "driver"],
        },
        "driver": {
            "term": "Driver",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The longest club, used to hit the ball the farthest.",
                ExpertiseLevel.INTERMEDIATE: "1-wood. Largest head, lowest loft (8-12°). Tee shots for distance.",
                ExpertiseLevel.ADVANCED: "460cc max volume (USGA). Hot spot for COR. Adjustable loft/face angle common.",
            },
            "related_terms": ["club", "loft", "club_head_speed"],
        },
        "iron": {
            "term": "Iron",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Clubs with flat, angled faces for accuracy.",
                ExpertiseLevel.INTERMEDIATE: "Numbered 3-9 plus wedges. Higher number = more loft = shorter distance.",
                ExpertiseLevel.ADVANCED: "Blade vs cavity back. MOI affects forgiveness. Progressive offset and CG.",
            },
            "related_terms": ["club", "loft", "wedge"],
        },
        "wedge": {
            "term": "Wedge",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "High-lofted clubs for short shots and bunkers.",
                ExpertiseLevel.INTERMEDIATE: "PW (~46°), GW (~52°), SW (~56°), LW (~60°). Bounce angle for turf interaction.",
                ExpertiseLevel.ADVANCED: "Grind options for versatility. Groove regulations affect spin. CG affects trajectory.",
            },
            "related_terms": ["iron", "loft", "bounce"],
        },
        "putter": {
            "term": "Putter",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Club used for rolling the ball on the green.",
                ExpertiseLevel.INTERMEDIATE: "Blade vs mallet. Face insert affects feel. Balance and alignment aids.",
                ExpertiseLevel.ADVANCED: "Face loft 3-4° typically. MOI for stability. CG height for roll.",
            },
            "related_terms": ["club", "putting", "green"],
        },
        "shaft": {
            "term": "Golf Shaft",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The long part of the club you hold.",
                ExpertiseLevel.INTERMEDIATE: "Steel or graphite. Flex (L,A,R,S,X) affects timing and feel.",
                ExpertiseLevel.ADVANCED: "EI profile (stiffness distribution). Kick point affects launch. Torque for stability.",
            },
            "related_terms": ["club", "shaft_flex", "kick_point"],
        },
        "shaft_flex": {
            "term": "Shaft Flex",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How bendy the shaft is.",
                ExpertiseLevel.INTERMEDIATE: "L(adies), A(senior), R(egular), S(tiff), X(tra stiff). Match to swing speed.",
                ExpertiseLevel.ADVANCED: "CPM (cycles per minute) measurement. Butt, mid, tip stiffness vary. Dynamic loading in swing.",
            },
            "related_terms": ["shaft", "club_head_speed", "timing"],
        },
        "loft": {
            "term": "Club Loft",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The angle of the club face - higher loft hits higher shots.",
                ExpertiseLevel.INTERMEDIATE: "Angle between face and vertical. Driver 8-12°, wedges 46-60°.",
                ExpertiseLevel.ADVANCED: "Static vs dynamic loft. Spin loft = dynamic loft - attack angle. Affects launch and spin.",
            },
            "units": "degrees",
            "related_terms": ["launch_angle", "spin_rate", "club"],
        },
        "lie_angle": {
            "term": "Lie Angle",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The angle between the shaft and ground.",
                ExpertiseLevel.INTERMEDIATE: "Affects where club face points at impact. Fitting based on height and posture.",
                ExpertiseLevel.ADVANCED: "1° upright opens face ~2.5°. Dynamic lie differs from static. Affects shot direction.",
            },
            "units": "degrees",
            "related_terms": ["club", "fitting", "face_angle"],
        },
        "club_head": {
            "term": "Club Head",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The part of the club that hits the ball.",
                ExpertiseLevel.INTERMEDIATE: "Materials: steel, titanium, composite. Design affects MOI and COR.",
                ExpertiseLevel.ADVANCED: "Mass distribution: perimeter weighting for forgiveness. CT limits COR. 460cc driver limit.",
            },
            "related_terms": ["club", "moi", "cor"],
        },
        "moi": {
            "term": "Moment of Inertia (Club)",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How resistant the club is to twisting on mishits.",
                ExpertiseLevel.INTERMEDIATE: "Higher MOI = more forgiving. Perimeter weighting increases MOI.",
                ExpertiseLevel.ADVANCED: "USGA limits: 5900 g·cm² max. Affects both horizontal and vertical MOI. Trade-off with workability.",
            },
            "units": "g·cm²",
            "related_terms": ["club_head", "forgiveness", "mishit"],
        },
        "cor": {
            "term": "Coefficient of Restitution",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How bouncy the club face is - affects ball speed.",
                ExpertiseLevel.INTERMEDIATE: "Ratio of separation to approach speed. COR = 0.83 max (USGA).",
                ExpertiseLevel.ADVANCED: "CT (characteristic time) test used by USGA. Depends on impact location. Trampoline effect in thin faces.",
            },
            "formula": "COR = v_after / v_before",
            "related_terms": ["smash_factor", "ball_speed", "club_head"],
        },
        "grip": {
            "term": "Golf Grip",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The rubber part you hold, or how you hold the club.",
                ExpertiseLevel.INTERMEDIATE: "Size affects control. Material: rubber, cord. Hand position: overlap, interlock, ten-finger.",
                ExpertiseLevel.ADVANCED: "Grip pressure affects muscle tension and wrist release. Neutral vs strong vs weak.",
            },
            "related_terms": ["club", "hand", "grip_pressure"],
        },
        "golf_ball": {
            "term": "Golf Ball",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The ball you hit in golf.",
                ExpertiseLevel.INTERMEDIATE: "2-piece (distance), 3-5 piece (spin). Cover: urethane (premium), Surlyn (durability).",
                ExpertiseLevel.ADVANCED: "Dimple pattern affects lift and drag. Compression rating affects feel and performance. Initial velocity 1.526× CHS max (USGA).",
            },
            "related_terms": ["ball_speed", "spin_rate", "dimples"],
        },
        "dimples": {
            "term": "Ball Dimples",
            "category": "equipment",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The little dents on a golf ball that help it fly.",
                ExpertiseLevel.INTERMEDIATE: "Create turbulent boundary layer reducing drag. Typically 300-500 dimples.",
                ExpertiseLevel.ADVANCED: "Dimple depth, edge angle, pattern symmetry affect aerodynamics. Reduces drag ~50% vs smooth.",
            },
            "related_terms": ["golf_ball", "aerodynamics", "drag"],
        },
    }


def get_control_terms() -> GlossaryData:
    """Get motor control terms."""
    return {
        "motor_control": {
            "term": "Motor Control",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How your brain controls your movements.",
                ExpertiseLevel.INTERMEDIATE: "Neural processes organizing muscle activation for skilled movement.",
                ExpertiseLevel.ADVANCED: "Hierarchical: cortex → brainstem → spinal cord. Feedforward + feedback. Internal models.",
            },
            "related_terms": ["neural_control", "coordination", "skill"],
        },
        "coordination": {
            "term": "Coordination",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Making body parts work together smoothly.",
                ExpertiseLevel.INTERMEDIATE: "Organizing multiple DOF for skilled action. Interlimb and intralimb.",
                ExpertiseLevel.ADVANCED: "Degrees of freedom problem (Bernstein). Synergies reduce dimensionality. Task-specific solutions.",
            },
            "related_terms": ["motor_control", "synergy", "skill"],
        },
        "feedforward": {
            "term": "Feedforward Control",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Planning movements in advance without relying on feedback.",
                ExpertiseLevel.INTERMEDIATE: "Open-loop: commands sent before movement. Based on internal model predictions.",
                ExpertiseLevel.ADVANCED: "Anticipatory postural adjustments. Ballistic phase of fast movements. Updated via motor learning.",
            },
            "related_terms": ["feedback", "internal_model", "motor_control"],
        },
        "feedback": {
            "term": "Feedback Control",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Adjusting movements based on what you feel.",
                ExpertiseLevel.INTERMEDIATE: "Closed-loop: sensory signals modify ongoing movement. Slower than feedforward.",
                ExpertiseLevel.ADVANCED: "Delays: visual ~200ms, proprioceptive ~80ms. Optimal feedback control theory. State estimation.",
            },
            "related_terms": ["feedforward", "proprioception", "motor_control"],
        },
        "internal_model": {
            "term": "Internal Model",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Your brain's simulation of how your body and world work.",
                ExpertiseLevel.INTERMEDIATE: "Forward model predicts outcomes. Inverse model computes required commands.",
                ExpertiseLevel.ADVANCED: "Cerebellum key for forward models. Parietal cortex for body schema. Updated via prediction errors.",
            },
            "related_terms": ["feedforward", "motor_learning", "prediction"],
        },
        "motor_learning": {
            "term": "Motor Learning",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Getting better at movements through practice.",
                ExpertiseLevel.INTERMEDIATE: "Relatively permanent change in motor behavior. Stages: cognitive, associative, autonomous.",
                ExpertiseLevel.ADVANCED: "Explicit vs implicit learning. Consolidation, interference. Error-based vs reinforcement learning.",
            },
            "related_terms": ["skill", "practice", "motor_control"],
        },
        "skill": {
            "term": "Motor Skill",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Ability to do a movement well consistently.",
                ExpertiseLevel.INTERMEDIATE: "Learned movement pattern. Acquired through practice. Retained over time.",
                ExpertiseLevel.ADVANCED: "Automaticity: reduced attention demands. Transfer across contexts. Expert-novice differences.",
            },
            "related_terms": ["motor_learning", "practice", "expertise"],
        },
        "variability": {
            "term": "Movement Variability",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How much movements differ from attempt to attempt.",
                ExpertiseLevel.INTERMEDIATE: "Natural variation in motor output. Not always bad - can be functional.",
                ExpertiseLevel.ADVANCED: "Good variability: task-irrelevant. Bad variability: affects outcome. Uncontrolled manifold analysis.",
            },
            "related_terms": ["coordination", "noise", "redundancy"],
        },
        "redundancy": {
            "term": "Motor Redundancy",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Having more ways to do something than needed.",
                ExpertiseLevel.INTERMEDIATE: "More DOF than task requires. Many solutions for same outcome.",
                ExpertiseLevel.ADVANCED: "Null space of Jacobian. Exploited for secondary objectives. Synergies organize redundancy.",
            },
            "related_terms": ["degrees_of_freedom", "synergy", "coordination"],
        },
        "synergy": {
            "term": "Motor Synergy",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Muscles or joints that work together as a unit.",
                ExpertiseLevel.INTERMEDIATE: "Coupling between elements that simplifies control. Reduces effective DOF.",
                ExpertiseLevel.ADVANCED: "Muscle synergies: low-dimensional control signals. Task-specific vs fundamental. PCA, NMF analysis.",
            },
            "related_terms": ["coordination", "redundancy", "motor_control"],
        },
        "proprioception": {
            "term": "Proprioception",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Sensing where your body parts are without looking.",
                ExpertiseLevel.INTERMEDIATE: "Body position and movement sense. Muscle spindles, Golgi tendon organs, joint receptors.",
                ExpertiseLevel.ADVANCED: "Ia afferents: velocity. II afferents: position. GTOs: force. Integrated with vestibular, visual.",
            },
            "related_terms": ["feedback", "sensory", "balance"],
        },
        "balance": {
            "term": "Balance",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Keeping yourself upright without falling.",
                ExpertiseLevel.INTERMEDIATE: "Maintaining CoM over base of support. Vestibular, visual, proprioceptive integration.",
                ExpertiseLevel.ADVANCED: "Inverted pendulum model. Ankle vs hip strategy. Anticipatory vs reactive control.",
            },
            "related_terms": ["center_of_mass", "proprioception", "postural_control"],
        },
        "reaction_time": {
            "term": "Reaction Time",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How quickly you respond to something.",
                ExpertiseLevel.INTERMEDIATE: "Time from stimulus to movement initiation. Simple RT ~200ms, choice RT longer.",
                ExpertiseLevel.ADVANCED: "Hick's law: RT = a + b·log₂(n). Preparation reduces RT. Speed-accuracy tradeoff.",
            },
            "units": "ms",
            "related_terms": ["motor_control", "anticipation", "decision"],
        },
        "timing": {
            "term": "Movement Timing",
            "category": "control",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Doing movements at the right moment.",
                ExpertiseLevel.INTERMEDIATE: "Temporal accuracy in movement. Absolute timing, relative timing, rhythm.",
                ExpertiseLevel.ADVANCED: "Emergent vs timekeeper models. Weber fraction for timing variability. Cerebellum role.",
            },
            "related_terms": ["coordination", "rhythm", "kinetic_chain"],
        },
    }


def get_statistics_terms() -> GlossaryData:
    """Get statistical analysis terms."""
    return {
        "mean": {
            "term": "Mean",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The average value.",
                ExpertiseLevel.INTERMEDIATE: "Sum of values divided by count. μ = Σxᵢ/n. Sensitive to outliers.",
                ExpertiseLevel.ADVANCED: "Sample mean x̄ estimates population μ. Standard error = σ/√n.",
            },
            "formula": "μ = Σxᵢ/n",
            "related_terms": ["median", "standard_deviation", "average"],
        },
        "median": {
            "term": "Median",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The middle value when sorted.",
                ExpertiseLevel.INTERMEDIATE: "50th percentile. Robust to outliers. Use for skewed data.",
                ExpertiseLevel.ADVANCED: "Nonparametric. Hodges-Lehmann estimator for confidence interval.",
            },
            "related_terms": ["mean", "percentile", "robust"],
        },
        "standard_deviation": {
            "term": "Standard Deviation",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How spread out the data is.",
                ExpertiseLevel.INTERMEDIATE: "√(Σ(x-μ)²/n). 68% within ±1 SD for normal data.",
                ExpertiseLevel.ADVANCED: "Population σ vs sample s (n-1 divisor). Coefficient of variation = σ/μ for relative spread.",
            },
            "formula": "σ = √(Σ(x-μ)²/n)",
            "related_terms": ["variance", "mean", "normal_distribution"],
        },
        "variance": {
            "term": "Variance",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Another measure of spread (squared).",
                ExpertiseLevel.INTERMEDIATE: "σ² = Σ(x-μ)²/n. Additive for independent variables.",
                ExpertiseLevel.ADVANCED: "Var(aX+b) = a²Var(X). Unbiased estimator uses n-1. Decomposable: total = within + between.",
            },
            "formula": "σ² = Σ(x-μ)²/n",
            "related_terms": ["standard_deviation", "covariance", "anova"],
        },
        "normal_distribution": {
            "term": "Normal Distribution",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "The bell curve - most common distribution.",
                ExpertiseLevel.INTERMEDIATE: "Symmetric, mean = median. Defined by μ and σ. Many natural phenomena.",
                ExpertiseLevel.ADVANCED: "CLT: sample means → normal. Z-scores: (x-μ)/σ. Shapiro-Wilk test for normality.",
            },
            "related_terms": ["mean", "standard_deviation", "z_score"],
        },
        "confidence_interval": {
            "term": "Confidence Interval",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Range where true value likely falls.",
                ExpertiseLevel.INTERMEDIATE: "95% CI: 95% of such intervals contain true parameter. Not probability statement about parameter.",
                ExpertiseLevel.ADVANCED: "For mean: x̄ ± t·SE. Bootstrap for non-normal. Bayesian credible intervals differ conceptually.",
            },
            "related_terms": ["standard_error", "significance", "estimation"],
        },
        "p_value": {
            "term": "P-Value",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Probability of seeing data this extreme if null hypothesis true.",
                ExpertiseLevel.INTERMEDIATE: "p < 0.05 traditionally 'significant'. Not probability null is true!",
                ExpertiseLevel.ADVANCED: "Misinterpretation common. Effect size and CI more informative. Multiple testing correction needed.",
            },
            "related_terms": ["significance", "hypothesis_test", "effect_size"],
        },
        "effect_size": {
            "term": "Effect Size",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How big the difference or relationship is.",
                ExpertiseLevel.INTERMEDIATE: "Cohen's d for means: small (0.2), medium (0.5), large (0.8). Independent of sample size.",
                ExpertiseLevel.ADVANCED: "d = (μ₁-μ₂)/σ. Hedge's g for small samples. Eta-squared for ANOVA. Interpret contextually.",
            },
            "related_terms": ["p_value", "significance", "power"],
        },
        "anova": {
            "term": "ANOVA",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Comparing averages of more than two groups.",
                ExpertiseLevel.INTERMEDIATE: "Analysis of Variance. F-test: between-group / within-group variance. Post-hoc for pairwise.",
                ExpertiseLevel.ADVANCED: "Assumptions: normality, homogeneity, independence. Repeated measures for within-subject. Mixed models flexible.",
            },
            "related_terms": ["t_test", "variance", "post_hoc"],
        },
        "t_test": {
            "term": "T-Test",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Comparing averages of two groups.",
                ExpertiseLevel.INTERMEDIATE: "t = (x̄₁-x̄₂)/SE. Independent or paired. Assumes normal distribution.",
                ExpertiseLevel.ADVANCED: "Welch's for unequal variance. Effect size: Cohen's d. Non-parametric: Mann-Whitney, Wilcoxon.",
            },
            "related_terms": ["anova", "mean", "significance"],
        },
        "regression": {
            "term": "Regression",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Finding relationship between variables.",
                ExpertiseLevel.INTERMEDIATE: "y = a + bx for linear. R² indicates fit quality. Predict y from x.",
                ExpertiseLevel.ADVANCED: "Multiple regression, polynomial, logistic. Assumptions: linearity, homoscedasticity, normality of residuals.",
            },
            "formula": "y = a + bx",
            "related_terms": ["correlation", "r_squared", "prediction"],
        },
        "r_squared": {
            "term": "R-Squared",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How much of the variation is explained.",
                ExpertiseLevel.INTERMEDIATE: "Coefficient of determination. R² = 1 - SS_res/SS_tot. 0-1 range.",
                ExpertiseLevel.ADVANCED: "Adjusted R² penalizes extra predictors. Can be negative for poor models. Not always best metric.",
            },
            "formula": "R² = 1 - SS_res/SS_tot",
            "related_terms": ["regression", "correlation", "explained_variance"],
        },
        "power": {
            "term": "Statistical Power",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Ability to detect a real effect.",
                ExpertiseLevel.INTERMEDIATE: "1 - β (probability of Type II error). Depends on n, effect size, α.",
                ExpertiseLevel.ADVANCED: "A priori power analysis for sample size. Post-hoc power not informative. 80% conventional target.",
            },
            "related_terms": ["effect_size", "sample_size", "significance"],
        },
        "normalization": {
            "term": "Normalization",
            "category": "statistics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Adjusting data to a common scale.",
                ExpertiseLevel.INTERMEDIATE: "Time normalization to 0-100%. Amplitude normalization to MVC. Enables comparison.",
                ExpertiseLevel.ADVANCED: "Z-score: (x-μ)/σ. Min-max: (x-min)/(max-min). Ensemble averaging after normalization.",
            },
            "related_terms": ["emg", "time_series", "comparison"],
        },
    }


def get_additional_dynamics_terms() -> GlossaryData:
    """Get additional dynamics and mechanics terms."""
    return {
        "rigid_body": {
            "term": "Rigid Body",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "An object that doesn't bend or deform.",
                ExpertiseLevel.INTERMEDIATE: "Idealization: distance between any two points constant. 6 DOF (3 translation, 3 rotation).",
                ExpertiseLevel.ADVANCED: "Configuration: SE(3). Euler's equations for rotational dynamics. Assumed for skeletal segments.",
            },
            "related_terms": ["segment", "degrees_of_freedom", "kinematics"],
        },
        "free_body_diagram": {
            "term": "Free Body Diagram",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A picture showing all forces on an object.",
                ExpertiseLevel.INTERMEDIATE: "Isolate body, draw all external forces and moments. Foundation for dynamics.",
                ExpertiseLevel.ADVANCED: "Include gravitational, contact, joint reaction forces. Sign conventions critical.",
            },
            "related_terms": ["force", "newton_laws", "equilibrium"],
        },
        "equilibrium": {
            "term": "Equilibrium",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "When forces balance and nothing accelerates.",
                ExpertiseLevel.INTERMEDIATE: "ΣF = 0, ΣM = 0. Static equilibrium for solving unknown forces.",
                ExpertiseLevel.ADVANCED: "Stable, unstable, neutral equilibrium. Dynamic equilibrium: constant velocity.",
            },
            "related_terms": ["force", "moment", "statics"],
        },
        "newton_laws": {
            "term": "Newton's Laws",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Three laws describing how forces cause motion.",
                ExpertiseLevel.INTERMEDIATE: "1: Inertia. 2: F=ma. 3: Action-reaction. Foundation of classical mechanics.",
                ExpertiseLevel.ADVANCED: "Formulated for point masses. Extend to rigid bodies via Euler. Valid in inertial frames.",
            },
            "formula": "F = ma",
            "related_terms": ["force", "acceleration", "inertia"],
        },
        "inertia": {
            "term": "Inertia",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Resistance to change in motion.",
                ExpertiseLevel.INTERMEDIATE: "More massive objects harder to accelerate. Newton's first law.",
                ExpertiseLevel.ADVANCED: "Translational: mass m. Rotational: inertia tensor I. F=ma, τ=Iα.",
            },
            "related_terms": ["mass", "moment_of_inertia", "newton_laws"],
        },
        "force": {
            "term": "Force",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A push or pull on an object.",
                ExpertiseLevel.INTERMEDIATE: "Vector quantity. Causes acceleration (F=ma). Units: Newtons.",
                ExpertiseLevel.ADVANCED: "Contact vs field forces. Superposition principle. Resultant and components.",
            },
            "units": "N (Newtons)",
            "related_terms": ["newton_laws", "mass", "acceleration"],
        },
        "moment": {
            "term": "Moment (Torque)",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "A twisting force that causes rotation.",
                ExpertiseLevel.INTERMEDIATE: "M = F × r (force times distance). Causes angular acceleration.",
                ExpertiseLevel.ADVANCED: "M = r × F (vector). Point of application matters. Pure couples: equal opposite forces.",
            },
            "formula": "M = r × F",
            "units": "N·m",
            "related_terms": ["force", "angular_acceleration", "lever_arm"],
        },
        "lever_arm": {
            "term": "Lever Arm (Moment Arm)",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Distance from pivot to where force is applied.",
                ExpertiseLevel.INTERMEDIATE: "Perpendicular distance to force line of action. Larger arm = more torque.",
                ExpertiseLevel.ADVANCED: "For muscles: varies with joint angle. Affects mechanical advantage. Imaging-based measurement.",
            },
            "units": "m",
            "related_terms": ["moment", "joint_torque", "mechanical_advantage"],
        },
        "mechanical_advantage": {
            "term": "Mechanical Advantage",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How much a lever multiplies force.",
                ExpertiseLevel.INTERMEDIATE: "Output force / input force. Trade-off with distance moved.",
                ExpertiseLevel.ADVANCED: "Most joints: MA < 1 (speed advantage). Varies with posture. Calculated from moment arms.",
            },
            "related_terms": ["lever_arm", "force", "velocity"],
        },
        "restitution": {
            "term": "Coefficient of Restitution",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "How bouncy a collision is.",
                ExpertiseLevel.INTERMEDIATE: "e = relative velocity after / before. e=1 perfectly elastic, e=0 perfectly plastic.",
                ExpertiseLevel.ADVANCED: "Energy loss: 1-e². Velocity-dependent for real materials. USGA ball limit.",
            },
            "formula": "e = v_sep / v_app",
            "related_terms": ["impact", "energy", "collision"],
        },
        "energy_conservation": {
            "term": "Conservation of Energy",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Energy can't be created or destroyed, only changed.",
                ExpertiseLevel.INTERMEDIATE: "KE + PE = constant (conservative systems). Losses via friction, damping.",
                ExpertiseLevel.ADVANCED: "Work-energy theorem: ΔKE = net work. Dissipation tracked for energy audit.",
            },
            "related_terms": ["kinetic_energy", "potential_energy", "work"],
        },
        "spatial_algebra": {
            "term": "Spatial Algebra",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "Math tools for 3D motion and forces.",
                ExpertiseLevel.INTERMEDIATE: "6D vectors combining linear and angular quantities. Efficient for multibody dynamics.",
                ExpertiseLevel.ADVANCED: "Plücker coordinates. Spatial velocity, force, inertia. Featherstone notation.",
            },
            "related_terms": ["rigid_body", "equations_of_motion", "screw_theory"],
        },
        "featherstone": {
            "term": "Featherstone Algorithm",
            "category": "dynamics",
            "definitions": {
                ExpertiseLevel.BEGINNER: "An efficient way to compute multibody dynamics.",
                ExpertiseLevel.INTERMEDIATE: "Articulated Body Algorithm. O(n) complexity for n bodies.",
                ExpertiseLevel.ADVANCED: "Recursive: base to tip, then tip to base. Used by Pinocchio, MuJoCo, Drake.",
            },
            "related_terms": ["equations_of_motion", "simulation", "physics_engine"],
        },
    }


def get_all_glossary_terms() -> GlossaryData:
    """Get all glossary terms combined.

    Returns:
        Complete dictionary of all glossary terms.
    """
    all_terms: GlossaryData = {}

    # Combine all category functions
    category_functions = [
        get_dynamics_terms,
        get_kinematics_terms,
        get_anatomy_terms,
        get_golf_terms,
        get_simulation_terms,
        get_data_terms,
        get_validation_terms,
    ]

    for get_terms in category_functions:
        all_terms.update(get_terms())

    return all_terms
