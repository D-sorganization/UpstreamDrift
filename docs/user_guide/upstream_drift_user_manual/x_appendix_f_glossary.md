## Appendix F: Glossary

Definitions of technical terms, acronyms, and domain-specific vocabulary
used throughout the UpstreamDrift documentation and codebase. Terms are
organized alphabetically.

---

### Acronyms

| Acronym    | Full Form                                       | Definition                                                                                                                                                                                                                               |
| ---------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ABA**    | Articulated Body Algorithm                      | $O(n)$ algorithm for computing forward dynamics of articulated rigid-body systems. Computes joint accelerations given joint positions, velocities, and applied forces. Used internally by MuJoCo and Pinocchio.                          |
| **ASGI**   | Asynchronous Server Gateway Interface           | Python web server interface standard. UpstreamDrift uses ASGI via uvicorn for the FastAPI server.                                                                                                                                        |
| **CMA-ES** | Covariance Matrix Adaptation Evolution Strategy | Derivative-free optimization algorithm that adapts a multivariate Gaussian search distribution. Used for swing parameter optimization when gradients are unavailable.                                                                    |
| **COM**    | Center of Mass                                  | The mass-weighted average position of all particles in a body. Critical for balance analysis and ground reaction force computation.                                                                                                      |
| **COP**    | Center of Pressure                              | The point on the ground surface where the resultant ground reaction force acts. Moves between the feet during the golf swing.                                                                                                            |
| **COR**    | Coefficient of Restitution                      | Ratio of relative separation speed to relative approach speed in a collision. The USGA limits driver COR to 0.830.                                                                                                                       |
| **CRBA**   | Composite Rigid Body Algorithm                  | $O(n^2)$ algorithm for computing the joint-space mass matrix $M(q)$ of an articulated system.                                                                                                                                            |
| **DAgger** | Dataset Aggregation                             | An imitation learning algorithm that iteratively collects data under the learned policy and retrains. Addresses distribution shift in behavior cloning.                                                                                  |
| **DCM**    | Divergent Component of Motion                   | A point derived from the COM and its velocity that characterizes dynamic balance. If the DCM stays within the support polygon, the system can maintain balance. Defined as $\xi = c + \dot{c}/\omega_0$ where $\omega_0 = \sqrt{g/z_c}$. |
| **DOF**    | Degrees of Freedom                              | The number of independent parameters that define the configuration of a mechanical system. A free rigid body in 3D has 6 DOF.                                                                                                            |
| **EMG**    | Electromyography                                | Measurement of electrical activity in muscles. Used for validation of muscle activation predictions and EMG-driven simulation.                                                                                                           |
| **FK**     | Forward Kinematics                              | Computing the position and orientation of end-effectors given joint angles. The mapping $q \mapsto \mathbf{x}$.                                                                                                                          |
| **FPS**    | Frames Per Second                               | Rate of simulation visualization or video capture. UpstreamDrift renders at 60 FPS by default.                                                                                                                                           |
| **GAIL**   | Generative Adversarial Imitation Learning       | An imitation learning algorithm that uses a discriminator to distinguish expert demonstrations from learned behavior. Combines adversarial training with reinforcement learning.                                                         |
| **GRF**    | Ground Reaction Force                           | The force exerted by the ground on a body in contact with it. Equal and opposite to the force the body exerts on the ground. Measured by force plates in biomechanics.                                                                   |
| **ICP**    | Instantaneous Capture Point                     | Synonymous with DCM. The point on the ground where a robot (or golfer) must step to come to a complete stop.                                                                                                                             |
| **IK**     | Inverse Kinematics                              | Computing joint angles that achieve a desired end-effector position/orientation. The mapping $\mathbf{x} \mapsto q$. May have zero, one, or many solutions.                                                                              |
| **IMU**    | Inertial Measurement Unit                       | A sensor combining accelerometers and gyroscopes to measure linear acceleration and angular velocity. Used for motion capture and swing analysis.                                                                                        |
| **IPOPT**  | Interior Point Optimizer                        | A large-scale nonlinear programming solver using interior-point/barrier methods. Available through Drake for trajectory optimization.                                                                                                    |
| **MJCF**   | MuJoCo Format                                   | XML-based model description format native to MuJoCo. Supports contacts, actuators, sensors, and tendon routing.                                                                                                                          |
| **MOI**    | Moment of Inertia                               | Resistance to angular acceleration about an axis. The rotational analog of mass.                                                                                                                                                         |
| **MPC**    | Model Predictive Control                        | A control strategy that solves a finite-horizon optimal control problem at each time step, applying only the first control action. Provides feedback by re-solving with updated state information.                                       |
| **NLP**    | Nonlinear Program                               | An optimization problem with a nonlinear objective function and/or nonlinear constraints. Trajectory optimization problems are typically NLPs.                                                                                           |
| **PPO**    | Proximal Policy Optimization                    | A reinforcement learning algorithm that constrains policy updates to a trust region. Used for training swing policies in the MyoSuite integration.                                                                                       |
| **QP**     | Quadratic Program                               | An optimization problem with a quadratic objective and linear constraints. Solved as a subproblem in SQP methods and for contact force resolution.                                                                                       |
| **RL**     | Reinforcement Learning                          | A machine learning paradigm where an agent learns to make decisions by interacting with an environment and receiving rewards.                                                                                                            |
| **RNEA**   | Recursive Newton-Euler Algorithm                | $O(n)$ algorithm for computing inverse dynamics. Given joint positions, velocities, and accelerations, computes the required joint torques.                                                                                              |
| **RK4**    | Runge-Kutta 4th Order                           | A fourth-order numerical integration method. Used by UpstreamDrift for ball flight trajectory integration.                                                                                                                               |
| **SDF**    | Simulation Description Format                   | An XML-based format for describing robots and environments, originating from the Gazebo simulator. Supports world-level descriptions including lighting, terrain, and multiple models.                                                   |
| **SE(3)**  | Special Euclidean Group (3D)                    | The group of rigid body transformations in 3D, combining rotation $SO(3)$ and translation $\mathbb{R}^3$. An element of $SE(3)$ represents a pose.                                                                                       |
| **SLSQP**  | Sequential Least Squares Programming            | A gradient-based constrained optimization algorithm. Used for small-to-medium scale optimization in UpstreamDrift via SciPy.                                                                                                             |
| **SLERP**  | Spherical Linear Interpolation                  | A method for smoothly interpolating between two rotations on the unit quaternion sphere. Constant angular velocity interpolation.                                                                                                        |
| **SNOPT**  | Sparse Nonlinear Optimizer                      | A large-scale SQP-based nonlinear programming solver. Available through Drake for trajectory optimization with sparse constraint Jacobians.                                                                                              |
| **SO(3)**  | Special Orthogonal Group (3D)                   | The group of 3D rotations, represented by $3 \times 3$ orthogonal matrices with determinant +1.                                                                                                                                          |
| **SPD**    | Symmetric Positive-Definite                     | A matrix property. The mass matrix $M(q)$ is always SPD for physical systems with positive masses. Ensures unique solutions to linear systems and positive kinetic energy.                                                               |
| **SQP**    | Sequential Quadratic Programming                | An optimization method that solves a sequence of QP subproblems to find the solution of a general NLP. SNOPT is an SQP solver.                                                                                                           |
| **URDF**   | Unified Robot Description Format                | An XML-based format for describing robot kinematics and dynamics. The primary model exchange format in UpstreamDrift. Supported by MuJoCo, Drake, and Pinocchio.                                                                         |
| **ZMP**    | Zero Moment Point                               | The point on the ground surface where the net moment of ground reaction forces and gravity has zero horizontal component. Used for balance analysis. If ZMP stays within the support polygon, the system is statically balanced.         |

---

### Technical Terms

**Activation dynamics**
: The time-dependent process by which neural excitation signals are transformed into muscle activation levels. Modeled as a first-order ODE with asymmetric time constants for activation (~10 ms) and deactivation (~40 ms). See `src/shared/python/activation_dynamics.py` and Appendix A.6.8.

**Admittance control**
: A control paradigm where the controller accepts force inputs and produces motion outputs. The robot behaves as if it has a specified mass-spring-damper system: $F_{ext} = M_d \ddot{x} + D_d \dot{x} + K_d x$. Complementary to impedance control.

**Articulated body inertia**
: The effective inertia of a body in an articulated chain, accounting for the dynamics of all descendant bodies. Used in the ABA for $O(n)$ forward dynamics.

**Attack angle (angle of attack)**
: The vertical angle of the clubhead's path at impact relative to the horizontal. Positive means upward (hitting up on the ball), negative means downward. Typically negative for irons, slightly positive for driver.

**Behavior cloning**
: An imitation learning technique where a neural network policy is trained via supervised learning on expert demonstration data. Simple but suffers from distribution shift (compounding errors).

**Carry distance**
: The distance the golf ball travels through the air from the point of impact to the first landing point, measured along the ground. Excludes roll. Computed in `BallFlightSimulator.calculate_carry_distance()`.

**Club path**
: The horizontal direction of the clubhead's center of gravity movement at impact, relative to the target line. Positive is to the right (for a right-handed golfer), negative is to the left.

**Co-contraction**
: Simultaneous activation of agonist and antagonist muscles crossing a joint. Increases joint stiffness at the cost of metabolic energy. Important for stability during the golf swing transition.

**Collocation (direct)**
: A trajectory optimization method that simultaneously optimizes states and controls at discrete time points (knot points), enforcing dynamics as equality constraints. See Appendix A.5.4.

**Composite rigid body inertia**
: The total inertia of a body and all its descendant bodies when treated as a single rigid body. Used in the CRBA to compute the mass matrix.

**Contact Jacobian**
: The matrix $J_c(q)$ that maps generalized velocities to contact point velocities: $v_{contact} = J_c(q) \dot{q}$. Its transpose maps contact forces to generalized forces: $\tau_{contact} = J_c^T f_c$.

**Damping ratio**
: A dimensionless measure describing how oscillations decay. For a second-order system: $\zeta = c / (2\sqrt{km})$. Critically damped at $\zeta = 1$.

**Design by Contract (DbC)**
: A software design methodology where interfaces are defined by formal, verifiable specifications (contracts): preconditions, postconditions, and invariants. Implemented in `src/shared/python/contracts.py`.

**Direct collocation**
: See "Collocation (direct)."

**Domain randomization**
: A sim-to-real transfer technique where simulation parameters (friction, mass, sensor noise, etc.) are randomly varied during training so the policy generalizes to the real world.

**Dynamic loft**
: The effective loft angle of the club face at the moment of impact, accounting for shaft lean, face angle, and attack angle. Different from the static loft stamped on the club.

**Face angle**
: The horizontal orientation of the club face at impact, relative to the target line. Open (pointing right of target for right-handed), closed (pointing left), or square.

**Flexible shaft model**
: A model that accounts for golf club shaft bending during the swing. UpstreamDrift supports three approaches: rigid (no flex), modal (2-4 bending mode shapes), and finite element (beam elements with 2 DOF per node). See `src/shared/python/flexible_shaft.py`.

**Force closure**
: A contact configuration where the contact forces can resist any external wrench. Requires contacts that can generate forces in enough independent directions.

**Force-length relationship**
: The relationship between a muscle's fiber length and its force-generating capacity. Peak force occurs at the optimal fiber length. Modeled as a Gaussian in Hill-type models. See Appendix A.6.3.

**Force-velocity relationship**
: The relationship between a muscle's shortening/lengthening velocity and its force output. Force decreases with shortening velocity (concentric) and increases above isometric during lengthening (eccentric). See Appendix A.6.4.

**Friction cone**
: The set of contact forces that satisfy Coulomb friction: $\|f_t\| \leq \mu f_n$, where $f_t$ is tangential force, $f_n$ is normal force, and $\mu$ is the friction coefficient. Forms a cone in 3D force space.

**Gear effect**
: The phenomenon where off-center impacts on a club head with a bulging face cause the ball to spin in a direction that curves it back toward the center of the face. Toe hits produce draw spin; heel hits produce fade spin.

**Generalized coordinates**
: The minimal set of independent variables $q$ that completely describe the configuration of a mechanical system. For a planar double pendulum: $q = [\theta_1, \theta_2]^T$.

**Gimbal lock**
: A loss of one degree of freedom in Euler angle representation when two rotation axes align ($\theta = \pm \pi/2$ in ZYX convention). Avoided by using quaternions.

**Grasp matrix**
: A matrix that maps contact forces to the resultant wrench on a grasped object. Used in multi-contact analysis of the golf grip.

**Hill-type muscle model**
: A phenomenological muscle model consisting of a contractile element (active force), parallel elastic element (passive stiffness), and series elastic element (tendon). Named after A.V. Hill (1938). See Appendix A.6.

**Impedance control**
: A control paradigm where the controller accepts motion inputs and produces force outputs. The robot behaves as if it has a virtual mass-spring-damper: $\tau = M_d \ddot{e} + D_d \dot{e} + K_d e$ where $e$ is the position error. Complementary to admittance control.

**Invariant**
: In Design by Contract, a condition that must be true before and after every public method call on an object. Example: "mass must always be positive." See `InvariantError` in `contracts.py`.

**Jacobian matrix**
: The matrix of partial derivatives $J(q) = \partial \mathbf{x} / \partial q$ relating joint velocities to task-space velocities: $\dot{\mathbf{x}} = J(q) \dot{q}$. Fundamental for operational space control and inverse kinematics.

**Kinematic chain**
: A series of rigid bodies (links) connected by joints. Open chains have one fixed base and free end-effectors. Closed chains form loops (e.g., both hands on the golf club).

**Kinematic sequence**
: The temporal ordering of peak angular velocities during the golf swing downswing, typically: pelvis, torso, arm, hand/club. Optimal sequencing maximizes energy transfer to the club.

**Kinetic energy**
: The energy of motion. In generalized coordinates: $T = \frac{1}{2} \dot{q}^T M(q) \dot{q}$. Decomposed into rotational and translational components for analysis.

**Knot point**
: A discrete time point in a trajectory optimization problem where decision variables (states and controls) are defined. Dynamics are enforced between consecutive knot points.

**Lag (club lag / wrist lag)**
: The angle between the lead forearm and the club shaft during the downswing. A large lag angle that is maintained until late in the downswing allows greater clubhead speed at impact.

**Moment arm (muscle)**
: The perpendicular distance from a joint's axis of rotation to the muscle's line of action. Determines the torque a muscle produces about a joint per unit force: $\tau = r \cdot F$. Computed as $r_j = -\partial l_{MT} / \partial q_j$.

**Nullspace**
: The set of joint velocities $\dot{q}$ that produce zero task-space velocity: $J(q)\dot{q} = 0$. Used for secondary task execution (posture optimization, singularity avoidance) in redundant systems.

**Operational space**
: The task space in which end-effector motion is controlled. Operational space control computes joint torques that produce desired end-effector forces/accelerations while accounting for the full dynamics.

**Pennation angle**
: The angle between a muscle's fibers and its tendon direction. Muscles with pennation can pack more fibers but transmit force at $\cos\alpha$ efficiency.

**Postcondition**
: In Design by Contract, a condition that must be true after a method completes. Example: "returned array must contain only finite values." See `PostconditionError` in `contracts.py`.

**Precondition**
: In Design by Contract, a condition that must be true before a method executes. Example: "engine must be initialized before calling step()." See `PreconditionError` in `contracts.py`.

**Proximal**
: Closer to the body's center or the base of a kinematic chain. The shoulder is proximal to the elbow. Opposite of distal.

**Sim-to-real transfer**
: The process of deploying policies trained in simulation to physical robots or real-world systems. Challenges include the "reality gap" due to modeling inaccuracies.

**Singularity (kinematic)**
: A configuration where the Jacobian $J(q)$ loses rank, meaning certain task-space velocities become unachievable. Near singularities, required joint velocities become very large.

**Smash factor**
: The ratio of ball speed to clubhead speed at impact. For a driver: typically 1.48-1.51. Determined by COR and the mass ratio $m_{club} / (m_{club} + m_{ball})$.

**Spatial velocity (twist)**
: A 6D vector combining angular velocity $\omega$ and linear velocity $v$ of a rigid body. See Appendix A.3.1.

**Spin axis tilt**
: The orientation of the ball's spin axis relative to vertical. Zero tilt means pure backspin. Tilt causes the ball to curve laterally (draw/fade).

**Spin loft**
: The 3D angle between the club face normal and the club path direction at impact. The primary determinant of spin rate.

**Stimpmeter**
: A device used to measure green speed in golf. A ball is rolled down a standardized ramp, and the distance it travels (in feet) is the Stimp reading. Fast greens: 10-13 ft.

**Stiffness (joint)**
: In impedance control, the virtual spring constant $K_d$ that defines how much force is produced per unit displacement. Higher stiffness means more resistance to perturbation.

**System identification**
: The process of estimating dynamic model parameters (masses, inertias, friction coefficients) from experimental data. Used to calibrate simulation models against real-world measurements.

**Tendon**
: The elastic tissue connecting muscle to bone. In Hill-type models, tendon compliance affects force transmission and energy storage. Characterized by slack length $l_T^{slack}$ and stiffness.

**Trajectory optimization**
: Finding an optimal trajectory (time-history of states and controls) that minimizes a cost function subject to dynamic constraints. Can be solved by direct methods (collocation, shooting) or indirect methods (Pontryagin's principle).

**Twist**
: See "Spatial velocity."

**Wrench**
: See "Spatial force" (Appendix A.3.2). A 6D vector combining moment $n$ and force $f$.

**X-factor**
: The angular difference between shoulder rotation and hip rotation at the top of the backswing. A larger X-factor indicates greater trunk coil and potential energy storage. Related to the "X-factor stretch" during the early downswing.

---

### Model Format Terms

**URDF (Unified Robot Description Format)**
: An XML format for describing robot kinematic trees with links and joints. Supports visual geometry, collision geometry, and inertial properties. Primary format in ROS ecosystem and the main exchange format for UpstreamDrift models. Limitations: no closed-loop kinematic chains, limited actuator/sensor description.

**MJCF (MuJoCo Format)**
: MuJoCo's native XML format. Richer than URDF: supports tendons, actuator models, contact parameters, equality constraints (closed loops), and sensors. UpstreamDrift auto-converts URDF to MJCF when using the MuJoCo engine.

**SDF (Simulation Description Format)**
: XML format from the Gazebo simulator. Can describe complete worlds with multiple models, lights, and terrain. Supports features beyond URDF including closed kinematic chains and model nesting.

**C3D**
: A binary file format for storing 3D motion capture data (marker positions and analog signals like force plates and EMG). UpstreamDrift reads C3D files via the `ezc3d` library for validation against real swing data.

**STL (Stereolithography)**
: A file format for 3D mesh geometry using triangulated surfaces. Used for collision and visual geometry in URDF models. Both ASCII and binary variants are supported.

---

### Software Engineering Terms

**Design by Contract**
: A methodology introduced by Bertrand Meyer where component interfaces are specified using preconditions, postconditions, and invariants. Violations indicate bugs in the caller (precondition) or implementer (postcondition/invariant). See `src/shared/python/contracts.py`.

**DRY (Don't Repeat Yourself)**
: The principle that every piece of knowledge should have a single, authoritative representation. Violations lead to inconsistency bugs. UpstreamDrift enforces DRY through shared constants, base classes, and the `physics_constants.py` module.

**Editable install**
: A Python package installation mode (`pip install -e .`) where the package is installed as a link to the source directory. Changes to source code take effect immediately without reinstalling.

**Orthogonality**
: The principle that components should be independent -- changing one should not affect others. In UpstreamDrift, physics engines, analysis modules, and the API layer are orthogonal: each can be modified independently.

**PhysicalConstant**
: A custom `float` subclass in UpstreamDrift that carries unit, source citation, and description metadata alongside the numeric value. Defined in `src/shared/python/physics_constants.py`.

**Structured logging**
: A logging approach where log entries are machine-parsable (typically JSON) with typed fields rather than free-form text. UpstreamDrift uses `structlog` for all logging.

---

_End of Appendices A through F._
