# Golf Modeling Suite - Research Ideas & Scientific Roadmap

**Last Updated**: 2026-02-18

This document serves as the central registry for scientific research topics, technical resources, and implementation ideas for the Golf Modeling Suite. It focuses on rigorous, scientifically grounded concepts in biomechanics, physics, and engineering.

## 1. Biomechanics & Human Movement

### Kinematic Sequence Analysis

- **Proximal Braking Efficiency**: Quantify the deceleration rates of proximal segments (pelvis, thorax) during the downswing. Research suggests that efficient energy transfer requires rapid deceleration of heavy segments to accelerate distal ones (whip effect).

  - _Data Needed_: Angular velocity time-series for pelvis, thorax, arm, and club.
  - _Outcome_: A "Braking Efficiency" metric ($rad/s^2$) to identify energy leaks.
  - _Ref_: Nesbit, S. M. (2005). "A three dimensional kinematic and kinetic study of the golf swing."

- **X-Factor Stretch**: Measure the increase in pelvis-thorax separation angle during the transition phase (start of downswing). This "stretch-shortening cycle" is correlated with increased clubhead speed.

  - _Data Needed_: 3D orientation of pelvis and thorax during transition.
  - _Outcome_: Quantification of elastic energy potential.
  - _Ref_: Cheetham, P. J., et al. (2001). "The importance of stretching the 'X-Factor' in the downswing of golf."

- **Ground Reaction Force (GRF) Efficiency**: Calculate the ratio of peak Vertical GRF to Clubhead Speed. This measures how effectively a golfer uses the ground to generate power.

  - _Data Needed_: Force plate data (Vertical, A-P, M-L forces) and clubhead speed.
  - _Outcome_: Efficiency ratio to guide strength vs. technique training.

- **Force Vector Steering**: Analysis of how the ground reaction force vector is directed relative to the Center of Mass (CoM). This steering mechanism is critical for generating angular momentum (torque = r x F) about the CoM.
  - _Data Needed_: 3D GRF vector and whole-body CoM position.
  - _Outcome_: Understanding of rotational power generation mechanics.
  - _Ref_: Kwon, Y.-H. (2018). "Ground reaction force and moment."

- **Center of Pressure (CoP) Patterns**: Categorize foot pressure patterns (e.g., "Linear", "Heel-to-Toe", "Back-Foot") and correlate them with swing styles and power sources.
  - _Data Needed_: Force plate CoP coordinates ($x(t), y(t)$) throughout the swing.
  - _Outcome_: Classification algorithm linking ground interaction style to club delivery metrics.
  - _Ref_: Ball, K., & Best, R. (2007). "Centre of pressure patterns in golf swings."

### Energy & Coordination

- **Inter-segmental Power Flow**: Calculate the flow of energy between body segments using joint power analysis ($P = M \cdot \omega$). Positive power indicates energy generation, while negative power indicates absorption or transfer via the kinetic chain.

  - _Data Needed_: Inverse dynamics output (Net Joint Moments) and angular velocities.
  - _Outcome_: Quantify "Kinetic Chain" efficiency and identify energy blockages.
  - _Ref_: Winter, D. A. (2009). "Biomechanics and Motor Control of Human Movement."

- **Muscle Synergy Identification**: Use Non-negative Matrix Factorization (NNMF) on EMG or simulated muscle activations to identify low-dimensional motor primitives (synergies) that control complex movements.
  - _Data Needed_: Time-series activation data from multiple muscle groups.
  - _Outcome_: Understanding of motor control strategies and coordination complexity.
  - _Ref_: d'Avella, A., et al. (2003). "Combinations of muscle synergies in the construction of a natural motor behavior."

- **Joint Quasi-Stiffness**: Estimate the dynamic joint stiffness (Slope of the Moment vs. Angle curve) during the swing. This quasi-stiffness reflects the net effect of passive structures and active muscle contraction.
  - _Data Needed_: Joint moments and angles time-series.
  - _Outcome_: Assessment of joint stability and injury risk.
  - _Ref_: Latash, M. L., & Zatsiorsky, V. M. (1993). "Joint stiffness: Myth or reality?"

- **Grip Pressure Profiling**: Measure the dynamic grip pressure distribution ($P(t)$) at the hands. Grip tension affects wrist mobility (ROM) and clubface control during the release.
  - _Data Needed_: Pressure sensor grid data mapped to hand segments.
  - _Outcome_: Identification of tension-induced speed leaks or face control issues.
  - _Ref_: Komi, E. R., et al. (2008). "Grip force during the golf swing."

### Injury & Learning

- **Footwear Traction Modeling**: Simulate the interaction between specific cleat patterns (geometry) and turf shear strength. Slippage risk is a function of normal force and shear stress.
  - _Data Needed_: Turf shear modulus and friction coefficients for various shoe outsoles.
  - _Outcome_: "Stability Index" for shoe fitting and injury prevention.

- **Heart Rate Variability (HRV) Correlation**: Link physiological arousal (HRV, LF/HF ratio) to tempo consistency and decision-making under pressure.
  - _Data Needed_: ECG time-series and shot outcomes.
  - _Outcome_: Biofeedback protocols for mental game training.

- **Muscle-Tendon Strain Modeling**: Quantify strain in specific high-risk muscle groups (e.g., lead adductor magnus, trailing internal oblique) using Hill-type muscle models to predict acute injury risk beyond generic joint stress.
  - _Data Needed_: Musculoskeletal geometry and kinematics.
  - _Outcome_: Specific "Strain Hazard" map for injury prevention.

- **Differential Learning Protocols**: Simulate the effect of movement variability on motor learning and robustness. Instead of minimizing error, this approach injects noise into target parameters to explore the solution space.
  - _Data Needed_: Stochastic target generation algorithms.
  - _Outcome_: Training protocols that accelerate skill acquisition.
  - _Ref_: Schöllhorn, W. I. (1999). "Individualality of human movements."

- **Quiet Eye Quantification**: Measure the "Quiet Eye" duration (fixation on ball before initiation). Longer fixation durations are strongly correlated with putting success and elite performance.
  - _Data Needed_: Eye-tracking coordinates $(x,y)$ and event timing.
  - _Outcome_: Cognitive state assessment and focus training metrics.
  - _Ref_: Vickers, J. N. (2007). "Perception, Cognition, and Decision Training."

## 2. Ball Flight Physics

### Aerodynamics

- **Boundary Layer Transition**: Explicitly model the movement of the laminar-turbulent transition point on the ball surface as a function of Reynolds number and surface roughness.
  - _Data Needed_: Surface roughness parameters and transition criteria.
  - _Outcome_: Precise drag prediction across critical Reynolds numbers (Driver vs. Wedge).

- **Knuckleball Aerodynamics**: Model the erratic lateral forces on low-spin shots caused by asymmetric flow separation (von Kármán vortex shedding).
  - _Data Needed_: Lateral force coefficients ($C_Y$) at low spin rates.
  - _Outcome_: Accurate simulation of "dead knuckle" shots.

- **Spin Decay Modeling**: Implement exponential decay of spin rate during flight. The current model uses constant spin, but real golf balls lose spin due to air friction, affecting lift and drag coefficients over the trajectory.

  - _Data Needed_: Time-dependent spin rate decay functions ($d\omega/dt$).
  - _Outcome_: More accurate carry distance and landing angle predictions, especially for long drives.
  - _Ref_: Smits, A. J., & Smith, D. R. (1994). "Aerodynamics of the Golf Ball."

- **Environmental Gradient Modeling**: Model wind shear (boundary layer) and air density changes with altitude. Ball flight apex can reach 30m+, where wind speed significantly differs from ground level.

  - _Data Needed_: Wind profile power law exponents.
  - _Outcome_: Improved accuracy for high-launching shots.

- **Turbulence Modeling**: Model the effect of atmospheric turbulence intensity on the drag coefficient. High turbulence can trip the boundary layer earlier, potentially reducing drag (drag crisis) at lower speeds or increasing it via instability.
  - _Data Needed_: Turbulence intensity parameters and modified Cd curves.
  - _Outcome_: Robustness of trajectory prediction in gusty conditions.

- **Variable Aerodynamic Coefficients**: Implement dynamic Lift ($C_L$) and Drag ($C_D$) coefficients that vary with Reynolds number ($Re$) and Spin Ratio ($S$). Constant coefficients fail to capture the "drag crisis" or low-speed behavior.
  - _Data Needed_: $C_L$ and $C_D$ surfaces as functions of $Re$ and $S$.
  - _Outcome_: High-fidelity trajectory simulation across all ball speeds (driver vs. wedge).
  - _Ref_: Bearman, P. W., & Harvey, J. K. (1976). "Golf ball aerodynamics."

- **Hydrodynamic Lubrication (Wet Play)**: Model the water film thickness at impact and its drastic reduction of friction/spin generation ($ \mu_{wet} \ll \mu_{dry} $).
  - _Data Needed_: Water film thickness vs. impact pressure.
  - _Outcome_: Accurate "Wet Weather" mode predicting flyer lies and reduced spin.
  - _Ref_: Cross, R. (2004). "Physics of baseball and softball." (Relevant friction principles).

- **Mud Ball Physics**: Simulate the flight of a ball with asymmetric mass distribution or surface roughness (mud adherence). This creates a wobbling spin axis and erratic lift forces.
  - _Data Needed_: Perturbed inertia tensor and localized drag coefficients.
  - _Outcome_: Simulation of "mud ball" deviation and flight stability analysis.

- **Dimple Geometry Optimization**: Use surrogate models to predict aerodynamic coefficients ($C_L, C_D$) for custom dimple patterns without wind tunnel testing.
  - _Data Needed_: CFD training data linking geometry features to aero coefficients.
  - _Outcome_: Rapid prototyping of ball designs.

### Trajectory

- **Bounce and Roll Physics**: Implement a rigid-body collision model for the ball-ground interaction, accounting for turf compliance (COR), friction, and slope.

  - _Data Needed_: Coefficients of restitution and friction for various turf types (fairway, green, rough).
  - _Outcome_: Prediction of total distance (Carry + Roll).

- **Trajectory Optimization**: Implement an optimizer (e.g., SQP or Genetic Algorithm) to find the optimal Launch Angle and Spin Rate for a given Ball Speed and Environmental Condition to maximize Carry or Total Distance.

  - _Data Needed_: Ball Flight Simulator and bounds for launch conditions.
  - _Outcome_: "Optimal Flight" recommendations for fitting.

- **Lie-Dependent Spin Generation**: Model the reduction in friction and spin generation caused by grass entrapment (Flyer Lie) or wet conditions.
  - _Data Needed_: Empirical friction coefficients for different lie conditions (fairway, rough, wet).
  - _Outcome_: Accurate prediction of "flyers" and run-out from rough.

## 3. Equipment Science

### Club Dynamics

- **Impact Acoustics ("Sound is Feel")**: Simulate the frequency spectrum of impact sound based on clubhead eigenmodes and material properties. Players perceive "feel" largely through sound.
  - _Data Needed_: Modal analysis frequencies and damping ratios.
  - _Outcome_: "Sound Signature" analysis for driver design.

- **Composite Layup Optimization**: Use Genetic Algorithms to optimize the orientation of carbon fiber layers (plies) in shafts to achieve specific bending and torsional profiles.
  - _Data Needed_: Lamina properties ($E_1, E_2, G_{12}, \nu_{12}$) and target stiffness matrix.
  - _Outcome_: Custom shaft designs with tailored kick points and torque.

- **Grip Friction Degradation**: Model the time-dependent reduction in coefficient of friction due to sweat accumulation and rubber aging/oxidation.
  - _Data Needed_: Friction coefficients vs. moisture content and age.
  - _Outcome_: Recommendations for grip replacement frequency.

- **Shaft Torsional Dynamics**: Model the twisting (torque) of the shaft during the downswing and impact. High-torque shafts can close the face more rapidly but may be less stable.

  - _Data Needed_: Shaft torsional stiffness (GJ) profile.
  - _Outcome_: Analysis of dynamic face closure rates.
  - _Ref_: MacKenzie, S. J., & Sprigings, E. J. (2009). "A three-dimensional forward dynamics model of the golf swing."

- **Coupled Bending-Torsion Shaft Model**: Extend the flexible shaft model to include torsional degrees of freedom and the coupling between bending and torsion (especially for non-axisymmetric shafts or off-axis loading).

  - _Data Needed_: Polar Moment of Inertia ($J$) and Shear Modulus ($G$) profiles.
  - _Outcome_: Analysis of dynamic face closure variability due to shaft twist.

- **Shaft Spine & Asymmetry**: Model non-uniform bending stiffness ($EI_{xx} \neq EI_{yy}$) caused by manufacturing tolerances ("spine"). This causes the shaft to bend out of the swing plane even with in-plane loading.
  - _Data Needed_: Shaft oscillation frequency in multiple planes (FLO).
  - _Outcome_: Prediction of impact inconsistency due to shaft orientation (puring).

- **Clubhead MOI Tensor**: Replace point-mass clubhead approximations with a full 3D Moment of Inertia tensor. This is critical for accurately predicting the gear effect on off-center hits.

  - _Data Needed_: CAD-derived MOI tensors ($I_{xx}, I_{yy}, I_{zz}, I_{xy}, \dots$) for standard clubheads.
  - _Outcome_: Accurate prediction of ball start line and spin axis tilt for toe/heel strikes.
  - _Ref_: United States Golf Association (USGA) Technical Protocols.

- **Full Rigid Body Impact**: Implement a full 3D rigid body collision model using the Clubhead Inertia Tensor ($I_{club}$) instead of a point mass approximation. This naturally captures gear effect physics without empirical factors.

  - _Data Needed_: Full Inertia Tensor ($I_{xx}, I_{yy}, I_{zz}, I_{xy}, \dots$) and CG location.
  - _Outcome_: Physics-based prediction of gear effect and sweet spot sensitivity.

- **Clubhead Aerodynamics**: Model the aerodynamic drag forces on the clubhead itself during the downswing. Bulky driver heads can experience significant drag near impact ($v > 100$ mph), reducing maximum speed.
  - _Data Needed_: Drag area ($C_D A$) of the clubhead vs. yaw/pitch angles.
  - _Outcome_: Calculation of clubhead speed loss due to head geometry.

- **Kick Point Optimization**: Analyze shaft EI profiles to determine the dynamic "kick point" and its effect on dynamic loft.

  - _Data Needed_: EI profiles (flexural stiffness) along the shaft.
  - _Outcome_: Algorithm to match shaft profiles to swing speed/tempo for optimal launch conditions.

- **Shaft Droop & Lead Deflection**: Model the "droop" (vertical bending) and "lead" (forward bending) caused by the clubhead's center of gravity offset (CG bias). This alters the dynamic lie and loft at impact compared to static measurements.
  - _Data Needed_: Clubhead CG coordinates relative to the hosel; Shaft stiffness matrices.
  - _Outcome_: Accurate prediction of impact position and dynamic face angle.
  - _Ref_: Mather, J. S. B. (2000). "The role of the shaft in the golf swing."

- **Bulge & Roll Optimization**: Optimize the horizontal (Bulge) and vertical (Roll) radii of the clubface to minimize dispersion for a specific player's impact pattern (Gear Effect compensation).
  - _Data Needed_: Impact distribution heatmaps and launch conditions.
  - _Outcome_: Custom face geometry recommendations for "shot correction".

- **Movable Weight Dynamics**: Model the shift in Center of Gravity (CG) and MOI tensor when moving discrete weights (e.g., sliding tracks).
  - _Data Needed_: Weight mass, track geometry, and base head properties.
  - _Outcome_: Prediction of shot shape bias (Draw/Fade) and stability changes.

### Ball & Face Mechanics

- **Multi-Layer Compression Dynamics**: Model the interaction between core, mantle, and cover layers to predict ball speed and spin separation. Finite Element or lumped-parameter modeling of deformation.
  - _Data Needed_: Viscoelastic properties of Polybutadiene (core) and Urethane (cover).
  - _Outcome_: Prediction of "Feel" and performance differences between 2-piece and 4-piece balls.

- **Groove-Edge Sharpness Degradation**: Model the wear of groove radii over time and its effect on "Launch Ratio" (Launch Angle / Dynamic Loft) and Spin Rate, particularly for wedges.
  - _Data Needed_: Tribological wear rates for varying steel hardness.
  - _Outcome_: "Wedge Lifespan" predictor based on practice volume.

## 4. Statistical Methods

### Analytics

- **Clutch Performance Index**: Quantify performance variance in high-pressure situations (e.g., last 3 holes, hazards in play) compared to baseline.
  - _Data Needed_: Shot outcomes tagged with "pressure level" context.
  - _Outcome_: Identification of "clutch" players vs. those who choke.

- **Weather-Adjusted Strokes Gained**: Normalize Strokes Gained baselines for environmental conditions (air density, wind, temperature) to isolate player skill from conditions.
  - _Data Needed_: Historical weather data matched to shot data.
  - _Outcome_: Fairer performance comparison across different venues/seasons.

- **Strokes Gained Baseline**: Develop a "Strokes Gained" implementation.

  - _Data Needed_: Baseline dataset of "shots to hole" from various distances and lies (e.g., Broadie's tables).
  - _Outcome_: Context-aware performance metrics.
  - _Ref_: Broadie, M. (2014). "Every Shot Counts."

- **Impact Location Heatmaps**: Generate 2D density plots of face impact locations.

  - _Data Needed_: Series of $(x, y)$ impact coordinates on the clubface.
  - _Outcome_: Visual tool to assess ball striking consistency.

- **Monte Carlo Strategy Engine**: Simulate thousands of shots from a specific lie using a dispersion model (covariance matrix) to calculate expected strokes-to-hole for various aim lines.

  - _Data Needed_: Shot dispersion statistics (ellipses) by club and lie condition.
  - _Outcome_: Optimal aim lines (Risk/Reward analysis) for course management.

- **Bayesian Parameter Estimation**: Use Bayesian inference (MCMC or Variational Inference) to estimate biomechanical parameters (e.g., max isometric force, tendon slack length) from motion capture data, providing uncertainty bounds.

  - _Data Needed_: Motion data and priors for physiological parameters.
  - _Outcome_: Personalized musculoskeletal models with confidence intervals.

- **Causal Discovery**: Apply causal inference algorithms (e.g., PC algorithm) to observational swing data to distinguish between correlations (e.g., "early extension is linked to slice") and causal chains.
  - _Data Needed_: Large dataset of swing metrics and outcomes.
  - _Outcome_: Identification of root causes vs. symptoms in swing faults.

- **Tempo Ratios**: Statistical analysis of Backswing time to Downswing time ratio (Tempo) and its correlation with performance consistency.
  - _Data Needed_: Swing event timestamps (Takeaway, Top, Impact) from large dataset.
  - _Outcome_: Validation of the "3:1 Tempo" rule and identification of player-specific optimal tempos.
  - _Ref_: Novosel, J., & Garrity, J. (2004). "Tour Tempo."

- **Fatigue-Induced Kinematic Drift**: Model the progressive degradation of peak power and coordination (sequencing) over a simulation of 18 holes (approx. 4 hours).
  - _Data Needed_: Decay constants for fast-twitch muscle fibers.
  - _Outcome_: Prediction of "Back 9" scoring collapse risks.

- **Synthetic Computer Vision Data**: Use the biomechanics engine to render synthetic video frames with perfect ground truth labels (joint centers) to train pose estimation models.
  - _Data Needed_: High-fidelity 3D golfer meshes and textures.
  - _Outcome_: Robust "Sim2Real" transfer for markerless motion capture.

- **Swing Signature Clustering**: Use Unsupervised Learning (e.g., K-Means, Hierarchical) to group swings into distinct styles (e.g., "Glider", "Spinner", "Launcher") based on kinematic feature vectors.
  - _Data Needed_: Large database of kinematic sequences.
  - _Outcome_: Tailored coaching and equipment recommendations based on swing type.

- **Green Reading Algorithms**: Simulate ball rolling physics on non-planar surfaces using localized gravity vectors to calculate the "Aim Point" and effective break.
  - _Data Needed_: Lidar or photogrammetry point clouds of greens.
  - _Outcome_: AR overlay of optimal putting lines.
  - _Ref_: Penner, A. R. (2002). "The physics of putting."

## 5. Simulation Technology

### Physics Engine

- **Finite Element Impact**: Implement a simplified Finite Element (FE) or discrete element model for the clubface to generate a Coefficient of Restitution (COR) map, rather than a single scalar COR.

  - _Data Needed_: Material properties (Young's Modulus, Poisson's ratio) and face thickness profile.
  - _Outcome_: Accurate smash factor prediction across the entire face (Variable Face Thickness modeling).

- **Soft-Body Ball Compression**: Implement Explicit FEM or Material Point Method (MPM) modeling of the golf ball's core deformation at impact. This captures hysteresis (energy loss) and heat generation more accurately than simple COR coefficients.
  - _Data Needed_: Hyperelastic material models (e.g., Mooney-Rivlin) for ball layers.
  - _Outcome_: Detailed contact mechanics and compression/restitution analysis.

- **Doppler Radar Emulation**: Simulate the raw radial velocity data seen by radar launch monitors (e.g., TrackMan) to study the difference between "measured" and "actual" impact parameters (e.g., Spin Loft vs. Dynamic Loft).

  - _Data Needed_: Relative velocity vectors of clubhead geometric center vs. radar origin.
  - _Outcome_: Synthetic validation environment for launch monitor algorithms.

- **Neural ODE Surrogate Models**: Train Neural Ordinary Differential Equations (Neural ODEs) to approximate the physics engine's output. This allows for differentiable simulation and drastically faster execution for real-time applications.
  - _Data Needed_: Large dataset of physics engine simulations (Input-Output pairs).
  - _Outcome_: Real-time trajectory prediction on mobile devices.

- **Granular Media (Bunker Physics)**: Implement Discrete Element Method (DEM) or continuum granular models for club-sand interaction.
  - _Data Needed_: Sand particle size distribution and friction angles.
  - _Outcome_: Accurate simulation of splash shots and energy dissipation in bunkers.

- **Sensor Fusion (Radar + Optical)**: Combine Doppler Radar (TrackMan) and Optical (Camera) data using Kalman Filtering to resolve discrepancies (e.g., Spin Axis) and improve robustness.
  - _Data Needed_: Synchronized streams from multiple sensor types with known covariance.
  - _Outcome_: "Ground Truth" generation from imperfect sensors.

### Haptics & Immersion

- **Procedural Audio Synthesis**: Generate real-time impact sounds in the physics engine based on collision impulse, material stiffness, and resonance.
  - _Data Needed_: Impulse response functions for club-ball collisions.
  - _Outcome_: Realistic audio feedback in simulation without pre-recorded samples.

- **Atmospheric Scattering**: Render physically-based atmospheric haze and aerial perspective to improve depth perception in VR environments.
  - _Data Needed_: Scattering coefficients (Rayleigh/Mie) for local weather.
  - _Outcome_: Enhanced visual realism and distance estimation cues.

- **Haptic Feedback Modeling**: Calculate force-feedback vectors for VR controllers to simulate impact feel (vibration frequency and amplitude).
  - _Data Needed_: Impact impulse and shaft vibration modes.
  - _Outcome_: Immersive training in VR environments.

## 6. Control Theory

### Robotics

- **Swing Robot Inverse Dynamics**: Calculate the required joint torques to drive a double-pendulum model along a desired kinematic path.

  - _Data Needed_: Target kinematic sequence (angular positions/velocities).
  - _Outcome_: Control inputs for a robotic swing device or biomechanical simulation.

- **Iterative Learning Control (ILC)**: Apply ILC algorithms to robotic swing simulations. By using the error history from previous swings, the controller "learns" the optimal input to track a target trajectory perfectly.
  - _Data Needed_: Error vectors from repeated trials.
  - _Outcome_: Rapid convergence to target swing parameters for robot testing automation.

- **Neuromuscular Noise Modeling**: Introduce signal-dependent noise into muscle torque actuators ($\sigma \propto u$) to simulate human motor variability. This reproduces the "speed-accuracy tradeoff" (Fitts's Law).
  - _Data Needed_: Noise scaling constants for different muscle groups.
  - _Outcome_: Realistic dispersion patterns generated from biomechanical simulations.
  - _Ref_: Harris, C. M., & Wolpert, D. M. (1998). "Signal-dependent noise determines motor planning."

- **Policy Gradient Swing Optimization**: Use Proximal Policy Optimization (PPO) to find optimal muscle activation patterns that maximize carry distance while minimizing injury risk penalties.
  - _Data Needed_: Reward function balancing distance, accuracy, and joint stress.
  - _Outcome_: Identification of theoretically optimal swing mechanics.

---

## Workflow Log

| Date       | Topic Added                                                                                             | Category | Status |
| ---------- | ------------------------------------------------------------------------------------------------------- | -------- | ------ |
| 2026-01-29 | Initial Population of Research Ideas                                                                    | All      | Active |
| 2026-02-01 | Added CoP, Aero Coeffs, Shaft Droop, Monte Carlo, Radar, Noise                                          | All      | Active |
| 2026-02-13 | Added Power Flow, Muscle Synergy, Trajectory Opt, Flyer Lie, Rigid Impact, Bayesian, Causal, Neural ODE | All      | Active |
| 2026-02-13 | Added Muscle Strain, Wet Play, Dimple Opt, Multi-Layer Ball, Fatigue, Synthetic Data, PPO               | All      | Active |
| 2026-02-14 | Added Force Vector, Stiffness, Turbulence, Mud Ball, Spine, Head Aero, Tempo, Soft Body, ILC            | All      | Active |
| 2026-02-15 | Added Grip Pressure, Quiet Eye, Bulge/Roll Opt, Movable Weights, Clustering, Green Reading, Bunker, Fusion | All | Active |
| 2026-02-18 | Added Footwear, HRV, Boundary Layer, Knuckleball, Acoustics, Layup, Grip Friction, Clutch Index, Weather SG, Audio, Scattering | All | Active |
