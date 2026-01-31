# Golf Modeling Suite - Research Ideas & Scientific Roadmap

**Last Updated**: 2026-01-29

This document serves as the central registry for scientific research topics, technical resources, and implementation ideas for the Golf Modeling Suite. It focuses on rigorous, scientifically grounded concepts in biomechanics, physics, and engineering.

## 1. Biomechanics & Human Movement

### Kinematic Sequence Analysis
- **Proximal Braking Efficiency**: Quantify the deceleration rates of proximal segments (pelvis, thorax) during the downswing. Research suggests that efficient energy transfer requires rapid deceleration of heavy segments to accelerate distal ones (whip effect).
  - *Data Needed*: Angular velocity time-series for pelvis, thorax, arm, and club.
  - *Outcome*: A "Braking Efficiency" metric ($rad/s^2$) to identify energy leaks.
  - *Ref*: Nesbit, S. M. (2005). "A three dimensional kinematic and kinetic study of the golf swing."

- **X-Factor Stretch**: Measure the increase in pelvis-thorax separation angle during the transition phase (start of downswing). This "stretch-shortening cycle" is correlated with increased clubhead speed.
  - *Data Needed*: 3D orientation of pelvis and thorax during transition.
  - *Outcome*: Quantification of elastic energy potential.
  - *Ref*: Cheetham, P. J., et al. (2001). "The importance of stretching the 'X-Factor' in the downswing of golf."

- **Ground Reaction Force (GRF) Efficiency**: Calculate the ratio of peak Vertical GRF to Clubhead Speed. This measures how effectively a golfer uses the ground to generate power.
  - *Data Needed*: Force plate data (Vertical, A-P, M-L forces) and clubhead speed.
  - *Outcome*: Efficiency ratio to guide strength vs. technique training.

## 2. Ball Flight Physics

### Aerodynamics
- **Spin Decay Modeling**: Implement exponential decay of spin rate during flight. The current model uses constant spin, but real golf balls lose spin due to air friction, affecting lift and drag coefficients over the trajectory.
  - *Data Needed*: Time-dependent spin rate decay functions ($d\omega/dt$).
  - *Outcome*: More accurate carry distance and landing angle predictions, especially for long drives.
  - *Ref*: Smits, A. J., & Smith, D. R. (1994). "Aerodynamics of the Golf Ball."

- **Environmental Gradient Modeling**: Model wind shear (boundary layer) and air density changes with altitude. Ball flight apex can reach 30m+, where wind speed significantly differs from ground level.
  - *Data Needed*: Wind profile power law exponents.
  - *Outcome*: Improved accuracy for high-launching shots.

### Trajectory
- **Bounce and Roll Physics**: Implement a rigid-body collision model for the ball-ground interaction, accounting for turf compliance (COR), friction, and slope.
  - *Data Needed*: Coefficients of restitution and friction for various turf types (fairway, green, rough).
  - *Outcome*: Prediction of total distance (Carry + Roll).

## 3. Equipment Science

### Club Dynamics
- **Shaft Torsional Dynamics**: Model the twisting (torque) of the shaft during the downswing and impact. High-torque shafts can close the face more rapidly but may be less stable.
  - *Data Needed*: Shaft torsional stiffness (GJ) profile.
  - *Outcome*: Analysis of dynamic face closure rates.
  - *Ref*:  MacKenzie, S. J., & Sprigings, E. J. (2009). "A three-dimensional forward dynamics model of the golf swing."

- **Clubhead MOI Tensor**: Replace point-mass clubhead approximations with a full 3D Moment of Inertia tensor. This is critical for accurately predicting the gear effect on off-center hits.
  - *Data Needed*: CAD-derived MOI tensors ($I_{xx}, I_{yy}, I_{zz}, I_{xy}, \dots$) for standard clubheads.
  - *Outcome*: Accurate prediction of ball start line and spin axis tilt for toe/heel strikes.
  - *Ref*: United States Golf Association (USGA) Technical Protocols.

- **Kick Point Optimization**: Analyze shaft EI profiles to determine the dynamic "kick point" and its effect on dynamic loft.
  - *Data Needed*: EI profiles (flexural stiffness) along the shaft.
  - *Outcome*: Algorithm to match shaft profiles to swing speed/tempo for optimal launch conditions.

## 4. Statistical Methods

### Analytics
- **Strokes Gained Baseline**: Develop a "Strokes Gained" implementation.
  - *Data Needed*: Baseline dataset of "shots to hole" from various distances and lies (e.g., Broadie's tables).
  - *Outcome*: Context-aware performance metrics.
  - *Ref*: Broadie, M. (2014). "Every Shot Counts."

- **Impact Location Heatmaps**: Generate 2D density plots of face impact locations.
  - *Data Needed*: Series of $(x, y)$ impact coordinates on the clubface.
  - *Outcome*: Visual tool to assess ball striking consistency.

## 5. Simulation Technology

### Physics Engine
- **Finite Element Impact**: Implement a simplified Finite Element (FE) or discrete element model for the clubface to generate a Coefficient of Restitution (COR) map, rather than a single scalar COR.
  - *Data Needed*: Material properties (Young's Modulus, Poisson's ratio) and face thickness profile.
  - *Outcome*: Accurate smash factor prediction across the entire face (Variable Face Thickness modeling).

## 6. Control Theory

### Robotics
- **Swing Robot Inverse Dynamics**: Calculate the required joint torques to drive a double-pendulum model along a desired kinematic path.
  - *Data Needed*: Target kinematic sequence (angular positions/velocities).
  - *Outcome*: Control inputs for a robotic swing device or biomechanical simulation.

---

## Workflow Log

| Date | Topic Added | Category | Status |
|------|-------------|----------|--------|
| 2026-01-29 | Initial Population of Research Ideas | All | Active |
