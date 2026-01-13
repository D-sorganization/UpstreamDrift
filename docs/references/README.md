# Golf Physics References

This directory contains academic papers and theses used as primary references
for the golf ball impact and flight physics models in the Golf Modeling Suite.

## Primary Sources

### University of Waterloo / McPhee Research Group

The motion research group at University of Waterloo (Prof. John McPhee) has
produced extensive research on golf club-ball dynamics, including:

1. **Three Dimensional Golf Clubhead Ball Impact Models for Drivers and Irons.pdf**
   - Authors: McPhee et al. (Waterloo)
   - Key content: 3D impulse-momentum impact models, COR characterization,
     spin generation mechanics
   - Used for: Impact model validation

2. **Development and Comparison of 3D Dynamics Models of Golf Clubhead Ball Impacts.pdf**
   - Authors: McPhee et al. (Waterloo)
   - Key content: Comparison of impulse-momentum vs volumetric contact models,
     shaft flexibility effects
   - Used for: Impact model architecture

3. **A Three Dimensional Forward Dynamics Model of the Golf Swing.pdf**
   - Authors: Waterloo
   - Key content: Full swing dynamics including golfer biomechanics
   - Used for: Future swing kinematics integration

4. **Effect of Clubhead Inertial Properties and Driver Face Geometry on Golf Ball Trajectories.pdf**
   - Key content: Gear effect, MOI effects on spin
   - Used for: Clubhead parameter sensitivity

### Ball Flight Aerodynamics

5. **Golf Ball Flight Dynamics.pdf**
   - Key content: Aerodynamic equations, drag/lift coefficient models
   - Source for coefficient equations:
     ```
     Cd = a + b*S + c*S²
     Cl = d + e*S + f*S²
     ```
     where S = (ω \* r) / v is the spin ratio

6. **Three Dimensional Golf Ball Flight.pdf**
   - Key content: 3D trajectory simulation, wind effects
   - Used for: Ball flight model validation

7. **On the Flight of a Golf Ball in the Vertical Plane.pdf**
   - Authors: Penner (Waterloo)
   - Key content: 2D trajectory fundamentals, optimal launch parameters
   - Used for: Physics validation

8. **The Physics of Golf - The Optimum Loft of a Driver.pdf**
   - Authors: A.R. Penner
   - Key content: Optimal driver loft analysis
   - Reference for: Launch parameter optimization

### Modern Approaches

9. **Combining Physics and Deep Learning Models to Simulate the Flight of a Golf Ball.pdf**
   - Key content: Hybrid physics/ML approach for trajectory prediction
   - Reference for: Future ML integration

10. **The Influence of Club Head Kinematics on Early Ball Flight Characteristics in the Golf Drive.pdf**
    - Key content: Club delivery to launch condition relationships
    - Used for: Impact-to-flight correlation

## Coefficient Sources

### Aerodynamic Coefficients (from Waterloo research)

The golf ball drag and lift coefficients are modeled as quadratic functions
of the spin ratio S:

```python
# Spin ratio (dimensionless)
S = (omega * r) / v  # where omega = spin rate [rad/s], r = radius, v = speed

# Drag coefficient (Penner/Waterloo model)
# Typical values: Cd0 ≈ 0.22, Cd1 ≈ 0.10, Cd2 ≈ 0.05
Cd = Cd0 + Cd1 * S + Cd2 * S**2

# Lift coefficient (Penner/Waterloo model)
# Typical values: Cl0 ≈ 0.00, Cl1 ≈ 0.40, Cl2 ≈ 0.10
Cl = Cl0 + Cl1 * S + Cl2 * S**2
```

### Impact Model Coefficients

From McPhee's rigid body impact model:

```python
# Coefficient of Restitution (COR)
COR_driver = 0.78  # USGA limit: 0.830
COR_iron = 0.72

# Contact duration
contact_time = 450e-6  # ~450 microseconds

# Club-ball friction
mu_friction = 0.4  # Grooved clubface
```

## Data Files

The `data/` directory contains supporting data files:

- `C3DExport Tour average.c3d` - C3D motion capture of tour average swing
- `C3DExport tour average iron.c3d` - C3D motion capture of tour average iron
- `Wiffle_ProV1_club_3D_data.xlsx` - Impact test data

## External Resources

- [Waterloo Motion Research Group](https://uwaterloo.ca/motion-research-group)
- [TrackMan University](https://blog.trackmangolf.com)
- [USGA Equipment Regulations](https://www.usga.org)

---

_Last Updated: January 2026_
