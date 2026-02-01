---
title: Environmental Model Ignores Altitude for Air Density Calculation
labels: physics-gap, high
---

# Issue Description
The `EnvironmentalConditions` class has an `altitude` field, but the simulation logic uses a static `air_density` value (defaulting to Sea Level) unless manually overridden.

**File:** `src/shared/python/ball_flight_physics.py`
**Line:** `BallFlightSimulator.simulate_trajectory`

# Expected Physics
Air density decreases with altitude. At 1600m (Denver), air density is ~15% lower than at sea level. This has a massive effect on carry distance (approx +10% distance).
A simulation tool with an `altitude` input is expected to automatically adjust the air density using the International Standard Atmosphere (ISA) model.

# Actual Implementation
```python
@dataclass(frozen=True)
class EnvironmentalConditions:
    air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3)
    altitude: float = 0.0
    # ...
```
The `altitude` field exists but is effectively a "dead" parameter in the `_calculate_accel_core` function, which reads `self.environment.air_density`.

# Impact
- **Accuracy:** Simulations run with `altitude=1000` but default `air_density` will yield sea-level results, confusing the user.

# Recommended Fix
1. Add a method `update_density_from_altitude()` to `EnvironmentalConditions`.
2. Or, in `BallFlightSimulator.__init__`, if `altitude` is non-zero and `air_density` is default, recalculate density.
3. Use the standard formula:
   $$ \rho = \rho_0 \left( 1 - \frac{L \cdot h}{T_0} \right)^{\frac{g M}{R L} - 1} $$
