# Physics Issue: Hardcoded Aerodynamic Coefficients

**ID**: ISSUE_P001
**Category**: Physics Accuracy / Legal Risk
**Status**: Open
**Severity**: High

## Description
The file `src/shared/python/physics/ball_flight_physics.py` contains hardcoded aerodynamic coefficients (`cd0=0.21`, `cd1=0.05`, `cl1=0.38`) within the `BallProperties` class. These values are uncited and may correspond to specific ball models, creating a legal risk if they match competitor confidential data. Furthermore, using fixed coefficients limits the simulation's ability to model different ball types (e.g., tour vs. distance balls).

## Impact
-   **Legal**: Potential infringement if coefficients match proprietary data.
-   **Accuracy**: Inability to simulate modern ball aerodynamics which have complex drag/lift dependencies on Reynolds number.

## Recommended Fix
1.  Deprecate `BallProperties` hardcoded defaults.
2.  Fully migrate to `src/shared/python/physics/aerodynamics.py` which implements Reynolds correction.
3.  Load coefficients from external JSON/YAML configuration files.
4.  Add citations for default values (e.g., Smits & Smith, 1994).
