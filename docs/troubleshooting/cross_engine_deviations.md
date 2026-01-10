# Cross-Engine Deviation Troubleshooting Guide

This guide helps diagnose and resolve discrepancies between physics engines (MuJoCo, Drake, Pinocchio, etc.) when cross-validation fails.

---

## Understanding Severity Levels

### ✅ PASSED (≤1× tolerance)

No action needed. Results are consistent within scientific tolerances.

### ⚠️ WARNING (1-2× tolerance)

- **Status:** Acceptable with caution
- **Action:** Monitor for trends; document if publishing
- **Common cause:** Integration method differences

### ❌ ERROR (2-10× tolerance)

- **Status:** Investigation required
- **Action:** Do not publish without investigation
- **Likely cause:** Configuration mismatch or model differences

### 🚫 BLOCKER (>100× tolerance)

- **Status:** Fundamental model error
- **Action:** Stop and fix before any analysis
- **Likely cause:** Wrong model, sign error, or major bug

---

## Common Deviation Causes

### 1. Integration Method Differences

**Symptom:** Positions diverge over long simulations (>1s)

**Root Cause:**
| Engine | Integrator | Order | Stability |
|--------|-----------|-------|-----------|
| MuJoCo | Semi-implicit Euler | 1st | Unconditional |
| Drake | Runge-Kutta 3 | 3rd | Conditional |
| Pinocchio | Euler | 1st | Conditional |

**Solution:**

1. Use smaller timestep: `dt < 0.0005s` for all engines
2. Compare at shorter time horizons (0.1s instead of 10s)
3. Accept deviations < 2× tolerance as "integration noise"

**When to Worry:**

- Deviation grows exponentially with time → instability
- Deviation > 10× tolerance → physics mismatch

---

### 2. Timestep Size Mismatch

**Symptom:** Results differ despite identical models

**Root Cause:** Engines using different `dt` values

**Solution:**

```python
# Ensure identical timesteps
UNIFIED_DT = 0.001  # 1ms, safe for all engines

mujoco_engine.set_timestep(UNIFIED_DT)
drake_engine.set_timestep(UNIFIED_DT)
pinocchio_engine.set_timestep(UNIFIED_DT)
```

---

### 3. Mass/Inertia Parameter Differences

**Symptom:** Torques differ by consistent percentage (e.g., 15%)

**Root Cause:** URDF parsing differences or default values

**Verification:**

```python
# Check mass parameters match
mj_mass = mujoco_engine.get_total_mass()
dk_mass = drake_engine.get_total_mass()

assert abs(mj_mass - dk_mass) / mj_mass < 0.01, f"Mass mismatch: {mj_mass} vs {dk_mass}"
```

**Solution:**

- Explicitly set all inertia parameters in model
- Avoid relying on engine defaults

---

### 4. Frame Convention Mismatch

**Symptom:** Jacobians differ; positions rotated/flipped

**Root Cause:** Different coordinate frame conventions

- MuJoCo: World frame, Z-up
- Drake: World frame, Z-up (usually matches)
- Pinocchio: Local frame often returned

**Solution:**

```python
# Transform to common frame
J_world = engine.get_jacobian(body, frame="world")  # Not "local"
```

**Verification Checklist:**

- [ ] Check reference frame documentation for each engine
- [ ] Verify joint ordering is identical
- [ ] Confirm sign conventions for rotations

---

### 5. Constraint Handling Differences

**Symptom:** Contact forces differ significantly

**Root Cause:** Engines solve constraints differently

- MuJoCo: Complementarity solver
- Drake: Convex optimization
- Pinocchio: Analytical (limited contact)

**Solution:**

- For contact-heavy simulations, prefer MuJoCo or Drake
- Cross-validate only non-contact portions with Pinocchio

---

### 6. Gravity Vector Mismatch

**Symptom:** Accelerations differ by constant offset

**Root Cause:** Different default gravity values

**Verification:**

```python
# Ensure identical gravity
GRAVITY = np.array([0, 0, -9.80665])  # NIST standard

mujoco_engine.set_gravity(GRAVITY)
drake_engine.set_gravity(GRAVITY)
pinocchio_engine.set_gravity(GRAVITY)
```

---

## Diagnostic Workflow

### Step 1: Check Basic Configuration

```python
def verify_configuration(engine1, engine2):
    """Verify engines have matching configuration."""
    checks = {
        "timestep": abs(engine1.dt - engine2.dt) < 1e-10,
        "gravity": np.allclose(engine1.gravity, engine2.gravity),
        "n_dof": engine1.nq == engine2.nq,
        "n_joints": engine1.nv == engine2.nv,
    }

    for name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")

    return all(checks.values())
```

### Step 2: Isolate the Discrepancy

Test each component separately:

1. Forward kinematics only (no dynamics)
2. Inverse dynamics at single configuration
3. Forward dynamics with zero velocity
4. Full simulation

### Step 3: Use Triangulation

When two engines disagree, use a third as tiebreaker:

```python
def triangulate(mj_result, dk_result, pin_result, tolerance):
    """Identify which engine is the outlier."""
    if np.allclose(mj_result, dk_result, atol=tolerance):
        if not np.allclose(mj_result, pin_result, atol=tolerance):
            return "Pinocchio outlier"
        return "All agree"
    elif np.allclose(mj_result, pin_result, atol=tolerance):
        return "Drake outlier"
    elif np.allclose(dk_result, pin_result, atol=tolerance):
        return "MuJoCo outlier"
    else:
        return "All three disagree - FUNDAMENTAL ERROR"
```

### Step 4: Check Known Limitations

| Engine    | Limitation                 |
| --------- | -------------------------- |
| Pinocchio | Contact support limited    |
| Drake     | 30× slower than MuJoCo     |
| MuJoCo    | No analytical Hill muscles |

---

## Tolerance Reference

From Guideline P3:

| Metric       | Tolerance | Units         |
| ------------ | --------- | ------------- |
| Position     | ±1e-6     | meters        |
| Velocity     | ±1e-5     | m/s           |
| Acceleration | ±1e-4     | m/s²          |
| Torque       | ±1e-3     | N·m           |
| Jacobian     | ±1e-8     | dimensionless |

---

## Quick Fixes Checklist

- [ ] Reduce timestep to 0.0005s
- [ ] Verify model file is identical
- [ ] Check gravity vector matches
- [ ] Confirm units (radians vs degrees)
- [ ] Use same reference frame for outputs
- [ ] Run at shorter time horizons first
- [ ] Test with simpler model (e.g., pendulum)

---

## When to Accept Deviation

Deviations are **acceptable** when:

- < 2× tolerance AND
- Source understood (e.g., integration method) AND
- Documented in methodology section

Deviations are **NOT acceptable** when:

- > 10× tolerance
- Growing exponentially
- Inconsistent across similar configurations
- Source unknown

---

## Getting Help

1. Run `tests/integration/test_cross_engine_validation.py -v` for diagnostics
2. Check `docs/assessments/FEATURE_ENGINE_MATRIX.md` for capability differences
3. Review engine-specific documentation in `engines/physics_engines/*/README.md`
4. Open an issue with: deviation magnitude, engines involved, model file

---

_Last Updated: January 2026_
