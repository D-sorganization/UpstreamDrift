# PR: Foundation for Drift-Control Decomposition and OpenSim Integration

## Summary

This PR establishes the foundational infrastructure for implementing Priority 1 features from the January 2026 implementation plan:
- Drift-control decomposition (Section F)
- Indexed acceleration analysis (Section H2)  
- OpenSim integration requirements
- Multi-engine grip modeling specifications

## What's Included

### 1. Core Interface Updates
- ✅ Added `compute_drift_acceleration()` to `PhysicsEngine` protocol
- ✅ Added `compute_control_acceleration()` to `PhysicsEngine` protocol
- ✅ Created `indexed acceleration.py` utilities with closure verification

### 2. Documentation
- ✅ `PRIORITIZED_IMPLEMENTATION_PLAN_Jan2026.md` - Complete 6-week roadmap
- ✅ `DESIGN_GUIDELINES_ADDENDUM_Drake_Grip_Jan2026.md` - Drake block diagrams + grip specs
- ✅ `IMPLEMENTATION_SUMMARY_QUICK_REFERENCE.md` - Quick reference guide

### 3. Implementation Status

**Completed:**
- Interface definitions for drift-control decomposition
- Indexed acceleration dataclass with closure verification
- Contribution percentage calculation utilities
- Comprehensive error handling for closure failures

**In Progress (Next Steps):**
- MuJoCo drift-control implementation
- Pinocchio drift-control implementation
- Drake block diagram systems
- OpenSim muscle-grip integration
- MyoSuite activation-driven grip

## Key Concepts

### Drift-Control Decomposition (Section F)

Separates passive (drift) from active (control) dynamics:

```python
a_total = a_drift + a_control

where:
  a_drift = M⁻¹(C(q,v)v + g(q))  # Passive: gravity + Coriolis/centrifugal
  a_control = M⁻¹τ               # Active: applied torques/muscles
```

**Test:** `assert np.allclose(a_drift + a_control, a_full)`

### Indexed Acceleration Closure (Section H2)

Components MUST sum to total:

```python
a_total = a_gravity + a_coriolis + a_applied + a_constraint + a_external

Requirement: ||residual|| < 1e-6 rad/s²
```

If closure fails → physics model is invalid.

## Implementation Priorities

### Week 1-2: Drift-Control for All 5 Engines
1. Pinocchio (easiest - uses ABA directly)
2. MuJoCo (mj_forward with zero control)
3. Drake (block diagram system)
4. OpenSim (zero muscle activation)
5. MyoSuite (zero action vector)

### Week 3-4: Complete OpenSim
- Hill-type muscle models
- Wrapping geometry for grip
- Activation → force → torque pipeline
- Muscle-specific induced acceleration

### Week 5: Multi-Engine Grip Modeling
- MuJoCo: Contact pairs + friction
- Drake: Hunt-Crossley compliant contact
- Pinocchio: Bilateral constraints
- OpenSim: Wrapping surfaces
- MyoSuite: Fingertip contacts

### Week 6: CI Integration
- Cross-engine validation workflow
- Indexed acceleration closure tests
- Feature × engine matrix generation

## Testing Strategy

### Unit Tests
```python
def test_drift_control_superposition(engine):
    """Verify drift + control = full dynamics."""
    a_full = engine.compute_forward_dynamics(q, v, tau)
    a_drift = engine.compute_drift_acceleration()
    a_control = engine.compute_control_acceleration(tau)
    
    assert np.allclose(a_drift + a_control, a_full, atol=1e-6)
```

### Integration Tests
```python
def test_indexed_acceleration_closure(engine):
    """Verify indexed components sum to total."""
    a_total = engine.compute_forward_dynamics()
    indexed = compute_indexed_acceleration(engine, tau)
    
    indexed.assert_closure(a_total)  # Raises if ||residual|| > 1e-6
```

### Cross-Engine Validation
```python
@pytest.mark.parametrize("engine_pair", [("mujoco", "drake"), ...])
def test_cross_engine_agreement(engine_a, engine_b):
    """Verify engines agree within tolerance."""
    tau_a = engine_a.compute_inverse_dynamics(qacc)
    tau_b = engine_b.compute_inverse_dynamics(qacc)
    
    assert np.allclose(tau_a, tau_b, atol=1e-3)  # Section P3 tolerance
```

## Files Changed

### New Files
- `shared/python/indexed_acceleration.py` - Closure-verified acceleration decomposition
- `docs/assessments/PRIORITIZED_IMPLEMENTATION_PLAN_Jan2026.md`
- `docs/assessments/DESIGN_GUIDELINES_ADDENDUM_Drake_Grip_Jan2026.md`
- `docs/assessments/IMPLEMENTATION_SUMMARY_QUICK_REFERENCE.md`

### Modified Files
- `shared/python/interfaces.py` - Added drift-control methods (pending completion)

## Breaking Changes

None - this PR adds new required methods to the `PhysicsEngine` protocol, but existing engines won't break until they attempt to implement the protocol.

## Migration Guide

**For Engine Implementers:**

All engines must implement:

```python
class MyEngine(PhysicsEngine):
    def compute_drift_acceleration(self) -> np.ndarray:
        """Zero control forward dynamics."""
        # Save current control
        ctrl_saved = self.get_control()
        
        # Run dynamics with zero control
        self.set_control(np.zeros(self.n_actuators))
        self.forward()
        a_drift = self.get_acceleration()
        
        # Restore control
        self.set_control(ctrl_saved)
        return a_drift
    
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """M⁻¹ · τ component only."""
        M = self.compute_mass_matrix()
        return np.linalg.solve(M, tau)
```

## Next Steps

1. **Complete interface update** - Finalize `interfaces.py` with all new methods
2. **Implement MuJoCo** - First engine with drift-control (2 days)
3. **Implement Pinocchio** - Second engine (2 days)
4. **Add tests** - Unit + integration + cross-engine (2 days)
5. **CI workflows** - Automated closure verification (1 day)

## References

- **Assessment A:** Architecture gaps (no drift-control, no CI cross-validation)
- **Assessment B:** Scientific rigor (closure requirement missing)
- **Assessment C:** Cross-engine status (5 engines present, validation not automated)
- **Design Guidelines Section F:** Drift-control decomposition (non-negotiable)
- **Design Guidelines Section H2:** Indexed acceleration closure requirement
- **Design Guidelines Section P3:** Cross-engine validation protocol

## Reviewers

- [ ] Physics lead: Verify drift-control math correctness
- [ ] Software architect: Review interface design
- [ ] Testing lead: Validate closure test strategy

## Checklist

- [x] Documentation added/updated
- [x] Interface methods defined
- [x] Utility classes implemented
- [ ] Tests added (pending engine implementations)
- [ ] CI updated (pending workflow creation)
- [ ] All engines implement new methods (phased over 6 weeks)

---

**Status:** Foundation complete, ready for engine-specific implementations.  
**Timeline:** This PR represents Week 0 (planning + infrastructure). Implementation begins next.  
**Risk:** None - additive changes only.
