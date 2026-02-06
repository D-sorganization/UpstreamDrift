# GitHub Issues Tracking

> **Generated**: January 30, 2026
> **Status**: Active tracking document for issue management

## Issues to Create

### Critical Priority (P0)

#### Issue #1: ML-Based Swing Phase Detection

**Title**: `feat(analysis): Implement ML-based swing phase detection`
**Labels**: `enhancement`, `analysis`, `ml`
**Description**:

```
## Summary
The current swing phase detection in `AnalysisService._detect_swing_phase()` uses simple heuristics.
For production use, this should be replaced with ML-based detection.

## Current State
- Location: `src/api/services/analysis_service.py:342-356`
- Returns `None` for most states (only detects "address" at time=0)

## Proposed Solution
1. Train a classifier on labeled swing sequence data
2. Use joint angles and velocities as features
3. Support real-time phase detection during simulation

## Acceptance Criteria
- [ ] Phase detection accuracy > 90%
- [ ] Latency < 10ms per frame
- [ ] Support for all 8 swing phases
```

---

### High Priority (P1)

#### Issue #2: Service Wrapper Implementation in Local Server

**Title**: `feat(api): Implement service wrappers in local_server.py`
**Labels**: `enhancement`, `api`
**Description**:

````
## Summary
The local server has TODO comments for service wrapper implementations.

## Current State
- Location: `src/api/local_server.py:104-105`
- Lines commented out:
  ```python
  # app.state.simulation_service = simulation_service # TODO: Implement service wrapper
  # app.state.analysis_service = analysis_service   # TODO: Implement service wrapper
````

## Proposed Solution

Implement proper service initialization similar to `server.py`.

## Acceptance Criteria

- [ ] Services initialized and stored in app.state
- [ ] Routes can access services via dependency injection

```

#### Issue #3: Kinematic Sequence Peak Detection
**Title**: `feat(analysis): Implement kinematic sequence peak detection`
**Labels**: `enhancement`, `analysis`, `biomechanics`
**Description**:
```

## Summary

Kinematic sequence analysis returns placeholder `None` values for peak velocities.

## Current State

- Location: `src/api/services/analysis_service.py:314-319`
- Returns: `{"pelvis_peak": None, "torso_peak": None, "arm_peak": None, "club_peak": None}`

## Proposed Solution

1. Track segment angular velocities over time
2. Detect peak values for each segment
3. Calculate timing relationships (kinematic sequence)

## Acceptance Criteria

- [ ] Detect peak velocities for pelvis, torso, arm, club
- [ ] Calculate time differences between peaks
- [ ] Validate against known biomechanical data

```

---

### Medium Priority (P2)

#### Issue #4: X-Factor Calculation
**Title**: `feat(analysis): Implement X-factor calculation in swing analysis`
**Labels**: `enhancement`, `analysis`, `biomechanics`
**Description**:
```

## Summary

X-factor (hip-shoulder separation angle) calculation returns `None`.

## Current State

- Location: `src/api/services/analysis_service.py:298`
- Returns: `"x_factor": None`

## Proposed Solution

1. Get pelvis and torso rotation angles from engine
2. Calculate separation angle (X-factor)
3. Track X-factor stretch during transition

## Acceptance Criteria

- [ ] Calculate X-factor at address
- [ ] Calculate X-factor at top of backswing
- [ ] Calculate X-factor stretch during transition

```

#### Issue #5: Ground Reaction Force Analysis
**Title**: `feat(analysis): Enhanced ground reaction force analysis`
**Labels**: `enhancement`, `analysis`, `kinetics`
**Description**:
```

## Summary

Ground reaction forces are captured but not analyzed.

## Current State

- Location: `src/api/services/analysis_service.py:188-192`
- Stores raw contact forces but doesn't compute derived metrics

## Proposed Solution

1. Calculate vertical, horizontal, and lateral forces
2. Compute center of pressure trajectory
3. Analyze weight transfer patterns

## Acceptance Criteria

- [ ] GRF decomposition (Fv, Fh, Fl)
- [ ] COP trajectory calculation
- [ ] Weight transfer percentage computation

```

---

### Low Priority (P3)

#### Issue #6: Energy Flow Analysis
**Title**: `feat(analysis): Implement energy flow analysis`
**Labels**: `enhancement`, `analysis`, `energetics`
**Description**:
```

## Summary

Energy flow analysis returns empty dictionary.

## Current State

- Location: `src/api/services/analysis_service.py:226`
- Returns: `"energy_flow": {}`

## Proposed Solution

Track energy transfer between body segments during swing.

## Acceptance Criteria

- [ ] Calculate segment-wise energy
- [ ] Compute energy transfer rates
- [ ] Identify energy leaks

```

---

## Architecture Verification Issues

#### Issue #7: API Architecture Verification Suite
**Title**: `test(api): Create comprehensive API architecture verification tests`
**Labels**: `testing`, `api`, `architecture`
**Description**:
```

## Summary

Create verification tests for the new API architecture to ensure:

- DRY compliance
- Orthogonality
- Design by Contract adherence
- Traceability

## Tests to Implement

1. **DRY Verification**

   - No duplicate datetime handling
   - Centralized error codes used throughout
   - Single logging configuration

2. **Orthogonality Verification**

   - Routes don't contain business logic
   - Services are independently testable
   - Middleware is composable

3. **Contract Verification**

   - All public APIs have contracts
   - Contracts are enforced in tests
   - Error messages include contract info

4. **Traceability Verification**
   - All responses have request_id
   - Errors include error codes
   - Logs contain correlation IDs

## Acceptance Criteria

- [ ] 100% coverage of verification scenarios
- [ ] Tests run in CI pipeline
- [ ] Documentation updated with test results

```

#### Issue #8: Engine Interface Contract Compliance
**Title**: `test(engines): Verify all engines implement PhysicsEngine protocol`
**Labels**: `testing`, `engines`, `contracts`
**Description**:
```

## Summary

Ensure all physics engine implementations comply with the PhysicsEngine protocol.

## Verification Points

1. All required methods implemented
2. Return types match protocol
3. Contracts on methods are honored

## Engines to Verify

- [ ] MuJoCo
- [ ] Drake
- [ ] Pinocchio
- [ ] OpenSim
- [ ] MyoSuite
- [ ] MATLAB adapters

```

#### Issue #9: Error Code Coverage Audit
**Title**: `audit(api): Ensure all error paths use structured error codes`
**Labels**: `audit`, `api`, `diagnostics`
**Description**:
```

## Summary

Audit all API routes to ensure they use structured error codes (GMS-XXX-YYY).

## Current Coverage

- core.py: Uses iso_format() ✓
- auth.py: Needs audit
- engines.py: Needs audit
- simulation.py: Needs audit
- analysis.py: Uses error codes ✓
- video.py: Needs audit
- export.py: Needs audit

## Acceptance Criteria

- [ ] All HTTPException uses replaced with raise_api_error
- [ ] Error responses include request_id
- [ ] Documentation of all error codes

````

---

## Completed Issues (To Close)

### Recently Completed

| Issue | Title | Status | Commit |
|-------|-------|--------|--------|
| - | Hardcoded timestamp in health check | Fixed | This commit |
| - | Analysis service placeholder implementations | Fixed | This commit |
| - | DRY violations in datetime imports | Fixed | Previous commit |
| - | Request tracing infrastructure | Implemented | Previous commit |
| - | Structured error codes | Implemented | Previous commit |
| - | Dependency injection inconsistency | Fixed | Previous commit |
| - | Documentation reorganization | Completed | Previous commit |

---

## Issue Creation Commands

When `gh` CLI is available, run these commands:

```bash
# P0 - Critical
gh issue create --title "feat(analysis): Implement ML-based swing phase detection" \
  --label "enhancement,analysis,ml" \
  --body-file /tmp/issue1.md

# P1 - High
gh issue create --title "feat(api): Implement service wrappers in local_server.py" \
  --label "enhancement,api"

gh issue create --title "feat(analysis): Implement kinematic sequence peak detection" \
  --label "enhancement,analysis,biomechanics"

# P2 - Medium
gh issue create --title "feat(analysis): Implement X-factor calculation" \
  --label "enhancement,analysis,biomechanics"

gh issue create --title "feat(analysis): Enhanced ground reaction force analysis" \
  --label "enhancement,analysis,kinetics"

# P3 - Low
gh issue create --title "feat(analysis): Implement energy flow analysis" \
  --label "enhancement,analysis,energetics"

# Architecture Verification
gh issue create --title "test(api): Create comprehensive API architecture verification tests" \
  --label "testing,api,architecture"

gh issue create --title "test(engines): Verify all engines implement PhysicsEngine protocol" \
  --label "testing,engines,contracts"

gh issue create --title "audit(api): Ensure all error paths use structured error codes" \
  --label "audit,api,diagnostics"
````

---

## Maintenance Notes

### How to Update This Document

1. When creating an issue, add it to the appropriate priority section
2. When closing an issue, move it to "Completed Issues"
3. Run architecture verification tests periodically
4. Update error code coverage after each API change

### Related Documentation

- [API Architecture](api/API_ARCHITECTURE.md)
- [Development Guide](api/DEVELOPMENT.md)
- [Design by Contract](development/design_by_contract.md)
- [Architecture Review](assessments/API_ARCHITECTURE_REVIEW_2026-01-30.md)
