# Priority Improvements - January 2026

Based on the comprehensive adversarial review, here are the prioritized improvements needed to bring the Golf Modeling Suite to production readiness.

---

## Tier 1: CRITICAL (Must Fix Before Any Release)

### 1. Implement Engine Loading (Effort: 3-5 days)
**Current State:** Placeholder methods that don't actually load engines
**Impact:** Core functionality is non-existent

**Files to modify:**
- `shared/python/engine_manager.py` - Implement `_load_mujoco_engine()`, `_load_drake_engine()`, `_load_pinocchio_engine()`

**Acceptance Criteria:**
- [ ] MuJoCo engine loads and validates model files
- [ ] Drake engine loads and creates diagram builder
- [ ] Pinocchio engine loads and builds sample models
- [ ] MATLAB engine integration (if available)
- [ ] Cleanup methods properly release resources

### 2. Fix Induced Acceleration Analysis Bugs (Effort: 2-3 days)
**Current State:** MuJoCo uses stale kinematics, Drake ignores control input
**Impact:** Core research feature is unreliable

**Files to modify:**
- `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/rigid_body_dynamics/induced_acceleration.py`
- Drake equivalent module

**Bug Details (from audit):**
```python
# MuJoCo Bug: Missing mj_kinematics() call after modifying qpos
# Drake Bug: acc_t = np.zeros_like(acc_g) - should use tau_app
```

### 3. Implement Basic Ball Flight Physics (Effort: 2 weeks)
**Current State:** Not implemented at all
**Impact:** Cannot predict shot outcomes - fundamental for any golf software

**New module:** `shared/python/ball_physics.py`

**Minimum Requirements:**
- [ ] Projectile motion with gravity
- [ ] Air drag (USGA ball coefficient: 0.25)
- [ ] Launch angle and velocity from club impact
- [ ] Landing position calculation
- [ ] Basic trajectory visualization

---

## Tier 2: HIGH PRIORITY (Required for Beta)

### 4. Increase Test Coverage to 40% (Effort: 4 weeks)
**Current State:** 17% coverage
**Target:** 40% for Beta, 50% for v1.0

**Priority Test Areas:**
1. Error handling paths (0 tests currently)
2. URDF generator (empty test file)
3. Cross-engine consistency
4. Comparative analysis module
5. Protocol compliance tests

### 5. Refactor Broad Exception Handling (Effort: 1 week)
**Current State:** 40+ instances of `except Exception as e:`
**Impact:** Masks bugs, makes debugging difficult

**Files with highest debt:**
- `launch_golf_suite.py` (10+ instances)
- `launchers/golf_launcher.py` (6+ instances)
- `validate_phase1_upgrades.py` (6+ instances)

### 6. Enable Security Scanning (Effort: 1 day)
**Current State:** Bandit installed but not running, pip-audit non-blocking
**Impact:** Security vulnerabilities may ship

**Changes to ci-standard.yml:**
```yaml
- name: Security audit with pip-audit
  run: pip-audit --strict  # Remove || true

- name: Run Bandit security scanner
  run: bandit -r shared/ launchers/ -ll
```

---

## Tier 3: MEDIUM PRIORITY (Production Polish)

### 7. Complete URDF 3D Visualization (Effort: 1 week)
**Current State:** Placeholder showing "Implementation in progress"
**Location:** `tools/urdf_generator/visualization.py`

**Requirements:**
- [ ] Open3D or OpenGL backend integration
- [ ] Real-time URDF preview
- [ ] Link/joint highlighting
- [ ] Camera controls

### 8. Implement Undo/Redo in GUI (Effort: 1 week)
**Current State:** Menu items exist but disabled
**Impact:** Standard expected feature

**Pattern:** QUndoStack with command pattern
```python
class EditCommand(QUndoCommand):
    def __init__(self, description, undo_fn, redo_fn):
        super().__init__(description)
        self._undo_fn = undo_fn
        self._redo_fn = redo_fn
```

### 9. Add Video-Based Pose Estimation (Effort: 4 weeks)
**Current State:** OpenPose GUI stub exists
**Impact:** Major competitive disadvantage without it

**Integration Options:**
1. MediaPipe (free, good accuracy)
2. OpenPose (research standard)
3. MMPose (state-of-the-art)

### 10. Refactor Complex Methods (Effort: 3 days)
**Target methods (>70 lines):**
- `golf_launcher.py:_launch_docker_container()` → Extract Docker logic
- `golf_launcher.py:init_ui()` → Split into component setup methods
- `golf_launcher.py:apply_styles()` → Move to external stylesheet

---

## Tier 4: FUTURE ROADMAP (v2.0+)

### 11. Drift vs Control Decomposition
**Roadmap Item:** Feature documented but not implemented
**Impact:** Key research differentiator

### 12. Musculoskeletal Integration
**Current State:** OpenSim/MyoSim are stubs
**Effort:** 8+ weeks for proper integration

### 13. Cloud Platform
**Current State:** Desktop-only
**Effort:** 12+ weeks for new architecture

### 14. Mobile App
**Current State:** None
**Options:** React Native, Flutter, or web-responsive

---

## Implementation Timeline

| Week | Focus Area | Deliverables |
|------|------------|--------------|
| 1 | Engine loading + IAA fixes | Core functionality working |
| 2-3 | Ball flight physics | Basic trajectory prediction |
| 4-5 | Test coverage expansion | 30% coverage |
| 6 | Security + error handling | Hardened codebase |
| 7-8 | URDF viz + Undo/Redo | GUI completeness |
| 9-12 | Video pose estimation | Competitive feature |

---

## Success Metrics

### Short-term (4 weeks)
- [ ] Engine loading implemented and tested
- [ ] IAA bugs fixed with regression tests
- [ ] Ball flight basic implementation
- [ ] Test coverage at 30%

### Medium-term (8 weeks)
- [ ] Test coverage at 40%
- [ ] Security scanning enabled and passing
- [ ] URDF visualization complete
- [ ] Exception handling refactored

### Long-term (12 weeks)
- [ ] Video pose estimation MVP
- [ ] Test coverage at 50%
- [ ] v1.0 Beta release
- [ ] Community launch

---

**Created:** January 1, 2026
**Review Date:** January 15, 2026
