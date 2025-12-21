# 🚀 Golf Modeling Suite - Implementation Roadmap
**Created:** December 20, 2025 19:10  
**Status:** Ready to Execute  
**Assessment Version:** 1.1 (Updated)

---

## ✅ COMPLETED (Just Now)

- [x] **Removed junk files** (`=*`)
- [x] **Updated .gitignore** (prevent recurrence)

---

## 🎯 PHASE 1: CRITICAL FIXES (This Week - 1 Week Total)

### Day 1: Documentation & Quick Wins (Today - 2 hours)

#### ✅ Task 1.1: Update Documentation Accuracy (30 min)
**Status:** REQUIRED - Fix overpromising

**Files to Update:**
1. `MIGRATION_STATUS.md`
2. `README.md`  
3. `validate_suite.py` output messages

**Changes:**
- Remove "100% COMPLETE" → "95% Complete (Engine loading pending)"
- Add beta status warnings
- Clarify what works vs. what's placeholder

#### ✅ Task 1.2: Fix get_version TODO (30 min)
**File:** `launchers/unified_launcher.py:103`

**Implementation:** See `IMMEDIATE_ACTION_PLAN.md` Task #2

#### ✅ Task 1.3: Add Dependency Bounds (30 min)
**File:** `pyproject.toml`

**Implementation:** See `IMMEDIATE_ACTION_PLAN.md` Task #5

#### ✅ Task 1.4: Initial Git Commit (30 min)
```bash
git add -A
git commit -m "fix: Remove pip artifacts, update .gitignore, fix documentation accuracy

- Remove =*.* files from malformed pip installs
- Add =* to .gitignore to prevent recurrence
- Update MIGRATION_STATUS.md to reflect actual completion (95%)
- Add beta status warnings to README
- Fix dependency bounds in pyproject.toml
- Fix get_version TODO in unified_launcher.py

Refs: CRITICAL_ASSESSMENT_UPDATE.md"
git push
```

---

### Day 2-4: Engine Loading Implementation (2-3 days)

#### 🔴 Task 2.1: Implement EngineManager Loading (CRITICAL++)
**Priority:** P0++ (Highest priority)  
**Effort:** 2-3 days  
**Blocker #7**

**Current State:**
```python
def _load_mujoco_engine(self):
    logger.debug("MuJoCo engine loading placeholder")
    # NO IMPLEMENTATION
```

**Required Implementation:**

**Step 1: MuJoCo Loading (Day 2 - 6 hours)**
```python
def _load_mujoco_engine(self) -> None:
    """Load MuJoCo engine with full initialization."""
    try:
        # 1. Run probe to validate dependencies
        from .engine_probes import MuJoCoProbe
        probe = MuJoCoProbe(self.suite_root)
        result = probe.probe()
        
        if not result.is_available():
            raise GolfModelingError(
                f"MuJoCo not ready:\n{result.diagnostic_message}\n"
                f"Fix: {result.get_fix_instructions()}"
            )
        
        # 2. Import MuJoCo (will fail if not installed)
        import mujoco
        logger.info(f"MuJoCo version {mujoco.__version__} imported successfully")
        
        # 3. Verify model files exist
        model_dir = self.engine_paths[EngineType.MUJOCO] / "assets"
        if not model_dir.exists():
            raise GolfModelingError(
                f"MuJoCo model directory not found: {model_dir}\n"
                f"Expected: {self.engine_paths[EngineType.MUJOCO]}/assets/"
            )
        
        # 4. Find and validate at least one model file
        model_files = list(model_dir.glob("*.xml"))
        if not model_files:
            raise GolfModelingError(
                f"No MuJoCo model files (.xml) found in {model_dir}"
            )
        
        logger.info(f"Found {len(model_files)} MuJoCo models in {model_dir}")
        
        # 5. Test load a model to verify MuJoCo works
        test_model = model_files[0]
        try:
            _ = mujoco.MjModel.from_xml_path(str(test_model))
            logger.info(f"Successfully validated MuJoCo with test model: {test_model.name}")
        except Exception as e:
            raise GolfModelingError(
                f"MuJoCo model validation failed for {test_model.name}: {e}"
            ) from e
        
        # 6. Store loaded state
        self._mujoco_module = mujoco
        self._mujoco_model_dir = model_dir
        
        logger.info("MuJoCo engine fully loaded and validated")
        
    except ImportError as e:
        raise GolfModelingError(
            "MuJoCo not installed. Install with: pip install mujoco>=3.2.3"
        ) from e
```

**Step 2: Drake Loading (Day 3 morning - 3 hours)**
```python
def _load_drake_engine(self) -> None:
    """Load Drake engine with full initialization."""
    try:
        # 1. Run probe
        from .engine_probes import DrakeProbe
        probe = DrakeProbe(self.suite_root)
        result = probe.probe()
        
        if not result.is_available():
            raise GolfModelingError(
                f"Drake not ready:\n{result.diagnostic_message}\n"
                f"Fix: {result.get_fix_instructions()}"
            )
        
        # 2. Import Drake
        import pydrake
        logger.info(f"Drake version {pydrake.__version__} imported successfully")
        
        # 3. Verify Drake can create systems
        from pydrake.systems.framework import DiagramBuilder
        builder = DiagramBuilder()
        diagram = builder.Build()
        logger.info("Drake system creation validated")
        
        # 4. Check Meshcat availability (for visualization)
        try:
            from pydrake.geometry import Meshcat
            meshcat = Meshcat()
            meshcat_url = meshcat.web_url()
            logger.info(f"Drake Meshcat available at: {meshcat_url}")
            self._drake_meshcat = meshcat
        except Exception as e:
            logger.warning(f"Drake Meshcat unavailable: {e}")
            self._drake_meshcat = None
        
        # 5. Store loaded state
        self._drake_module = pydrake
        
        logger.info("Drake engine fully loaded and validated")
        
    except ImportError as e:
        raise GolfModelingError(
            "Drake not installed. Install with: pip install drake>=1.22.0"
        ) from e
```

**Step 3: Pinocchio Loading (Day 3 afternoon - 3 hours)**
```python
def _load_pinocchio_engine(self) -> None:
    """Load Pinocchio engine with full initialization."""
    try:
        # 1. Run probe
        from .engine_probes import PinocchioProbe
        probe = PinocchioProbe(self.suite_root)
        result = probe.probe()
        
        if not result.is_available():
            raise GolfModelingError(
                f"Pinocchio not ready:\n{result.diagnostic_message}\n"
                f"Fix: {result.get_fix_instructions()}"
            )
        
        # 2. Import Pinocchio
        import pinocchio
        logger.info(f"Pinocchio version {pinocchio.__version__} imported")
        
        # 3. Verify model directory
        model_dir = self.engine_paths[EngineType.PINOCCHIO] / "models"
        if model_dir.exists():
            logger.info(f"Pinocchio models available in {model_dir}")
        else:
            logger.warning(f"Pinocchio model directory not found: {model_dir}")
        
        # 4. Test basic Pinocchio functionality
        test_model = pinocchio.buildSampleModelHumanoid()
        logger.info("Pinocchio humanoid model creation validated")
        
        # 5. Store loaded state
        self._pinocchio_module = pinocchio
        
        logger.info("Pinocchio engine fully loaded and validated")
        
    except ImportError as e:
        raise GolfModelingError(
            "Pinocchio not installed. Install with: pip install pin>=2.6.0"
        ) from e
```

**Step 4: MATLAB & Pendulum Loading (Day 4 - 4 hours)**
```python
def _load_matlab_engine(self, engine_type: EngineType) -> None:
    """Load MATLAB engine with validation."""
    try:
        # 1. Check MATLAB installation
        import matlab.engine
        
        # 2. Start MATLAB Engine
        logger.info("Starting MATLAB Engine (this may take 30-60 seconds)...")
        engine = matlab.engine.start_matlab()
        
        # 3. Verify model directory
        model_dir = self.engine_paths[engine_type] / "matlab"
        if not model_dir.exists():
            raise GolfModelingError(f"MATLAB model directory not found: {model_dir}")
        
        # 4. Add to MATLAB path
        engine.addpath(str(model_dir), nargout=0)
        
        # 5. Store engine
        self._matlab_engine = engine
        self._matlab_model_dir = model_dir
        
        logger.info(f"MATLAB engine loaded for {engine_type.value}")
        
    except ImportError as e:
        raise GolfModelingError(
            "MATLAB Engine for Python not installed.\n"
            "Install from: matlabroot/extern/engines/python\n"
            "Run: python setup.py install"
        ) from e

def _load_pendulum_engine(self) -> None:
    """Load pendulum models with validation."""
    model_dir = self.engine_paths[EngineType.PENDULUM]
    
    if not model_dir.exists():
        raise GolfModelingError(f"Pendulum models not found: {model_dir}")
    
    # Verify Python pendulum implementations exist
    python_dir = model_dir / "python"
    if not python_dir.exists():
        raise GolfModelingError(f"Pendulum Python directory not found: {python_dir}")
    
    logger.info(f"Pendulum models available in {model_dir}")
    self._pendulum_model_dir = model_dir
```

**Step 5: Add Attributes to __init__ (30 min)**
```python
def __init__(self, suite_root: Path | None = None):
    # ... existing code ...
    
    # Add storage for loaded engines
    self._mujoco_module: Any = None
    self._mujoco_model_dir: Path | None = None
    self._drake_module: Any = None
    self._drake_meshcat: Any = None
    self._pinocchio_module: Any = None
    self._matlab_engine: Any = None
    self._matlab_model_dir: Path | None = None
    self._pendulum_model_dir: Path | None = None
```

**Step 6: Add Cleanup Method (30 min)**
```python
def cleanup(self) -> None:
    """Clean up loaded engines."""
    if self._matlab_engine is not None:
        try:
            self._matlab_engine.quit()
            logger.info("MATLAB engine shut down")
        except Exception as e:
            logger.warning(f"Error shutting down MATLAB: {e}")
        self._matlab_engine = None
    
    if self._drake_meshcat is not None:
        try:
            # Drake Meshcat cleanup if needed
            pass
        except Exception as e:
            logger.warning(f"Error cleaning up Drake Meshcat: {e}")
        self._drake_meshcat = None
    
    logger.info("Engine cleanup complete")
```

---

#### 🔴 Task 2.2: Add Subprocess Validation (Day 4 - 4 hours)
**Priority:** P0  
**Blocker #8**

**Files to Update:**
- `launch_golf_suite.py`: All `launch_*()` functions

**Pattern to Apply:**
```python
def launch_mujoco():
    """Launch MuJoCo engine with validation."""
    from shared.python.engine_manager import EngineManager
    from pathlib import Path
    
    suite_root = Path(__file__).parent
    manager = EngineManager(suite_root)
    
    # Validate engine is ready
    probe_result = manager.get_probe_result(EngineType.MUJOCO)
    if not probe_result.is_available():
        logger.error(f"MuJoCo not ready:\n{probe_result.diagnostic_message}")
        logger.info(f"Fix: {probe_result.get_fix_instructions()}")
        return False
    
    # Find script
    mujoco_script = (
        suite_root / "engines" / "physics_engines" / "mujoco" 
        / "python" / "mujoco_humanoid_golf" / "advanced_gui.py"
    )
    
    if not mujoco_script.exists():
        raise GolfModelingError(f"MuJoCo script not found: {mujoco_script}")
    
    # Verify working directory
    work_dir = mujoco_script.parent.parent
    if not work_dir.exists():
        raise GolfModelingError(f"Working directory not found: {work_dir}")
    
    logger.info(f"Launching MuJoCo from {work_dir}")
    subprocess.run([sys.executable, str(mujoco_script)], cwd=str(work_dir))
    return True
```

---

### Day 5: Security & Cleanup (Day 5 - 6 hours)

#### 🔴 Task 3.1: Security Audit - Eval/Exec (4 hours)
**Blocker #3**

See `IMMEDIATE_ACTION_PLAN.md` Task #3 for full details.

**Files to Audit:**
- `launchers/golf_launcher.py`
- `launchers/unified_launcher.py`
- `launchers/golf_suite_launcher.py`

#### ✅ Task 3.2: Fix Docker Root User (1 hour)
**Blocker (medium priority)**

See `IMMEDIATE_ACTION_PLAN.md` Task #4

#### ✅ Task 3.3: Final Phase 1 Commit (1 hour)
```bash
# Run quality checks
black .
ruff check .
mypy .
pytest

# Commit
git add -A
git commit -m "feat: Implement engine loading and subprocess validation

BREAKING CHANGES:
- EngineManager now fully initializes engines (no longer placeholder)
- Engine loading validates dependencies via probes
- Subprocess launches validate engines before execution
- Security: Removed eval/exec usage in launchers

Features:
- Full MuJoCo engine loading with model validation
- Full Drake engine loading with Meshcat support
- Full Pinocchio engine loading with model validation
- MATLAB Engine API integration
- Pendulum model validation
- Subprocess validation for all engine launches
- Cleanup method for proper resource management

Fixes:
- Engine switching now actually initializes engines
- Early dependency failure detection
- Actionable error messages via probes
- Docker security (non-root user)

Testing:
- Added engine loading integration tests
- Added subprocess validation tests
- Coverage increased to 45%

Refs: CRITICAL_ASSESSMENT_UPDATE.md, Blockers #3, #7, #8"
git push
```

---

## 🟡 PHASE 2: VALIDATION & DOCS (Weeks 2-4)

### Week 2: Testing (5 days)

#### Task 4.1: Engine Loading Tests (2 days)
```python
# tests/integration/test_engine_loading.py

def test_mujoco_loading_with_dependencies():
    """Test MuJoCo loads when dependencies present."""
    manager = EngineManager()
    
    # Should succeed if MuJoCo installed
    result = manager.switch_engine(EngineType.MUJOCO)
    if result:
        assert manager.get_current_engine() == EngineType.MUJOCO
        assert manager._mujoco_module is not None

def test_engine_loading_without_dependencies():
    """Test graceful failure when dependencies missing."""
    # Mock missing dependency
    with patch('builtins.__import__', side_effect=ImportError):
        manager = EngineManager()
        result = manager.switch_engine(EngineType.MUJOCO)
        assert result is False

def test_engine_cleanup():
    """Test engine cleanup releases resources."""
    manager = EngineManager()
    manager.switch_engine(EngineType.MUJOCO)
    manager.cleanup()
    assert manager._mujoco_module is None
```

#### Task 4.2: Physics Validation Suite (3 days)
```python
# tests/validation/test_physics_accuracy.py

def test_energy_conservation_ballistic():
    """Verify energy conserved in ballistic trajectory."""
    # Use MuJoCo to simulate ball drop
    # Calculate initial energy: E0 = mgh
    # Calculate final energy: Ef = 0.5*m*v^2
    # Assert |E0 - Ef| < 1% (accounting for numerical error)
    pass

def test_cross_engine_consistency():
    """Compare same simulation across engines."""
    # Run simple pendulum in MuJoCo, Drake, Pinocchio
    # Compare trajectories
    # Assert differences < 5% (accounting for solver differences)
    pass
```

### Week 3: Documentation (5 days)

#### Task 5.1: API Documentation (3 days)
```bash
# Configure Sphinx
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

cd docs
sphinx-quickstart --no-sep --project="Golf Modeling Suite" \
  --author="Golf Modeling Suite Team" --release="1.0.0" \
  --language=en --ext-autodoc --ext-viewcode

# Generate API docs
sphinx-apidoc -f -o source/ ../shared/python/
sphinx-apidoc -f -o source/ ../launchers/

# Build
make html

# Add to CI
```

#### Task 5.2: Tutorial Notebooks (2 days)
```python
# examples/01_basic_simulation.ipynb

"""
# Tutorial 1: Basic Golf Swing Simulation

This tutorial shows how to:
1. Initialize the Golf Modeling Suite
2. Load a physics engine
3. Run a simple simulation
4. Visualize results
"""

from shared.python.engine_manager import EngineManager, EngineType

# Initialize manager
manager = EngineManager()

# Check available engines
print(manager.get_diagnostic_report())

# Load MuJoCo
manager.switch_engine(EngineType.MUJOCO)

# Run simulation (implementation details...)
```

### Week 4: Integration & Testing (5 days)

#### Task 6.1: Increase Coverage to 50%+ (3 days)
- Add launcher integration tests
- Add output manager tests
- Add parameter validation tests

#### Task 6.2: Final Quality Pass (2 days)
- Run all checks
- Fix remaining issues
- Documentation review

---

## 🟢 PHASE 3: COMMUNITY & LAUNCH (Months 2-3)

### Month 2: Validation Study
- Design experiment comparing engines
- Run benchmarks
- Write paper/preprint
- Submit to arXiv

### Month 3: Community Building
- Create forum/Discord
- Announce on mailing lists
- Present at conferences
- v1.0 Release Candidate

---

## 📊 Success Metrics

### End of Phase 1 (Week 1)
- [ ] All 8 blockers resolved
- [ ] Engine loading functional
- [ ] Test coverage ≥ 45%
- [ ] Security clean
- [ ] Documentation accurate

### End of Phase 2 (Week 4)
- [ ] Test coverage ≥ 55%
- [ ] API docs generated
- [ ] 5 tutorial notebooks
- [ ] Physics validation suite
- [ ] v1.0 RC ready

### End of Phase 3 (Month 3)
- [ ] Validation study published
- [ ] Community of 50+ users
- [ ] v1.0 Stable released
- [ ] Production rating ≥ 9/10

---

## 🚦 Current Status

**Phase:** 2 (Validation & Docs)  
**Day:** 1  
**Progress:** 40% (Docs Structure Created, Guides Started)

**Next Steps:**
1. Update documentation accuracy (30 min)
2. Fix get_version TODO (30 min)
3. Begin engine loading implementation (2-3 days)

---

**Roadmap Version:** 1.0  
**Last Updated:** December 20, 2025 19:10  
**Status:** Ready to Execute
