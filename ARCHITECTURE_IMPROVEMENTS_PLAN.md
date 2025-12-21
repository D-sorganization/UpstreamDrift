# Architecture & Integration Improvements - Implementation Plan

## Overview

This document outlines the plan to address the findings from the Deep Dive Evaluation, focusing on launcher integration, engine validation, physics parameter visibility, and data handling improvements.

---

## Priority 1: Launcher Contract Unification (CRITICAL)

### Problem
- `launch_golf_suite.py` expects `UnifiedLauncher` with Tkinter-style `mainloop()`
- Actual implementation is `GolfLauncher` built on PyQt
- Runtime failure at entry point
- Not exercised in tests

### Solution

#### 1.1 Create UnifiedLauncher Wrapper
```python
# launchers/unified_launcher.py
class UnifiedLauncher:
    """Unified launcher interface wrapping PyQt GolfLauncher."""
    
    def __init__(self):
        from .golf_launcher import GolfLauncher
        self.launcher = GolfLauncher()
    
    def mainloop(self):
        """Start the launcher (PyQt exec loop)."""
        return self.launcher.exec()
    
    def show_status(self):
        """Show suite status."""
        # Implement status display
        pass
```

#### 1.2 Update launch_golf_suite.py
```python
# launch_golf_suite.py
from launchers.unified_launcher import UnifiedLauncher

def main():
    launcher = UnifiedLauncher()  # Now works!
    if args.status:
        launcher.show_status()
    else:
        launcher.mainloop()
```

#### 1.3 Add Integration Test
```python
# tests/integration/test_launcher_integration.py
def test_launch_golf_suite_imports():
    """Verify launch_golf_suite can import and instantiate launcher."""
    from launch_golf_suite import UnifiedLauncher
    launcher = UnifiedLauncher()
    assert launcher is not None
```

**Estimated Effort:** 4 hours
**Priority:** P0 (Blocks primary user flow)

---

## Priority 2: Engine Readiness Checks

### Problem
- `EngineManager` only toggles flags, doesn't verify engines exist
- No dependency probing (MuJoCo, Drake, Pinocchio binaries)
- No asset/model validation
- Users unaware of missing prerequisites until launch

### Solution

#### 2.1 Engine Probe Interface
```python
# shared/python/engine_probes.py
from dataclasses import dataclass
from enum import Enum

class ProbeStatus(Enum):
    AVAILABLE = "available"
    MISSING_BINARY = "missing_binary"
    MISSING_ASSETS = "missing_assets"
    VERSION_MISMATCH = "version_mismatch"
    NOT_INSTALLED = "not_installed"

@dataclass
class EngineProbeResult:
    engine_type: EngineType
    status: ProbeStatus
    version: str | None
    missing_dependencies: list[str]
    diagnostic_message: str
    
class EngineProbe:
    """Base class for engine readiness probes."""
    
    def probe(self) -> EngineProbeResult:
        """Check if engine is ready to use."""
        raise NotImplementedError
```

#### 2.2 MuJoCo Probe
```python
class MuJoCoProbe(EngineProbe):
    def probe(self) -> EngineProbeResult:
        missing = []
        
        # Check for mujoco package
        try:
            import mujoco
            version = mujoco.__version__
        except ImportError:
            return EngineProbeResult(
                engine_type=EngineType.MUJOCO,
                status=ProbeStatus.NOT_INSTALLED,
                version=None,
                missing_dependencies=["mujoco"],
                diagnostic_message="Install with: pip install mujoco"
            )
        except OSError as e:
            return EngineProbeResult(
                engine_type=EngineType.MUJOCO,
                status=ProbeStatus.MISSING_BINARY,
                version=None,
                missing_dependencies=["MuJoCo DLLs"],
                diagnostic_message=f"DLL error: {e}. Install MuJoCo binaries."
            )
        
        # Check for required assets
        assets_path = MUJOCO_ROOT / "assets"
        if not assets_path.exists():
            missing.append("assets directory")
        
        # Check for model files
        models = list(assets_path.glob("*.xml")) if assets_path.exists() else []
        if not models:
            missing.append("model XML files")
        
        if missing:
            return EngineProbeResult(
                engine_type=EngineType.MUJOCO,
                status=ProbeStatus.MISSING_ASSETS,
                version=version,
                missing_dependencies=missing,
                diagnostic_message=f"Missing: {', '.join(missing)}"
            )
        
        return EngineProbeResult(
            engine_type=EngineType.MUJOCO,
            status=ProbeStatus.AVAILABLE,
            version=version,
            missing_dependencies=[],
            diagnostic_message=f"MuJoCo {version} ready"
        )
```

#### 2.3 Drake Probe
```python
class DrakeProbe(EngineProbe):
    def probe(self) -> EngineProbeResult:
        try:
            import pydrake
            version = pydrake.__version__
        except ImportError:
            return EngineProbeResult(
                engine_type=EngineType.DRAKE,
                status=ProbeStatus.NOT_INSTALLED,
                version=None,
                missing_dependencies=["drake"],
                diagnostic_message="Install with: pip install drake"
            )
        
        # Check meshcat port availability
        import socket
        meshcat_available = False
        for port in range(7000, 7011):
            try:
                sock = socket.socket()
                sock.bind(('localhost', port))
                sock.close()
                meshcat_available = True
                break
            except OSError:
                continue
        
        if not meshcat_available:
            return EngineProbeResult(
                engine_type=EngineType.DRAKE,
                status=ProbeStatus.MISSING_BINARY,
                version=version,
                missing_dependencies=["meshcat ports 7000-7010"],
                diagnostic_message="Meshcat ports blocked. Close other instances."
            )
        
        return EngineProbeResult(
            engine_type=EngineType.DRAKE,
            status=ProbeStatus.AVAILABLE,
            version=version,
            missing_dependencies=[],
            diagnostic_message=f"Drake {version} ready, meshcat available"
        )
```

#### 2.4 Update EngineManager
```python
# shared/python/engine_manager.py
class EngineManager:
    def __init__(self):
        self.probes = {
            EngineType.MUJOCO: MuJoCoProbe(),
            EngineType.DRAKE: DrakeProbe(),
            EngineType.PINOCCHIO: PinocchioProbe(),
        }
        self.probe_results = {}
    
    def probe_all_engines(self) -> dict[EngineType, EngineProbeResult]:
        """Probe all engines for readiness."""
        for engine_type, probe in self.probes.items():
            self.probe_results[engine_type] = probe.probe()
        return self.probe_results
    
    def get_available_engines(self) -> list[EngineType]:
        """Get list of available engines."""
        if not self.probe_results:
            self.probe_all_engines()
        
        return [
            engine_type
            for engine_type, result in self.probe_results.items()
            if result.status == ProbeStatus.AVAILABLE
        ]
    
    def get_diagnostic_report(self) -> str:
        """Get human-readable diagnostic report."""
        if not self.probe_results:
            self.probe_all_engines()
        
        lines = ["Engine Readiness Report", "=" * 50, ""]
        for engine_type, result in self.probe_results.items():
            status_icon = "✅" if result.status == ProbeStatus.AVAILABLE else "❌"
            lines.append(f"{status_icon} {engine_type.value.upper()}")
            lines.append(f"   Status: {result.status.value}")
            if result.version:
                lines.append(f"   Version: {result.version}")
            if result.missing_dependencies:
                lines.append(f"   Missing: {', '.join(result.missing_dependencies)}")
            lines.append(f"   {result.diagnostic_message}")
            lines.append("")
        
        return "\n".join(lines)
```

**Estimated Effort:** 8 hours
**Priority:** P0 (Critical for user experience)

---

## Priority 3: Physics Parameter Registry

### Problem
- Well-documented constants in pendulum model not accessible elsewhere
- No consolidated parameter registry
- Users can't inspect or override physical properties
- No unit validation

### Solution

#### 3.1 Create Parameter Registry
```python
# shared/python/physics_parameters.py
from dataclasses import dataclass
from enum import Enum
from typing import Any

class ParameterCategory(Enum):
    BALL = "ball"
    CLUB = "club"
    ENVIRONMENT = "environment"
    BIOMECHANICS = "biomechanics"

@dataclass
class PhysicsParameter:
    name: str
    value: Any
    unit: str
    category: ParameterCategory
    description: str
    source: str
    min_value: float | None = None
    max_value: float | None = None
    
    def validate(self, new_value: Any) -> bool:
        """Validate a new value against constraints."""
        if self.min_value is not None and new_value < self.min_value:
            return False
        if self.max_value is not None and new_value > self.max_value:
            return False
        return True

class PhysicsParameterRegistry:
    """Central registry for all physics parameters."""
    
    def __init__(self):
        self.parameters: dict[str, PhysicsParameter] = {}
        self._load_default_parameters()
    
    def _load_default_parameters(self):
        """Load default parameters from pendulum constants."""
        # Ball parameters
        self.register(PhysicsParameter(
            name="BALL_MASS",
            value=0.04593,  # kg
            unit="kg",
            category=ParameterCategory.BALL,
            description="Golf ball mass",
            source="USGA Rules of Golf",
            min_value=0.04593,
            max_value=0.04593  # Exact per rules
        ))
        
        self.register(PhysicsParameter(
            name="BALL_DIAMETER",
            value=0.04267,  # m
            unit="m",
            category=ParameterCategory.BALL,
            description="Golf ball diameter",
            source="USGA Rules of Golf",
            min_value=0.04267,
            max_value=float('inf')  # Minimum per rules
        ))
        
        # Club parameters
        self.register(PhysicsParameter(
            name="CLUB_MASS",
            value=0.310,  # kg
            unit="kg",
            category=ParameterCategory.CLUB,
            description="Driver club mass",
            source="Typical driver specifications",
            min_value=0.200,
            max_value=0.500
        ))
        
        # Environment
        self.register(PhysicsParameter(
            name="GRAVITY",
            value=9.80665,  # m/s²
            unit="m/s²",
            category=ParameterCategory.ENVIRONMENT,
            description="Standard gravity",
            source="NIST",
            min_value=9.80665,
            max_value=9.80665  # Standard value
        ))
    
    def register(self, param: PhysicsParameter):
        """Register a parameter."""
        self.parameters[param.name] = param
    
    def get(self, name: str) -> PhysicsParameter:
        """Get a parameter by name."""
        return self.parameters[name]
    
    def set(self, name: str, value: Any) -> bool:
        """Set a parameter value with validation."""
        param = self.parameters[name]
        if param.validate(value):
            param.value = value
            return True
        return False
    
    def get_by_category(self, category: ParameterCategory) -> list[PhysicsParameter]:
        """Get all parameters in a category."""
        return [
            param for param in self.parameters.values()
            if param.category == category
        ]
    
    def export_to_dict(self) -> dict:
        """Export all parameters to dictionary."""
        return {
            name: {
                "value": param.value,
                "unit": param.unit,
                "description": param.description,
                "source": param.source
            }
            for name, param in self.parameters.items()
        }
```

#### 3.2 Expose in Launcher
```python
# Add to GolfLauncher
def show_physics_parameters(self):
    """Show physics parameters dialog."""
    from shared.python.physics_parameters import PhysicsParameterRegistry
    
    registry = PhysicsParameterRegistry()
    
    dialog = QDialog(self)
    dialog.setWindowTitle("Physics Parameters")
    
    # Create table showing all parameters
    table = QTableWidget()
    table.setColumnCount(5)
    table.setHorizontalHeaderLabels(["Name", "Value", "Unit", "Description", "Source"])
    
    params = registry.parameters.values()
    table.setRowCount(len(params))
    
    for row, param in enumerate(params):
        table.setItem(row, 0, QTableWidgetItem(param.name))
        table.setItem(row, 1, QTableWidgetItem(str(param.value)))
        table.setItem(row, 2, QTableWidgetItem(param.unit))
        table.setItem(row, 3, QTableWidgetItem(param.description))
        table.setItem(row, 4, QTableWidgetItem(param.source))
    
    layout = QVBoxLayout()
    layout.addWidget(table)
    dialog.setLayout(layout)
    dialog.exec()
```

**Estimated Effort:** 6 hours
**Priority:** P1 (Improves transparency and usability)

---

## Priority 4: Data Contract Strengthening

### Problem
- No validation for sampling rates, coordinate frames, metadata
- No checksum or schema validation
- Timestamp injection complicates reproducibility
- Corrupted files could propagate silently

### Solution

#### 4.1 Create Data Schemas with Pydantic
```python
# shared/python/data_schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional
import hashlib

class SimulationMetadata(BaseModel):
    """Metadata for simulation results."""
    engine: str
    version: str
    timestamp: str
    duration: float = Field(gt=0)
    timestep: float = Field(gt=0)
    sampling_rate: float = Field(gt=0)
    coordinate_frame: str = Field(default="world")
    units: dict[str, str]
    parameters: dict[str, Any]
    
    @validator('coordinate_frame')
    def validate_frame(cls, v):
        allowed = ['world', 'body', 'local']
        if v not in allowed:
            raise ValueError(f"Frame must be one of {allowed}")
        return v

class SimulationResults(BaseModel):
    """Validated simulation results."""
    metadata: SimulationMetadata
    data: dict[str, list[float]]
    checksum: Optional[str] = None
    
    def compute_checksum(self) -> str:
        """Compute SHA256 checksum of data."""
        import json
        data_str = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def validate_checksum(self) -> bool:
        """Validate stored checksum matches data."""
        if self.checksum is None:
            return True
        return self.checksum == self.compute_checksum()
```

#### 4.2 Update OutputManager
```python
# shared/python/output_manager.py
def save_simulation_results(
    self,
    results: pd.DataFrame | dict[str, Any],
    filename: str,
    format_type: OutputFormat = OutputFormat.CSV,
    engine: str = "mujoco",
    metadata: dict[str, Any] | None = None,
    deterministic_name: bool = False,  # NEW
) -> Path:
    """Save simulation results with schema validation."""
    
    # Validate metadata
    if metadata:
        try:
            validated_metadata = SimulationMetadata(**metadata)
        except ValidationError as e:
            logger.error(f"Invalid metadata: {e}")
            raise
    
    # Deterministic naming option
    if not deterministic_name:
        # Add timestamp (existing behavior)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"
    
    # Compute checksum
    if isinstance(results, pd.DataFrame):
        data_dict = results.to_dict('list')
    else:
        data_dict = results
    
    checksum = hashlib.sha256(
        json.dumps(data_dict, sort_keys=True).encode()
    ).hexdigest()
    
    # Save with checksum in metadata
    output_data = {
        "metadata": metadata or {},
        "results": results,
        "checksum": checksum,
        "timestamp": datetime.now().isoformat(),
    }
    
    # ... rest of save logic
```

**Estimated Effort:** 6 hours
**Priority:** P2 (Improves reliability and reproducibility)

---

## Priority 5: Integration Tests

### Problem
- Tests use heavy mocking
- No end-to-end validation
- `validate_suite.py` not tested
- Physical parameters and outputs not verified in real environments

### Solution

#### 5.1 End-to-End Test Suite
```python
# tests/integration/test_end_to_end.py
def test_validate_suite_without_mocks():
    """Run validate_suite.py without mocks."""
    result = subprocess.run(
        [sys.executable, "validate_suite.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "ready for use" in result.stdout or "issues found" in result.stdout

def test_launch_golf_suite_status():
    """Test launch_golf_suite.py --status."""
    result = subprocess.run(
        [sys.executable, "launch_golf_suite.py", "--status"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Engine" in result.stdout

def test_engine_loads_resources():
    """Verify at least one engine can load resources."""
    from shared.python.engine_manager import EngineManager
    
    manager = EngineManager()
    results = manager.probe_all_engines()
    
    # At least one engine should be available
    available = [r for r in results.values() if r.status == ProbeStatus.AVAILABLE]
    assert len(available) > 0, "No engines available"

def test_physics_parameters_accessible():
    """Verify physics parameters are accessible."""
    from shared.python.physics_parameters import PhysicsParameterRegistry
    
    registry = PhysicsParameterRegistry()
    
    # Should have ball parameters
    ball_mass = registry.get("BALL_MASS")
    assert ball_mass.value == 0.04593
    assert ball_mass.unit == "kg"
    
    # Should have gravity
    gravity = registry.get("GRAVITY")
    assert gravity.value == 9.80665

def test_output_manager_real_save_load():
    """Test OutputManager with real file I/O."""
    from shared.python.output_manager import OutputManager, OutputFormat
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = OutputManager(tmpdir)
        
        # Save data
        data = {"time": [0, 1, 2], "angle": [0.0, 0.5, 1.0]}
        path = manager.save_simulation_results(
            data,
            "test_sim",
            format_type=OutputFormat.JSON,
            deterministic_name=True  # No timestamp for reproducibility
        )
        
        # Load data
        loaded = manager.load_simulation_results(
            "test_sim",
            format_type=OutputFormat.JSON
        )
        
        # Verify checksum
        assert loaded["checksum"] is not None
        # Verify data matches
        assert loaded["results"] == data
```

**Estimated Effort:** 4 hours
**Priority:** P1 (Ensures real-world functionality)

---

## Implementation Timeline

### Week 1: Critical Fixes
- [ ] Day 1-2: Launcher contract unification (P0)
- [ ] Day 3-4: Engine readiness probes (P0)
- [ ] Day 5: Integration tests for launcher (P1)

### Week 2: Parameter & Data Improvements
- [ ] Day 1-2: Physics parameter registry (P1)
- [ ] Day 3-4: Data schema validation (P2)
- [ ] Day 5: Integration tests for parameters/data (P1)

### Week 3: Polish & Documentation
- [ ] Update all documentation
- [ ] Add user guides for new features
- [ ] Performance testing
- [ ] Bug fixes

---

## Success Criteria

1. ✅ `launch_golf_suite.py` runs without errors
2. ✅ `validate_suite.py` provides actionable diagnostics
3. ✅ At least one engine probe returns AVAILABLE status
4. ✅ Physics parameters accessible via GUI and API
5. ✅ Data validation prevents corrupted files
6. ✅ All integration tests pass
7. ✅ Test coverage > 60%

---

## Risk Mitigation

- **Breaking changes:** Create feature flags for new validation
- **Performance:** Make probes optional/cached
- **Backward compatibility:** Support both timestamped and deterministic names
- **Testing:** Incremental rollout with comprehensive tests

---

## Next Steps

1. Review and approve this plan
2. Create GitHub issues for each priority
3. Begin implementation starting with P0 items
4. Regular progress reviews
