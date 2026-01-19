# Testing Guide for Golf Modeling Suite

## Philosophy

**Tests should verify behavior, not implementation.**

Our testing strategy focuses on:
- ✅ Testing that code does the right thing
- ✅ Catching real bugs before they reach production
- ✅ Documenting expected behavior
- ❌ NOT gaming coverage metrics
- ❌ NOT testing that mocks work
- ❌ NOT testing implementation details

## Test Quality Over Coverage Metrics

**Coverage percentage is a vanity metric if the tests are poor quality.**

A test suite with:
- 95% coverage testing only mock interactions → **worthless**
- 40% coverage testing actual behavior and edge cases → **valuable**

We measure test quality by:
1. **Bug detection rate**: How many CI failures catch real bugs?
2. **Test clarity**: Can someone unfamiliar understand what's being tested?
3. **Behavior coverage**: Are critical paths and edge cases tested?
4. **Maintenance burden**: Do tests break when refactoring implementation?

## Types of Tests

### Unit Tests (`tests/unit/`)

**Purpose**: Test individual functions/classes in isolation

**What to test:**
- Business logic
- Edge cases and error handling
- State transformations
- Input validation

**How to test:**
```python
# ✅ GOOD - Tests actual behavior
def test_convert_units_degrees_to_radians():
    result = convert_units(180, "deg", "rad")
    assert result == pytest.approx(np.pi)

def test_convert_units_invalid_units():
    with pytest.raises(ValueError, match="not supported"):
        convert_units(1, "kg", "lb")

# ❌ BAD - Tests mock interactions
def test_convert_units():
    with patch('math.pi', return_value=3.14):
        result = convert_units(180, "deg", "rad")
        assert result == 3.14  # Testing the mock, not the code
```

**Mocking guidelines for unit tests:**
- ✅ Mock external services (network, filesystem I/O, databases)
- ✅ Mock expensive operations (rendering, large computations)
- ✅ Mock at system boundaries (physics engines, external libraries)
- ❌ Don't mock the code you're testing
- ❌ Don't mock at module level (`sys.modules`)
- ❌ Don't mock stdlib functions unless necessary

### Integration Tests (`tests/integration/`)

**Purpose**: Test that components work together correctly

**What to test:**
- Component boundaries and contracts
- Data flow between subsystems
- Real dependencies working together
- Configuration loading and wiring

**How to test:**
```python
# ✅ GOOD - Tests real integration
def test_engine_manager_loads_mujoco():
    manager = EngineManager()

    # Both components must work together
    if EngineType.MUJOCO in manager.get_available_engines():
        engine = manager.get_engine(EngineType.MUJOCO)
        engine.load_from_path("tests/assets/simple_arm.urdf")

        # Verify actual behavior
        assert engine.model is not None
        assert engine.model.nq > 0

# ❌ BAD - Fake integration test
def test_mujoco_drake_comparison():
    # Just comparing hardcoded mock data - not testing anything real
    mock_results = {
        EngineType.MUJOCO: {"distance": 250.0},
        EngineType.DRAKE: {"distance": 248.5}
    }
    assert abs(mock_results[EngineType.MUJOCO]["distance"] -
               mock_results[EngineType.DRAKE]["distance"]) < 10
```

**Integration test patterns:**
- Use `@pytest.mark.skipif` for optional dependencies
- Use real filesystem with `tempfile.TemporaryDirectory()`
- Check if dependencies are mocked before running
- Test actual data flow, not mocked interactions

### End-to-End Tests (Future)

**Purpose**: Test complete workflows from user perspective

**Examples:**
- Load model → Run simulation → Generate output → Verify results
- Launch GUI → Load configuration → Run analysis → Export data

## Anti-Patterns to Avoid

### 1. Module-Level Mocking

```python
# ❌ BAD - Mocks entire dependency tree
sys.modules["mujoco"] = MagicMock()
sys.modules["pydrake"] = MagicMock()
sys.modules["pinocchio"] = MagicMock()

from engines.mujoco_engine import MuJoCoEngine

def test_engine():
    engine = MuJoCoEngine()
    engine.step()  # Calls mock.step(), tests nothing
```

**Why it's bad:**
- Tests that Python can import files, not that code works
- No real dependencies are used
- Cannot catch integration issues
- False sense of security

**Fix:**
```python
# ✅ GOOD - Use skipif for optional dependencies
@pytest.mark.skipif(not HAS_MUJOCO, reason="MuJoCo not installed")
def test_engine():
    from engines.mujoco_engine import MuJoCoEngine

    engine = MuJoCoEngine()
    engine.load_from_path("test.xml")

    # Test actual behavior
    initial_state = engine.get_state()
    engine.step()
    new_state = engine.get_state()

    assert not np.array_equal(initial_state, new_state)
```

### 2. Testing Mock Interactions

```python
# ❌ BAD - Only tests that mock was called
def test_process_data():
    with patch('module.process') as mock_process:
        process_data(input_data)
        mock_process.assert_called_once_with(input_data)
```

**Why it's bad:**
- Breaks when refactoring (even if behavior unchanged)
- Doesn't verify correctness
- Tests implementation, not behavior

**Fix:**
```python
# ✅ GOOD - Tests actual behavior
def test_process_data():
    input_data = create_test_data()
    result = process_data(input_data)

    # Verify output is correct
    assert result.shape == (10, 3)
    assert result.min() >= 0
    assert result.max() <= 1
```

### 3. Swallowing Exceptions

```python
# ❌ BAD - Exceptions make test pass
def test_widget_creation():
    try:
        widget = ComplexWidget()
        widget.initialize()
    except Exception as e:
        print(f"Skipped: {e}")  # Test "passes" even if it fails!
```

**Why it's bad:**
- Test passes even when code is broken
- No actual verification
- Hides real issues

**Fix:**
```python
# ✅ GOOD - Test succeeds only if code works
@pytest.mark.skipif(not HAS_QT, reason="Qt not installed")
def test_widget_creation():
    widget = ComplexWidget()
    widget.initialize()

    # Verify widget state
    assert widget.is_initialized
    assert widget.width > 0
```

### 4. Coverage Theater

```python
# ❌ BAD - "Shotgun testing" for coverage
def test_coverage_boost():
    # Just instantiate everything to hit lines
    obj1 = ClassA()
    obj2 = ClassB()
    obj3 = ClassC()
    # No assertions, no behavior verification
```

**Why it's bad:**
- Inflates coverage metrics without testing anything
- Provides false confidence
- Wastes CI resources

**Fix:** Delete these tests. Write fewer, better tests.

## Writing Good Tests

### 1. Use Descriptive Names

```python
# ❌ BAD
def test_1():
    ...

# ✅ GOOD
def test_convert_units_handles_invalid_unit_types():
    ...
```

### 2. Test One Thing

```python
# ❌ BAD - Tests multiple unrelated things
def test_engine():
    engine = Engine()
    assert engine.name == "test"
    assert engine.step() is not None
    assert engine.reset() is True
    assert engine.get_state().shape == (10,)

# ✅ GOOD - Focused tests
def test_engine_has_correct_name():
    engine = Engine()
    assert engine.name == "test"

def test_engine_step_updates_state():
    engine = Engine()
    initial_state = engine.get_state()
    engine.step()
    assert not np.array_equal(engine.get_state(), initial_state)
```

### 3. Use Fixtures for Setup

```python
# ✅ GOOD - Reusable setup
@pytest.fixture
def simple_arm_urdf():
    return Path("tests/assets/simple_arm.urdf")

@pytest.fixture
def mujoco_engine(simple_arm_urdf):
    engine = MuJoCoEngine()
    engine.load_from_path(simple_arm_urdf)
    return engine

def test_engine_has_correct_dofs(mujoco_engine):
    assert mujoco_engine.model.nq == 7
```

### 4. Test Edge Cases

```python
def test_divide():
    # ✅ Test normal case
    assert divide(10, 2) == 5

    # ✅ Test edge cases
    assert divide(0, 5) == 0
    assert divide(-10, 2) == -5

    # ✅ Test error cases
    with pytest.raises(ValueError):
        divide(10, 0)
```

## Test Organization

```
tests/
├── unit/                          # Fast, isolated tests
│   ├── test_engine_manager.py     # Tests EngineManager class
│   ├── test_common_utils.py       # Tests utility functions
│   └── ...
├── integration/                   # Component integration tests
│   ├── test_real_engine_loading.py
│   ├── test_cross_engine_consistency.py
│   └── ...
├── assets/                        # Test fixtures and data
│   ├── simple_arm.urdf
│   └── ...
└── conftest.py                    # Shared fixtures
```

## Running Tests

### Platform-Specific Requirements (Issue #539)

#### Linux

GUI tests require a display or virtual framebuffer. If running on a headless system:

```bash
# Install Xvfb
sudo apt-get install xvfb

# Run tests with virtual display
xvfb-run pytest tests/unit/test_launcher_ux.py

# Or run all tests with display
xvfb-run pytest
```

#### Windows

Some tests may require specific DLLs for physics engines. Ensure the following are installed:
- Visual C++ Redistributable
- MuJoCo dependencies (if testing MuJoCo engine)

#### macOS

GUI tests should work out of the box if running in a graphical session.

### Basic Commands

# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_engine_manager.py

# Run specific test
pytest tests/unit/test_engine_manager.py::TestEngineManager::test_initialization

# Run with coverage
pytest --cov=shared --cov=launchers

# Run only unit tests
pytest tests/unit

# Run only integration tests
pytest tests/integration

# Skip slow tests
pytest -m "not slow"

# Run tests for specific engine
pytest -m mujoco
```

## CI Requirements

All PRs must:
1. ✅ Have all tests passing
2. ✅ Not decrease coverage (unless removing bad tests)
3. ✅ Include tests for new functionality
4. ✅ Follow testing standards in this guide

Tests that will be rejected in code review:
- ❌ Tests that only verify mock interactions
- ❌ Tests with `try/except` that swallow all errors
- ❌ Module-level mocking via `sys.modules`
- ❌ Tests with no assertions
- ❌ Tests that don't test documented behavior

## Examples

See these files for exemplary tests:
- `tests/unit/test_engine_manager.py` - Unit tests with real filesystem
- `tests/unit/test_common_utils_coverage.py` - Testing utility functions
- `tests/integration/test_real_engine_loading.py` - Real integration testing

## Getting Help

Questions about testing? Ask in:
- PR comments
- GitHub Discussions
- Project maintainers

## Further Reading

- [pytest documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://testdriven.io/blog/testing-best-practices/)
- [Effective Python Testing With Pytest](https://realpython.com/pytest-python-testing/)
