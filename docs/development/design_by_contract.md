# Design by Contract Guide

> **Location**: `src/shared/python/contracts.py`

Design by Contract (DbC) is a software correctness methodology that treats specifications as contracts between callers and implementations.

## Core Concepts

### Preconditions

What must be true **before** a method executes.

- Checked at method entry
- Caller's responsibility to satisfy
- Raises `PreconditionError` if violated

### Postconditions

What must be true **after** a method completes.

- Checked at method exit
- Implementation's responsibility
- Raises `PostconditionError` if violated

### Invariants

What must **always** be true for an object.

- Checked after state-modifying methods
- Class's responsibility
- Raises `InvariantError` if violated

## Usage Examples

### Preconditions

```python
from src.shared.python.contracts import precondition

@precondition(lambda x: x > 0, "x must be positive")
def square_root(x: float) -> float:
    return math.sqrt(x)

# OK
square_root(4)  # Returns 2.0

# Raises PreconditionError: x must be positive
square_root(-1)
```

### Method Preconditions with self

```python
from src.shared.python.contracts import precondition

class Engine:
    _is_initialized = False

    def initialize(self):
        self._is_initialized = True

    @precondition(lambda self: self._is_initialized, "Engine must be initialized")
    def step(self, dt: float):
        # Simulation step
        pass
```

### Postconditions

```python
from src.shared.python.contracts import postcondition

@postcondition(lambda result: result is not None, "Must return a result")
@postcondition(lambda result: len(result) > 0, "Result must not be empty")
def get_engines() -> list[str]:
    return ["mujoco", "drake"]
```

### State Requirements

```python
from src.shared.python.contracts import require_state

class PhysicsEngine:
    _model_loaded = False

    @require_state(lambda self: self._model_loaded, "model loaded")
    def simulate(self):
        # Requires model to be loaded first
        pass
```

### Class Invariants

```python
from src.shared.python.contracts import ContractChecker, invariant_checked

class RigidBody(ContractChecker):
    def __init__(self, mass: float):
        self._mass = mass

    def _get_invariants(self):
        return [
            (lambda: self._mass > 0, "mass must be positive"),
        ]

    @invariant_checked
    def set_mass(self, mass: float):
        self._mass = mass  # Invariant checked after this
```

## Built-in Validators

The contracts module includes common validation functions:

```python
from src.shared.python.contracts import (
    check_finite,         # No NaN or Inf in array
    check_shape,          # Array has expected shape
    check_positive,       # Value(s) > 0
    check_non_negative,   # Value(s) >= 0
    check_symmetric,      # Matrix is symmetric
    check_positive_definite,  # Matrix is positive definite
)

# Example usage
@postcondition(lambda arr: check_finite(arr), "Result must be finite")
def compute_jacobian() -> np.ndarray:
    ...
```

## Convenience Decorators

```python
from src.shared.python.contracts import (
    requires_initialized,    # Requires self._is_initialized
    requires_model_loaded,   # Requires self.model is not None
    finite_result,          # Result has no NaN/Inf
    non_empty_result,       # Result is not empty
)

class Engine:
    _is_initialized = False
    model = None

    @requires_initialized
    @finite_result
    def get_state(self) -> np.ndarray:
        ...
```

## Error Types

| Exception                | When Raised                 |
| ------------------------ | --------------------------- |
| `PreconditionError`      | Input validation failed     |
| `PostconditionError`     | Output validation failed    |
| `InvariantError`         | Class invariant violated    |
| `StateError`             | Invalid state for operation |
| `ContractViolationError` | Base class for all          |

## Error Messages

Errors include diagnostic information:

```python
PreconditionError: Precondition violation in Engine.step: Engine must be initialized
  Details: {'parameter': 'self', 'value': '<Engine object>'}
```

## Performance Considerations

### Disabling in Production

Contracts can be disabled globally for performance:

```python
from src.shared.python.contracts import (
    enable_contracts,
    disable_contracts,
    contracts_enabled,
)

# In production startup
disable_contracts()

# Check status
if contracts_enabled():
    print("Contracts are active")
```

### Per-Contract Disable

```python
@precondition(lambda x: x > 0, "x must be positive", enabled=False)
def fast_function(x):
    ...
```

## Best Practices

### DO

- Use preconditions for parameter validation
- Use postconditions for return value guarantees
- Use invariants for class state consistency
- Keep conditions simple and fast
- Provide clear error messages

### DON'T

- Use contracts for business logic
- Put side effects in contract conditions
- Use slow operations in frequently-called contracts
- Catch and ignore contract violations

## Integration with API

The API layer uses contracts for validation:

```python
# In services
class SimulationService:
    @precondition(lambda self: self.engine_manager is not None,
                  "Engine manager required")
    @postcondition(lambda result: result.success or result.error,
                   "Must have result or error")
    async def run_simulation(self, request) -> SimulationResponse:
        ...
```

Combined with structured error codes:

```python
from src.api.utils import ErrorCode, raise_api_error

try:
    result = await service.run_simulation(request)
except PreconditionError as e:
    raise_api_error(
        ErrorCode.SIMULATION_INVALID_PARAMS,
        message=str(e),
        contract_type="precondition"
    )
```

## See Also

- [API Development Guide](../api/DEVELOPMENT.md)
- [API Architecture](../api/API_ARCHITECTURE.md)
- Original: "Object-Oriented Software Construction" by Bertrand Meyer
