# Completist Report: 2026-02-15

## Overview
This report summarizes the "completeness" of the codebase based on explicit markers (TODOs, FIXMEs, Not Implemented) and structural gaps.

## Metrics

| Category | Count |
|----------|-------|
| TODO Markers | 49 |
| Abstract Methods | 776 |
| Not Implemented | 36 |
| Stub Functions | 440 |
| Missing Docstrings (Approx) | 2694 |

## Critical Gaps

### 1. Incomplete Implementation
Found 36 instances of `NotImplementedError` or `pass` in critical paths.
Examples:
- `./tests/unit/test_launchers.py:88:                    pass  # Help typically causes SystemExit`
- `./tests/unit/test_method_citations.py:28:            pass  # Expected — frozen dataclass`
- `./tests/unit/test_contracts.py:420:                pass  # Expected`
- `./tests/integration/test_physics_interfaces.py:79:        except NotImplementedError:`
- `./tests/integration/test_physics_interfaces.py:80:            pytest.fail("MyoSuite ZTCF/ZVCF raised NotImplementedError")`

### 2. Technical Debt (TODOs)
- `./ui/src/api/useSimulation.test.ts:582:      // We need to go through MAX_RECONNECT_ATTEMPTS (5) reconnection attempts`
- `./ui/src/api/client.ts:33:const MAX_RECONNECT_ATTEMPTS = 5;`
- `./ui/src/api/client.ts:170:      if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {`
- `./ui/src/api/client.ts:172:        console.log(`WebSocket closed unexpectedly. Reconnecting in ${Math.round(delay)}ms (attempt ${reconnectAttemptsRef.current + 1}/${MAX_RECONNECT_ATTEMPTS})`);`
- `./tests/unit/api/test_error_codes.py:36:        """Postcondition: All codes follow GMS-XXX-NNN format."""`

## Recommendations
1.  **Prioritize Not Implemented Errors**: These are runtime risks.
2.  **Burn Down TODOs**: Schedule a "cleanup sprint".
3.  **Enforce Docstrings**: Use CI checks to prevent regression.
