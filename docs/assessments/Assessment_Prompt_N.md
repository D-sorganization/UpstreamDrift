# Assessment N: Reliability & Error Recovery

**Assessment Type**: Reliability Audit
**Rotation Day**: Day 14 (Bi-monthly)
**Focus**: Fault tolerance, error recovery, graceful degradation, resilience

---

## Objective

Conduct a reliability audit identifying:

1. Single points of failure
2. Error recovery mechanisms
3. Graceful degradation patterns
4. State preservation
5. Crash recovery

---

## Mandatory Deliverables

### 1. Reliability Summary

- Critical failure points: X
- Recovery mechanisms: X implemented
- Error handling: Comprehensive/Partial/Minimal
- Mean time to recovery: X

### 2. Reliability Scorecard

| Category             | Score (0-10) | Weight | Evidence Required   |
| -------------------- | ------------ | ------ | ------------------- |
| Error Handling       |              | 2x     | Exception coverage  |
| Recovery             |              | 2x     | Recovery mechanisms |
| Graceful Degradation |              | 1.5x   | Fallback behavior   |
| State Preservation   |              | 2x     | Save/restore logic  |
| Logging              |              | 1.5x   | Diagnostic logs     |
| Monitoring           |              | 1.5x   | Health checks       |

### 3. Reliability Findings

| ID  | Component | Failure Mode | Impact | Recovery | Fix | Priority |
| --- | --------- | ------------ | ------ | -------- | --- | -------- |
|     |           |              |        |          |     |          |

---

## Categories to Evaluate

### 1. Error Handling

- [ ] All exceptions caught appropriately
- [ ] No bare except: clauses
- [ ] Error messages user-friendly
- [ ] Stack traces logged (not shown to user)

### 2. Recovery Mechanisms

- [ ] Auto-save for user work
- [ ] Crash recovery on restart
- [ ] Retry logic for transient failures
- [ ] Rollback capability

### 3. Graceful Degradation

- [ ] Continues with reduced functionality
- [ ] Clear messaging for degraded state
- [ ] Core features prioritized
- [ ] Non-critical failures isolated

### 4. State Preservation

- [ ] User state persisted
- [ ] Settings saved
- [ ] Work-in-progress recoverable
- [ ] Undo/redo capability

### 5. Logging & Diagnostics

- [ ] Structured logging implemented
- [ ] Log levels appropriate
- [ ] Correlation IDs for tracing
- [ ] Log rotation configured

### 6. Health & Monitoring

- [ ] Health check endpoints (if applicable)
- [ ] Resource monitoring
- [ ] Alert thresholds defined
- [ ] Status reporting

---

## Failure Mode Analysis

### Common Failure Scenarios

| Scenario        | Expected Behavior      | Tested? |
| --------------- | ---------------------- | ------- |
| File not found  | Graceful error message |         |
| Network timeout | Retry with backoff     |         |
| Out of memory   | Warning before crash   |         |
| Invalid input   | Validation message     |         |
| Corrupted save  | Fallback to backup     |         |

---

## Analysis Commands

```bash
# Find bare except clauses
grep -rn "except:" --include="*.py" | grep -v "except: *#"

# Find error handling patterns
grep -rn "try:" --include="*.py" | wc -l
grep -rn "except \w" --include="*.py" | wc -l

# Check for logging
grep -rn "logger\.\|logging\." --include="*.py" | wc -l

# Check for recovery patterns
grep -rn "retry\|recover\|backup\|restore" --include="*.py"
```

---

## Reliability Patterns

### Recommended Implementations

- [ ] Circuit breaker for external calls
- [ ] Timeout wrappers
- [ ] Retry with exponential backoff
- [ ] Bulkhead isolation
- [ ] Fallback responses

---

_Assessment N focuses on reliability. See Assessment A-M for other dimensions._
