# Assessment J: API Design & Interface Quality

**Assessment Type**: API Design Review
**Rotation Day**: Day 10 (Monthly)
**Focus**: Interface consistency, backward compatibility, developer experience

---

## Objective

Conduct an API design audit identifying:

1. Interface consistency issues
2. Breaking changes potential
3. Documentation gaps
4. Naming convention violations
5. Error handling patterns

---

## Mandatory Deliverables

### 1. API Health Summary

- Public functions/classes: X
- Documented: X%
- Breaking changes risk: Low/Medium/High
- Consistency score: X/10

### 2. API Design Scorecard

| Category               | Score (0-10) | Weight | Evidence Required  |
| ---------------------- | ------------ | ------ | ------------------ |
| Naming Consistency     |              | 2x     | Convention audit   |
| Parameter Design       |              | 2x     | Signature review   |
| Return Types           |              | 1.5x   | Type hints         |
| Error Handling         |              | 2x     | Exception patterns |
| Documentation          |              | 2x     | Docstring coverage |
| Backward Compatibility |              | 1.5x   | Deprecation audit  |

### 3. API Findings

| ID  | Location | Issue | Impact | Breaking? | Fix | Priority |
| --- | -------- | ----- | ------ | --------- | --- | -------- |
|     |          |       |        |           |     |          |

---

## Categories to Evaluate

### 1. Naming Conventions

- [ ] Consistent naming across modules
- [ ] PEP 8 compliance (snake_case)
- [ ] Meaningful, descriptive names
- [ ] No abbreviations without context

### 2. Function Signatures

- [ ] Reasonable parameter count (≤5)
- [ ] Sensible defaults
- [ ] Keyword-only for optional params
- [ ] \*args/\*\*kwargs used appropriately

### 3. Type Hints

- [ ] All public functions typed
- [ ] Return types specified
- [ ] Union types minimal
- [ ] Generic types used correctly

### 4. Error Handling

- [ ] Custom exceptions defined
- [ ] Exceptions documented
- [ ] Error messages helpful
- [ ] Graceful degradation

### 5. Documentation

- [ ] All public functions documented
- [ ] Examples in docstrings
- [ ] Parameter descriptions
- [ ] Return value documented

### 6. Versioning & Deprecation

- [ ] Semver followed
- [ ] Deprecation warnings used
- [ ] Migration guides provided
- [ ] CHANGELOG maintained

---

## Analysis Commands

```bash
# Check public API surface
grep -rn "^def \|^class " --include="*.py" | grep -v "^_" | wc -l

# Check docstring coverage
pip install interrogate
interrogate -vv . --fail-under 80

# Check type hint coverage
pip install mypy
mypy --strict . 2>&1 | grep "error:" | wc -l

# Analyze function signatures
ast-grep --pattern 'def $NAME($$$PARAMS): $$$BODY' .
```

---

## Best Practices Checklist

### Interface Design

- [ ] Principle of Least Surprise
- [ ] Make wrong code look wrong
- [ ] Fail fast and explicitly
- [ ] Provide sensible defaults

### Compatibility

- [ ] Semantic versioning
- [ ] Deprecate before removing
- [ ] Provide migration path
- [ ] Test backward compatibility

---

_Assessment J focuses on API design. See Assessment A-I for other dimensions._
