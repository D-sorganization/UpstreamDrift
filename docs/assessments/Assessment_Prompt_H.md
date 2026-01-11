# Assessment H: Maintainability & Technical Debt

**Assessment Type**: Code Maintainability Audit
**Rotation Day**: Day 8 (Bi-weekly)
**Focus**: Cyclomatic complexity, code smells, technical debt, refactoring needs

---

## Objective

Conduct a maintainability audit identifying:

1. High-complexity functions requiring simplification
2. Code duplication and DRY violations
3. Technical debt accumulation
4. Refactoring opportunities
5. Long-term maintenance risks

---

## Mandatory Deliverables

### 1. Maintainability Summary

- Total files: X
- Average complexity: X
- High-complexity functions: X
- Code duplication: X%
- Technical debt estimate: X hours

### 2. Maintainability Scorecard

| Category              | Score (0-10) | Weight | Evidence Required    |
| --------------------- | ------------ | ------ | -------------------- |
| Cyclomatic Complexity |              | 2x     | radon/flake8 results |
| Code Duplication      |              | 2x     | jscpd/pylint results |
| Function Length       |              | 1.5x   | LOC per function     |
| Coupling              |              | 1.5x   | Import analysis      |
| Cohesion              |              | 1.5x   | Module analysis      |
| Naming Quality        |              | 1x     | Identifier review    |

### 3. Technical Debt Register

| ID  | Location | Debt Type | Severity | Effort | Interest | Priority |
| --- | -------- | --------- | -------- | ------ | -------- | -------- |
|     |          |           |          |        |          |          |

---

## Categories to Evaluate

### 1. Cyclomatic Complexity

- [ ] No function exceeds complexity 15
- [ ] Average complexity < 5
- [ ] Complex functions documented
- [ ] Refactoring plan for high-complexity

### 2. Code Duplication

- [ ] Duplication < 5%
- [ ] No copy-paste anti-patterns
- [ ] Shared utilities extracted
- [ ] DRY principle followed

### 3. Function/Method Size

- [ ] Functions < 50 lines average
- [ ] No function > 100 lines
- [ ] Single responsibility per function
- [ ] Clear function boundaries

### 4. Coupling & Cohesion

- [ ] Low coupling between modules
- [ ] High cohesion within modules
- [ ] Clear dependency direction
- [ ] No circular imports

### 5. Dead Code

- [ ] No unreachable code
- [ ] Unused imports removed
- [ ] Commented code removed
- [ ] Deprecated code marked

### 6. Technical Debt Markers

- [ ] TODO count tracked
- [ ] FIXME count tracked
- [ ] HACK count tracked
- [ ] Debt documented and planned

---

## Analysis Commands

```bash
# Cyclomatic complexity with radon
pip install radon
radon cc . -a -s -n C  # Show C grade and worse

# Maintainability index
radon mi . -s

# Code duplication
pip install jscpd
jscpd . --reporters console --format "python"

# Dead code detection
pip install vulture
vulture . --min-confidence 80

# Combined metrics with wily
pip install wily
wily build .
wily report .

# TODO/FIXME counting
grep -rn "TODO\|FIXME\|HACK\|XXX" --include="*.py" | wc -l
```

---

## Complexity Thresholds

| Grade | Complexity | Maintainability |
| ----- | ---------- | --------------- |
| A     | 1-5        | Excellent       |
| B     | 6-10       | Good            |
| C     | 11-15      | Moderate        |
| D     | 16-20      | Poor            |
| F     | 21+        | Unmaintainable  |

---

_Assessment H focuses on maintainability. See Assessment A-G for other dimensions._
