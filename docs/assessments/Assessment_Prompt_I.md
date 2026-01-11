# Assessment I: Accessibility Compliance

**Assessment Type**: Accessibility Audit
**Rotation Day**: Day 9 (Monthly)
**Focus**: WCAG 2.1 AA compliance, screen reader support, keyboard navigation

---

## Objective

Conduct an accessibility audit identifying:

1. WCAG 2.1 AA compliance gaps
2. Keyboard navigation issues
3. Screen reader compatibility
4. Color contrast violations
5. Alternative text and ARIA labels

---

## Mandatory Deliverables

### 1. Accessibility Summary

- WCAG Level: Not Tested / A / AA / AAA
- Critical violations: X
- Color contrast issues: X
- Missing alt text: X instances
- Keyboard traps: X

### 2. Accessibility Scorecard

| Category       | Score (0-10) | Weight | Evidence Required  |
| -------------- | ------------ | ------ | ------------------ |
| Perceivable    |              | 2x     | Alt text, contrast |
| Operable       |              | 2x     | Keyboard, timing   |
| Understandable |              | 1.5x   | Labels, errors     |
| Robust         |              | 1.5x   | ARIA, validation   |
| Screen Reader  |              | 2x     | Testing results    |
| Keyboard       |              | 2x     | Navigation test    |

### 3. Accessibility Findings

| ID  | WCAG | Category | Element | Issue | Impact | Fix | Priority |
| --- | ---- | -------- | ------- | ----- | ------ | --- | -------- |
|     |      |          |         |       |        |     |          |

---

## Categories to Evaluate

### 1. Perceivable (WCAG 1.x)

- [ ] All images have alt text
- [ ] Videos have captions
- [ ] Color not sole indicator
- [ ] Contrast ratio ≥ 4.5:1
- [ ] Text resizable to 200%

### 2. Operable (WCAG 2.x)

- [ ] All functions keyboard accessible
- [ ] No keyboard traps
- [ ] Skip links for navigation
- [ ] Focus indicators visible
- [ ] No seizure-inducing content

### 3. Understandable (WCAG 3.x)

- [ ] Language declared
- [ ] Navigation consistent
- [ ] Labels and instructions
- [ ] Error identification
- [ ] Input assistance

### 4. Robust (WCAG 4.x)

- [ ] Valid HTML/markup
- [ ] ARIA properly used
- [ ] Name, Role, Value
- [ ] Status messages

---

## Applicability by Repository

| Repository   | Type             | Accessibility Relevance   |
| ------------ | ---------------- | ------------------------- |
| Tools        | Desktop (PyQt6)  | Medium - GUI widgets      |
| Games        | Desktop (Pygame) | Low - Visual games        |
| AffineDrift  | Web (Quarto)     | **High** - Web content    |
| Gasification | Desktop (PyQt6)  | Medium - Scientific UI    |
| Golf Suite   | Desktop (PyQt6)  | Medium - 3D visualization |

---

## Testing Commands

### Web (AffineDrift)

```bash
# axe-core automated testing
npm install -g @axe-core/cli
axe https://affinedrift.com --dir ./accessibility-report

# Pa11y
npm install -g pa11y
pa11y https://affinedrift.com

# Lighthouse
npm install -g lighthouse
lighthouse https://affinedrift.com --only-categories=accessibility
```

### Desktop (PyQt6/Tkinter)

- Manual screen reader testing (NVDA, JAWS)
- Keyboard navigation testing
- High contrast mode testing

---

## WCAG Quick Reference

| Level | Requirement             |
| ----- | ----------------------- |
| A     | Minimum (must have)     |
| AA    | Standard (recommended)  |
| AAA   | Enhanced (aspirational) |

---

_Assessment I focuses on accessibility. See Assessment A-H for other dimensions._
