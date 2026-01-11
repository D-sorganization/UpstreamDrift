# Assessment L: Data Privacy & Compliance

**Assessment Type**: Privacy Audit
**Rotation Day**: Day 12 (Quarterly)
**Focus**: PII handling, data retention, GDPR/privacy compliance, consent

---

## Objective

Conduct a data privacy audit identifying:

1. Personal data collection points
2. Data retention practices
3. Consent mechanisms
4. Privacy policy compliance
5. Data subject rights implementation

---

## Mandatory Deliverables

### 1. Privacy Summary

- PII collected: Yes/No
- Data categories: X types
- Retention period: X
- Privacy policy: Present/Missing
- Risk level: Low/Medium/High

### 2. Privacy Scorecard

| Category          | Score (0-10) | Weight | Evidence Required  |
| ----------------- | ------------ | ------ | ------------------ |
| Data Minimization |              | 2x     | Collection audit   |
| Consent           |              | 2x     | Consent mechanisms |
| Retention         |              | 1.5x   | Retention policy   |
| Security          |              | 2x     | Encryption, access |
| Rights            |              | 1.5x   | Subject rights     |
| Documentation     |              | 1x     | Privacy docs       |

### 3. Privacy Findings

| ID  | Data Type | Location | Risk | Compliance Gap | Fix | Priority |
| --- | --------- | -------- | ---- | -------------- | --- | -------- |
|     |           |          |      |                |     |          |

---

## Categories to Evaluate

### 1. Data Collection

- [ ] Only necessary data collected
- [ ] Purpose clearly defined
- [ ] Legal basis established
- [ ] No sensitive data without consent

### 2. Consent Management

- [ ] Consent obtained before collection
- [ ] Consent freely given
- [ ] Consent withdrawable
- [ ] Consent records maintained

### 3. Data Security

- [ ] PII encrypted at rest
- [ ] PII encrypted in transit
- [ ] Access controls implemented
- [ ] Audit logging for PII access

### 4. Data Retention

- [ ] Retention periods defined
- [ ] Automatic deletion implemented
- [ ] Backup retention aligned
- [ ] Archive policies documented

### 5. Subject Rights (GDPR)

- [ ] Right to access
- [ ] Right to rectification
- [ ] Right to erasure
- [ ] Right to portability
- [ ] Right to object

### 6. Documentation

- [ ] Privacy policy exists
- [ ] Data processing records
- [ ] DPO designated (if required)
- [ ] DPIA conducted (if required)

---

## Applicability Assessment

| Repository   | Collects PII? | Risk Level | Notes                   |
| ------------ | ------------- | ---------- | ----------------------- |
| Tools        | ❌ No         | Very Low   | Local file tools        |
| Games        | ❌ No         | Very Low   | Local games, save files |
| AffineDrift  | ⚠️ Analytics? | Low        | Check for tracking      |
| Gasification | ❌ No         | Very Low   | Scientific data only    |
| Golf Suite   | ❌ No         | Very Low   | Simulation data only    |

---

## Analysis Commands

```bash
# Search for potential PII patterns
grep -rn "email\|password\|name\|phone\|address\|ssn\|credit" \
  --include="*.py" --include="*.js" | grep -v test

# Check for tracking/analytics
grep -rn "analytics\|tracking\|pixel\|gtag\|facebook" \
  --include="*.js" --include="*.html"

# Check for cookies
grep -rn "cookie\|localStorage\|sessionStorage" \
  --include="*.js"
```

---

## Compliance Framework Reference

| Regulation | Scope      | Key Requirements     |
| ---------- | ---------- | -------------------- |
| GDPR       | EU         | Consent, rights, DPO |
| CCPA       | California | Disclosure, opt-out  |
| PIPEDA     | Canada     | Consent, access      |

---

_Assessment L focuses on data privacy. See Assessment A-K for other dimensions._
