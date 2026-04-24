---
type: reference
skill: eval-run
id: EVALS-FORMAT-001
updated_at: 2026-03-15
---

# evals.yaml Format Specification

This document defines the canonical format for `evals.yaml` files used by the `eval-run` skill.

---

## Overview

An `evals.yaml` file contains a set of assertions to be run against a single output file. It is authored by the Discovery Agent as part of the Skill Optimize protocol. It is consumed by the `eval-run` skill as an immutable input.

---

## Top-Level Structure

```yaml
skill: { skill-name } # The name of the skill whose output is being evaluated
version: "1.0" # Version of this evals file (semantic version string)
description: { string } # One-sentence description of what this eval set covers
assertions:
  - { assertion } # One or more assertions (see below)
```

### Required Fields

- id: { assertion-id } # Unique identifier within this evals file (e.g. A01)
  type: code
  description: { string } # Human-readable description of what is being checked
  check: { check-type } # One of: word_count, char_count, regex_match, regex_not_match, structure_present, structure_absent
  params:
  { param-key }: { param-value } # Parameters required by the check type (see below)
  expected: { pass-condition } # Description of the condition that constitutes PASS

```

#### Supported `check` Values and Their `params`

- id: { assertion-id } # Unique identifier within this evals file
  type: llm-judge
  description: { string } # Human-readable description of what is being evaluated
  prompt: |
    {evaluation prompt}       # The prompt given to the LLM judge. Must end with a binary question.
  rubric: # Criteria that define PASS
    pass_if: { string } # The condition under which the LLM judge should answer PASS
    fail_if: { string } # The condition under which the LLM judge should answer FAIL
```

#### Prompt Pattern Rules

The `prompt` field must:
