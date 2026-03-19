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
skill: {skill-name}           # The name of the skill whose output is being evaluated
version: "1.0"                # Version of this evals file (semantic version string)
description: {string}         # One-sentence description of what this eval set covers
assertions:
  - {assertion}               # One or more assertions (see below)
```

### Required Fields

| Field | Type | Description |
|---|---|---|
| `skill` | string | Name of the target skill (must match `SKILL.md` `name` field) |
| `version` | string | Semantic version of this evals file |
| `description` | string | One sentence describing what is being evaluated |
| `assertions` | list | One or more assertion objects (see assertion types below) |

---

## Assertion Types

Two assertion types are supported: `code` and `llm-judge`.

### Type: `code`

Mechanically verifiable assertions that do not require LLM evaluation. The `eval-run` skill executes these deterministically using counts, regex matching, or structural checks.

```yaml
- id: {assertion-id}          # Unique identifier within this evals file (e.g. A01)
  type: code
  description: {string}       # Human-readable description of what is being checked
  check: {check-type}         # One of: word_count, char_count, regex_match, regex_not_match, structure_present, structure_absent
  params:
    {param-key}: {param-value} # Parameters required by the check type (see below)
  expected: {pass-condition}  # Description of the condition that constitutes PASS
```

#### Supported `check` Values and Their `params`

| `check` | Required `params` | Description |
|---|---|---|
| `word_count` | `min`, `max` | Word count must be within [min, max] (inclusive) |
| `char_count` | `min`, `max` | Character count must be within [min, max] (inclusive) |
| `regex_match` | `pattern` | The output must contain at least one match for `pattern` |
| `regex_not_match` | `pattern` | The output must contain zero matches for `pattern` |
| `structure_present` | `marker` | The output must contain the literal string `marker` (e.g., a required section heading) |
| `structure_absent` | `marker` | The output must NOT contain the literal string `marker` |

#### Example

```yaml
- id: A01
  type: code
  description: "Word count is within ±15% of 1200-word target"
  check: word_count
  params:
    min: 1020
    max: 1380
  expected: "Word count between 1020 and 1380"

- id: A02
  type: code
  description: "Kill list word 'leverage' does not appear"
  check: regex_not_match
  params:
    pattern: "\\bleverag(e|ing|ed)\\b"
  expected: "Zero matches for the pattern"

- id: A03
  type: code
  description: "Output contains a Hook section"
  check: structure_present
  params:
    marker: "## Hook"
  expected: "Marker '## Hook' present in output"
```

---

### Type: `llm-judge`

Assertions that require LLM judgment to evaluate. The `eval-run` skill uses a structured prompt pattern to produce a binary PASS/FAIL result with a rationale.

```yaml
- id: {assertion-id}          # Unique identifier within this evals file
  type: llm-judge
  description: {string}       # Human-readable description of what is being evaluated
  prompt: |
    {evaluation prompt}       # The prompt given to the LLM judge. Must end with a binary question.
  rubric:                     # Criteria that define PASS
    pass_if: {string}         # The condition under which the LLM judge should answer PASS
    fail_if: {string}         # The condition under which the LLM judge should answer FAIL
```

#### Prompt Pattern Rules

The `prompt` field must:
1. Provide the context needed to evaluate the output (e.g., "You are evaluating a LinkedIn post draft")
2. State the criterion being evaluated clearly
3. End with a binary question: "Does the output [criterion]? Answer PASS or FAIL, then explain in one sentence."

The `rubric` field describes the PASS/FAIL boundary in plain English for auditability.

#### Example

```yaml
- id: A04
  type: llm-judge
  description: "Post is written in first-person singular"
  prompt: |
    You are evaluating a LinkedIn post draft. The post should be written in first-person singular voice (using "I", not "we" or second-person).
    Review the output below and determine: Is the post written consistently in first-person singular voice?
    Answer PASS or FAIL, then explain in one sentence.
  rubric:
    pass_if: "The post uses first-person singular voice ('I') throughout, with no 'we' or 'you' as primary voice"
    fail_if: "The post uses 'we', 'you', or third-person as the primary voice"

- id: A05
  type: llm-judge
  description: "Standalone test: post makes sense without prior context"
  prompt: |
    You are evaluating a LinkedIn post draft as if you have never seen any prior content from this author.
    Read the post below and determine: Does this post stand alone and make complete sense without any prior context?
    Answer PASS or FAIL, then explain in one sentence.
  rubric:
    pass_if: "A reader with no prior context can fully understand the post and its point"
    fail_if: "The post references prior context, uses unexplained abbreviations, or assumes knowledge the reader cannot have"
```

---

## Complete evals.yaml Example

```yaml
skill: content-draft
version: "1.0"
description: "Baseline eval set for LinkedIn post drafts produced by content-draft (CNT-003)"
assertions:
  - id: A01
    type: code
    description: "Word count is within ±15% of 1200-word target"
    check: word_count
    params:
      min: 1020
      max: 1380
    expected: "Word count between 1020 and 1380"

  - id: A02
    type: code
    description: "Kill list word 'leverage' does not appear"
    check: regex_not_match
    params:
      pattern: "\\bleverag(e|ing|ed)\\b"
    expected: "Zero matches for the pattern"

  - id: A03
    type: code
    description: "Output contains a Hook section"
    check: structure_present
    params:
      marker: "## Hook"
    expected: "Marker '## Hook' present in output"

  - id: A04
    type: llm-judge
    description: "Post is written in first-person singular"
    prompt: |
      You are evaluating a LinkedIn post draft. The post should be written in first-person singular voice (using "I", not "we" or second-person).
      Review the output below and determine: Is the post written consistently in first-person singular voice?
      Answer PASS or FAIL, then explain in one sentence.
    rubric:
      pass_if: "The post uses first-person singular voice ('I') throughout"
      fail_if: "The post uses 'we', 'you', or third-person as the primary voice"

  - id: A05
    type: llm-judge
    description: "Standalone test: post makes sense without prior context"
    prompt: |
      You are evaluating a LinkedIn post draft as if you have never seen any prior content from this author.
      Read the post below and determine: Does this post stand alone and make complete sense without any prior context?
      Answer PASS or FAIL, then explain in one sentence.
    rubric:
      pass_if: "A reader with no prior context can fully understand the post"
      fail_if: "The post references prior context or assumes knowledge the reader cannot have"
```

---

## Constraints and Validation Rules

1. Every assertion `id` must be unique within the file.
2. `type` must be exactly `code` or `llm-judge` — no other values are valid.
3. `code` assertions must have a `check` field with a supported value.
4. `code` assertions must have all required `params` for the selected `check`.
5. `llm-judge` assertions must have both `prompt` and `rubric` fields.
6. `llm-judge` prompts must end with a binary question ("Answer PASS or FAIL").
7. `assertions` list must contain at least one entry.

---

## What an evals.yaml Does NOT Contain

- Instructions to invoke the skill being evaluated
- Instructions to modify the output
- Thresholds for overall pass rate (that is a Skill Optimize protocol decision)
- Comparison logic between runs (that is agent orchestration)
