# The Pragmatic Programmer — Core Principles (Faithful Expansion)

## 1. Responsibility: *You Own Your Work*

> **“The Pragmatic Programmer takes responsibility.”**

### Meaning (original intent)

You don’t blame tools, languages, reviewers, collaborators, or requirements.
If something is confusing, fragile, undocumented, or misleading — **it is your responsibility** to fix or surface it.

### In a research / AffineDrift context

* If a derivation is “obvious,” it isn’t.
* If a script “only works if you know how,” it’s broken.
* If a reviewer could misinterpret something, they will.

**Pragmatic rule:**
If you *know* a reader or future-you could get lost, you are obligated to add clarity — not excuses.

---

## 2. Communication Is the Job

> **“Programming is about communication, not typing.”**

### Meaning

Code, math, and documentation are all **executable communication**.
Elegance is measured by how little the reader has to guess.

### Applied rigor

* Variable names are part of the proof.
* Function boundaries are part of the argument.
* File structure is part of the model.

In AffineDrift terms:

> If the control structure is affine, the code structure should be affine too — drift vs input should not be mentally reconstructed.

---

## 3. Don’t Live with Broken Windows

> **“Don’t tolerate broken windows.”**

### Meaning

Small flaws compound into systemic decay.

Broken windows include:

* dead code
* commented-out blocks
* unexplained constants
* duplicated logic with “slight differences”
* TODOs that never move

### Research translation

* A hand-wavy paragraph weakens the entire paper.
* A silent assumption invalidates downstream rigor.
* An undocumented preprocessing step poisons results.

**Pragmatic response:**
Fix small problems *immediately*, or explicitly quarantine them.

---

## 4. Be a Catalyst for Change (But Keep It Reversible)

> **“Make reversible decisions.”**

### Meaning

You cannot predict the future — so **avoid decisions that lock you in**.

### In practice

* Prefer configuration over hardcoding
* Prefer modular decomposition over monoliths
* Prefer data-driven definitions over structural duplication

### In AffineDrift

* New biomechanical joints should not require refactoring the entire solver
* New cost functions should not require rewriting dynamics
* New datasets should not change core abstractions

If a change is hard to undo, it must be *exceptionally justified*.

---

## 5. DRY — Don’t Repeat Yourself (The Real Meaning)

> **“Every piece of knowledge must have a single, authoritative representation.”**

### This is NOT “no duplicate lines”

It is **no duplicate facts, assumptions, or rules**.

Violations in research code:

* same equation derived twice in different notation
* same constraint enforced in two places
* same parameter defined in code *and* prose independently

**Pragmatic test:**
If something changes, how many places must be updated to remain correct?

If the answer >1, DRY is violated.

---

## 6. Orthogonality: Keep Concerns Independent

> **“Design components that are orthogonal.”**

### Meaning

Changing one thing should not ripple through unrelated things.

### Scientific mapping

* Mathematics ≠ numerics ≠ visualization
* Kinematics ≠ dynamics ≠ optimization
* Model structure ≠ solver strategy

### Red flag

If adding a new plot breaks an optimizer, your system is entangled.

Orthogonality is **the enabler of research velocity**.

---

## 7. Don’t Program by Coincidence

> **“Understand what you are doing.”**

### Meaning

If something works and you don’t know why, it is already broken.

### In research

* “The solver converges if I set this tolerance low” is not an explanation
* “This regularization stabilizes things” is not a justification
* “It matches the data” is not a proof

**Pragmatic discipline:**
If you can’t explain it, you must isolate it.

---

## 8. Estimate, Then Refine

> **“Estimate to avoid surprises.”**

### Meaning

You don’t need perfect predictions — you need *bounded ignorance*.

### Applied

* Estimate runtime before launching a sweep
* Estimate sensitivity before tuning parameters
* Estimate refactor scope before touching files

This prevents both:

* premature optimization
* endless procrastination

---

## 9. Keep Knowledge in Plain Text

> **“Plain text is durable.”**

### Meaning

Knowledge should survive tools, platforms, and time.

### Research implication

* YAML / JSON / CSV over opaque binaries
* Quarto / Markdown over proprietary formats
* Scripts that regenerate figures over stored figures

AffineDrift benefits enormously from this principle already — lean into it.

---

## 10. Design by Contract (Even If Informal)

> **“Specify and check contracts.”**

### Meaning

Every function/module has:

* what it expects
* what it guarantees
* what it does not promise

### In math-heavy code

* input units
* coordinate frames
* assumptions (smoothness, invertibility, convexity)

If contracts are implicit, bugs become philosophical debates.

---

## 11. Test What You Mean, Not What You Wrote

> **“Test for intent.”**

### Meaning

Tests should verify **behavioral truth**, not implementation detail.

### Research equivalent

* invariants (energy, constraints)
* limiting cases (zero torque, rigid shaft)
* symmetry checks
* conservation laws

If a refactor breaks a test, either the test was wrong or the refactor was.

---

## 12. Refactor Early, Refactor Often (But Gently)

> **“Refactoring is not rewriting.”**

### Meaning

Refactoring improves structure **without changing behavior**.

### Pragmatic constraint

* small steps
* frequent checkpoints
* always runnable

In AffineDrift:

> If a refactor changes results, it was not a refactor.

---

## 13. Avoid the “Big Design Up Front” Trap

> **“Design evolves.”**

### Meaning

Over-architecting before understanding the problem is worse than under-designing.

### Balance

* enough structure to move safely
* not so much that ideas are frozen

AffineDrift is *exactly* the kind of project that benefits from evolutionary design.

---

## 14. Build for People Who Are Not You

> **“Your code will be read more than it is written.”**

Future readers include:

* future you (the most dangerous one)
* skeptical reviewers
* collaborators
* intelligent outsiders

If understanding requires oral tradition, the system is fragile.

---

## 15. Pride in Craft (Without Ego)

> **“Care about what you build.”**

Not perfectionism.
Not cleverness.
**Care.**

Care shows up as:

* explicit assumptions
* honest limitations
* clean exits
* readable failures

---

# How This Becomes an Agent Constitution

Your **Pragmatic Programmer Agent** should explicitly check:

* ❓ What assumption is hidden here?
* ❓ What would future-me misunderstand?
* ❓ What change would be hardest to undo?
* ❓ Where is knowledge duplicated?
* ❓ What is coincidental vs causal?
