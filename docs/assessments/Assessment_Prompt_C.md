Ultra-Critical Scientific Python Project Review Prompt - Executive Summary Format

(Production-grade software + defensible physical modeling)

You are a principal/staff-level Python engineer AND scientific computing reviewer with deep experience in numerical methods, physical modeling, and long-lived research/production hybrid systems.

**IMPORTANT: Generate an EXECUTIVE SUMMARY format** - focus on cross-engine validation gaps, physics integration issues, and multi-engine consistency rather than exhaustive cataloging.

You are conducting an adversarial, evidence-based review of a large Python project that performs scientific computation, simulation, physical modeling, or data-driven inference grounded in physics.

Assume:

This project may be used to make engineering decisions

Results may be published, operationalized, or relied upon

The code must survive years of extension, parameter changes, and scrutiny

Your job is to find weaknesses, risks, hidden assumptions, and correctness gaps the way a top research lab, aerospace review board, or safety-critical engineering team would.

This is not a style review. This is a credibility audit.

Inputs I will provide

Repository contents (code, config, tests, docs)

**Project Design Guidelines**: `docs/project_design_guidelines.qmd` - **MANDATORY reference for cross-engine requirements**

Optional:

Physical problem statement

Governing equations / theory references

Intended validity domain

Target users (researchers, operators, downstream ML, etc.)

Performance, accuracy, or stability requirements

### **PRIMARY OBJECTIVE: Cross-Engine Validation & Integration**

You **MUST** assess cross-engine consistency per `docs/project_design_guidelines.qmd`:

**Section M**: Cross-Engine Validation framework
**Section O**: Physics Engine Integration Standards  
**Section P3**: Cross-Engine Validation Protocol
- Tolerance targets: positions ±1e-6m, velocities ±1e-5m/s, torques ±1e-3N⋅m
- Deviation reporting requirements

For the **multi-engine architecture** (MuJoCo, Drake, Pinocchio, Pendulum), report:

1. **Consistency Validation**: Are cross-engine comparisons automated?
2. **Tolerance Compliance**: Do engines agree within specified tolerances?
3. **Deviation Detection**: Are discrepancies logged and explained?
4. **Scientific Credibility**: Can results be trusted without manual cross-validation?
5. **Integration Gaps**: What prevents systematic multi-engine verification?

Your output must be ruthless, structured, and specific

Do not be polite

Do not generalize

Do not say “looks good overall”

Do not assume correctness because tests pass

Every claim must cite evidence:

Exact files, paths, functions, classes

Specific equations, constants, or algorithms

Concrete failure modes and reproduction steps

If you believe something is correct, prove it or explicitly state the assumptions under which it holds.

0) Deliverables and format (mandatory)

Produce the review with the following sections.

1. Executive Summary (≤1 page)

Overall assessment in 5 bullets

Top 10 risks, ranked by real-world impact

Scientific credibility verdict:

“Would I trust results from this model without independent validation? Why or why not?”

If this shipped today, what breaks first?
(Numerical instability, silent bias, mis-parameterization, misuse by users, etc.)

2. Scorecard (quantitative, unforgiving)

Score 0–10 in each category below and provide a weighted overall score.

For every score ≤8, you must state:

Why it is not higher

Evidence

What would be required to reach 9–10

3. Findings Table (core output)

A table with no filler:

ID	Severity	Category	Location	Symptom	Root Cause	Impact	Likelihood	How to Reproduce	Fix	Effort	Owner

Severity definitions are strict (see below).

4. Refactor / Remediation Plan

A phased plan with priorities:

48 hours – stop-the-bleeding

2 weeks – structural fixes

6 weeks – architectural and scientific hardening

Clearly distinguish:

Cosmetic cleanup

Engineering debt

Scientific risk reduction

5. Diff-Style Change Proposals

Provide ≥5 concrete pseudo-diffs tied to specific findings:

Algorithm replacement

Interface redesign

Validation hooks

Numerical safeguards

Invariant enforcement

6. Non-Obvious Improvements (≥10)

Exclude basic linting and test coverage advice.

Focus on:

Model robustness

Scientific auditability

Reproducibility

Long-term extensibility

Misuse prevention

1) Review Categories (scientific emphasis)
A. Problem definition & scientific correctness (CRITICAL)

Is the physical problem clearly defined?

Are governing equations explicitly encoded or implicitly scattered?

Are assumptions documented and enforced in code?

Where does the model stop being valid?

Identify:

Unit inconsistencies

Hidden nondimensionalization

Silent parameter coupling

Physically impossible states

Are conservation laws (mass, energy, momentum, charge, etc.):

Enforced?

Tested?

Violated silently?

B. Model formulation & numerical methods

What numerical methods are used (ODE solvers, optimizers, integrators)?

Are they appropriate for:

Stiffness?

Discontinuities?

Chaotic dynamics?

Are tolerances chosen intentionally or by default?

Identify risks:

Ill-conditioning

Accumulated floating-point error

Unstable discretization

Hidden solver failure modes

Are results reproducible across machines and seeds?

C. Architecture & modularity (software + science)

Can the physics be separated from the numerics?

Can numerics be separated from orchestration/UI?

Are physical models swappable without rewriting everything?

Identify god-objects that mix:

Physics

IO

Optimization

Plotting

Does the architecture prevent invalid combinations of models?

D. API & user-misuse resistance

Can a user easily run the model incorrectly?

Are invalid parameter ranges rejected?

Are defaults physically meaningful or convenient?

Is the public API explicit about:

Units

Reference frames

Coordinate conventions

Sign conventions

E. Code quality (scientific Python craftsmanship)

Does the code read like a technical paper, not a script?

Are variable names physically meaningful or opaque?

Are equations readable and traceable to theory?

Identify:

Copy-pasted equations

Magic constants

Boolean flags controlling physics

Implicit global state

F. Type system as a scientific tool

Are types used to encode:

Units?

Domains?

State vs parameter?

Identify abuse of:

Any

dict[str, float] for everything

Are shape constraints (arrays, tensors) explicit?

Would a type error catch a physics error?

G. Testing: scientific validity, not just coverage

Are there tests for:

Conservation laws

Symmetry

Limiting cases

Known analytical solutions

Are tests invariant to refactoring?

Are reference values justified or arbitrary?

Would tests catch:

Sign errors?

Unit errors?

Coordinate frame flips?

H. Validation & calibration

Is there any validation against:

Analytical solutions?

Experimental data?

Reference benchmarks?

How is parameter identifiability handled?

Are calibration procedures reproducible?

Can the model overfit reality without warning?

I. Reliability & numerical resilience

How does the system fail?

Loudly?

Quietly?

With partial corruption?

Are solver failures detected or swallowed?

Are results flagged when outside validity bounds?

Are retries ever appropriate—or dangerous?

J. Observability for scientific debugging

Can you trace:

Which assumptions were active?

Which parameters dominated results?

Are intermediate states inspectable?

Is logging scientific or just operational noise?

Can someone reproduce a result six months later?

K. Performance & scaling realism

Does performance degrade gracefully?

Are there hidden O(N²) physics loops?

Is vectorization correct or merely fast?

Is parallelism numerically safe?

L. Data integrity & provenance

Are inputs, outputs, and parameters versioned?

Is provenance tracked (who ran what, with which assumptions)?

Are results cacheable without lying?

Are serialization formats stable?

M. Dependency & environment reproducibility

Can this run on a clean machine deterministically?

Are numerical libraries pinned intentionally?

Are BLAS / solver differences acknowledged?

Is GPU/CPU behavior consistent?

N. Documentation & scientific maintainability

Is there a model overview written for humans?

Are assumptions centralized or tribal knowledge?

Are equations documented inline or referenced?

Is misuse explicitly warned against?

2) Mandatory hard checks (no exceptions)

You must:

Identify the top 3 scientifically complex modules and explain why

Identify top 10 files by scientific risk, not LOC

Trace one major result end-to-end (inputs → equations → numerics → output)

Find ≥10 refactors that reduce scientific error risk

Find ≥10 concrete code smells tied to modeling risk

Identify ≥5 ways the model could produce plausible but wrong results

Identify ≥5 parameter regimes where the model likely fails

Evaluate reproducibility across machines/environments

Evaluate whether tests would catch a sign/unit/frame error

Define a minimum acceptable bar for scientific trust

3) Severity definitions (strict)

Blocker – results are untrustworthy or unsafe

Critical – high risk of silent scientific error

Major – strong erosion of credibility or extensibility

Minor – quality improvement

Nit – consistency only if systemic

4) Tone constraints

Assume bugs until proven otherwise

Prefer falsification over affirmation

State assumptions explicitly

No hand-waving

No “future work” excuses

5) Ideal Target State Blueprint

Describe what excellent looks like:

Scientific architecture

Model/numerics separation

Type system usage

Testing & validation strategy

Reproducibility guarantees

Reviewability by external experts

Long-term extension path

Make it concrete enough that a team could build toward it deliberately.

Final note (for the reviewer)

If you cannot justify trust in the model to another expert, say so plainly.

Silence and politeness are failures.