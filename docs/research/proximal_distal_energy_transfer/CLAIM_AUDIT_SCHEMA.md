# Scientific Claim Audit Contract

## Purpose

The claim audit is a fail-closed review surface for the scientific paper. It
does not assume that a cited, plotted, tested, or previously reviewed statement
is correct. It creates a complete candidate queue and requires a reviewer to
decide which candidates are material before the audit can be called complete.

The machine-readable authorities are:

- `data/claim_candidate_inventory.json`, the deterministic inventory of
  narrative paragraphs in the Quarto master and all included chapters; and
- `data/claim_audit_registry.json`, the adjudicated claim records, release-claim
  reconciliation, research-collection status, and external dependencies; and
- `data/claim_evidence_manifest.json`, the deterministic coverage and content-
  integrity record for every local and external evidence reference.

## Candidate Inventory

`python -m scripts.research.proximal_distal_energy.claim_audit inventory`
expands the master document in publication order and records every narrative
paragraph with:

- a stable source-path/content/duplicate-ordinal identifier;
- canonical source path and exact line range;
- normalized text and its SHA-256 digest;
- bibliography citation keys;
- numeric-content and assertive-language flags;
- a deterministic review-priority score and transparent triage flags; and
- an initial `unadjudicated` review state.

Display equations and fenced code are excluded, but parsing resumes after their
closing delimiters. Quarto equation labels on display-math closers are supported,
including `$$ {#eq-label}`. The source digest canonicalizes CRLF and CR line
endings to LF so checkout platform does not change scientific content identity.

The inventory is deliberately overinclusive. A question, transition, caption,
or limitation may not be a scientific claim, but it remains visible until a
reviewer marks it non-material. Automated extraction is a queue generator, not
scientific adjudication.

Quarto references such as `@sec-results`, `@fig-speed`, and `@eq-power` are
excluded from `citation_keys`; only bibliography keys are retained. The triage
score raises candidates containing numbers, assertive terms, external
citations, or causal/generalizing language. It prioritizes human review but
never assigns support or materiality.

## Candidate Review

Every inventory candidate receives exactly one review record before completion.
The allowed dispositions are:

- `material_claims_mapped`, with reciprocal links to every atomic claim record
  needed to cover the paragraph;
- `non_material`, for narrative that contains no scientific proposition;
- `editorial_or_navigation`, for headings, signposts, and document mechanics;
  or
- `requires_split`, when the reviewer has not yet completed atomic coverage.

A claim-to-candidate link is valid only when the candidate review links back to
the same claim. This reciprocal rule prevents a paragraph from appearing
adjudicated merely because one of several propositions was registered.

Line ranges are mutable locators rather than identity inputs. Inserting unrelated
text above a paragraph therefore updates its locator without invalidating its
review. Editing the paragraph's normalized content changes its identifier and
requires a new review. Repeated identical paragraphs within one source are
distinguished by their zero-based occurrence order.

## Claim Record

Each material claim receives a stable `PD-CLAIM-*` identifier and records:

| Field                             | Required Meaning                                                                                                           |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `statement`                       | One atomic proposition, narrowed to its declared domain                                                                    |
| `classification`                  | Definition, identity, numerical verification, model result, empirical synthesis, hypothesis, limitation, or interpretation |
| `published_status`                | The status currently exposed by the release                                                                                |
| `audit_status`                    | Current independent-review state; never inferred from the published status                                                 |
| `candidate_ids`                   | Source candidates covered by this adjudication                                                                             |
| `source_locations`                | Human-readable canonical source locators                                                                                   |
| `evidence_artifacts`              | Exact data, code, tests, figures, or original sources used                                                                 |
| `model_domain`                    | Coordinates, constraints, events, parameters, solver, and fidelity boundary                                                |
| `uncertainty_boundary`            | What was varied or measured and what remains outside the interval                                                          |
| `competing_explanations`          | Plausible alternatives that could produce the observation                                                                  |
| `negative_controls`               | Interventions expected to remove or reverse the result                                                                     |
| `falsifier`                       | A result that changes the status to contradicted or inconclusive                                                           |
| `adjudication`                    | Finding-by-finding reasoning and remaining gaps                                                                            |
| `reviewer` and `last_verified_on` | Review provenance                                                                                                          |

One paragraph may contain several material claims and therefore map to several
claim records. One claim may also recur in the abstract, results, discussion,
caption, and conclusion; those locations share one claim identifier only when
their estimand and scope are genuinely identical.

## Evidence Classes

The audit keeps these classes separate:

1. **Definition or mathematical identity:** checked algebraically and against
   stated conventions; it is not empirical support.
2. **Numerical verification:** closure, convergence, or agreement between
   implementations; shared assumptions limit independence.
3. **Model-conditional result:** valid only for the declared equations,
   parameters, event, intervention, and tolerance.
4. **External empirical evidence:** checked against the original source,
   population, measurement method, uncertainty, and correction status.
5. **Project-originated evidence:** useful for reproducibility and hypothesis
   development, but never independent confirmation.
6. **Hypothesis or interpretation:** retained only with a measurable prediction,
   alternative explanation, and prospective falsifier.

## Completion Rule

Run
`python -m scripts.research.proximal_distal_energy.claim_audit validate`.
Validation fails on stale paper bytes, duplicate identifiers, missing required
adjudication fields, missing bibliography keys, non-reciprocal claim mappings,
unresolvable or out-of-range `path:line` source locators, drift from the public
release-claim inventory, or a `complete` status while any candidate remains
unadjudicated or still requires splitting.

Run
`python -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate`
to require exact claim coverage and current SHA-256/byte-size records for every
local support artifact. External URLs are inventoried by exact URL and host but
are not fetched by this deterministic gate. Consequently, a passing manifest
proves that a reviewer can recover the exact registered local bytes and see
every external lead; it does not establish source independence, live-link
availability, empirical adequacy, or scientific correctness.

Completion additionally requires recomputation of every quantitative claim,
figure-data verification, original-source and live-link review, and an
independent finding-by-finding adjudication. Passing the validator proves
contract integrity; it does not by itself prove scientific correctness.

## NotebookLM Boundary

The Biomechanics and Nonlinear Control collections are research indexes. Their
answers can identify leads and omissions, but the audit cites the original
paper, dataset, standard, or other authority after independent verification.
Authentication state and search coverage are recorded without committing
cookies or bearer credentials.
