# Fleet Synthesis — What the Four Reviews Say Together (2026-10-16)

This page is identical in all four repositories (UpstreamDrift, Tools, Tools_Private, AffineDrift). It is the
"global review of programming approach" that was asked for: not another finding list, but the handful of habits that
produced most of the findings, and the alternative habit for each. Per-repo detail lives in the sibling `README.md`
and `findings/` pages. Numbers here were measured on the clones reviewed (commits in each README).

## 1. The One-Paragraph Read

The mathematics is in better shape than the software around it. Where a reviewer could run a kernel against a
textbook identity — AffineDrift's manipulator dynamics (Ṃ−2C skew to 3e-16, RNEA vs Lagrange to 1e-9), Tools'
rotation conventions (12 Euler sequences agree with SciPy to 9e-16), Tools_Private's JANAF-derived thermochemistry,
UpstreamDrift's integrators — it passed. Where the code converts between _units or conventions_ at a boundary
(kmol/h vs Nm³/h, pure-species vs solution dew point, feed-oxygen vs oxidant oxygen, forward-integrated vs pointwise
counterfactual) it failed, and every one of those failures sailed through a test suite with thousands of tests, because
the suites overwhelmingly assert _shape_ (`> 0`, `is not None`, "the mock was called", "the prose contains the word")
rather than _value_. Meanwhile a governance apparatus of roughly 150k lines (GAAI copies, SPEC change logs, readiness
ledgers, baseline JSONs, nine-script manual pipelines, 137 workflows) produces a strong _feeling_ of rigor while
several of its gates provably cannot fire. The fix is not more gates. It is fewer, sharper ones, aimed at values.

### The Five Findings to Act on This Week

| Repo                  | ID                  | What                                                                                                                             | Why first                                                                                        |
| --------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| UpstreamDrift         | F-UD-001            | `/launcher/logos/{filename:path}` joins an absolute path onto `ASSETS_DIR`; `GET /launcher/logos//etc/hostname` returns the file | Arbitrary file read from a public router; one-line fix (`resolve_contained_path` already exists) |
| Tools                 | F-TL-001            | WGS reactor volume = kmol/h ÷ GHSV; 22.4× undersized; heat duty uses total feed, not CO                                          | A vessel bought from this number is a third of the diameter it needs                             |
| Tools / Tools_Private | F-TL-003 / F-TP-001 | "Acid gas dew point" is pure-species Antoine (HCl → −152 °C, "very low risk")                                                    | Non-conservative for corrosion; the public Tools copy is the one to fix                          |
| Tools_Private         | F-TP-002            | `equivalence_ratio` _replaces_ feed (and steam) oxygen instead of adding oxidant                                                 | 32 % oxygen deficit for wood at φ=3; a test enshrines it                                         |
| AffineDrift           | F-AD-001            | Four flagship figures are hand-parameterised curves labelled "Simulation (Qualified Multi-Body)"                                 | Ends a peer review badly; the real model sits in the next directory                              |

## 2. Habits That Generated Most Findings, and What to Do Instead

### 2.1 Bare Floats Carry Units in Their Name, if at All

`feed_rate / ghsv  # m3` (Tools WGS) is wrong by 22.4× because `feed_rate` is kmol/h. `elements["O"] = stoich_o / phi`
(Tools_Private) silently discards feed oxygen. Lucas viscosity (Tools) mixes three half-remembered formulas with Pc in
Pa, bar and MPa. None of these are "math errors"; they are _type_ errors Python cannot see.

**Instead:** make units a type at every public boundary. Three graduated options, cheapest first:

1. Suffix every public float parameter with its unit (`feed_rate_kmol_h`, `pressure_pa`) and let the test name the
   unit too. Costs nothing; catches the WGS bug at code-review time.
2. `typing.NewType("Pascal", float)` / `Annotated[float, "Pa"]` so mypy flags `Pascal` passed where `Bar` is expected.
3. `pint` quantities at the _edges_ (GUI → engine, file → engine), stripped to plain floats inside hot loops.
   Pick one and apply it to the ~30 numeric entry points downstream repos actually call before touching anything else.

### 2.2 Tests That Cannot Fail on a Wrong Number

Measured texture: UpstreamDrift 923 skip sites, 5,830 mock/patch sites, 49 literal `assert True`; Tools 477 tests
asserting only `is not None`/`> 0`, 400 with no assertion, 175 mock-only; AffineDrift 2,638 prose-presence asserts;
Tools_Private's MMS "verification" asserts convergence order on a fabricated 1/n² error series with no solver run.
A 22× sizing error, a −152 °C dew point and a 32 % oxygen deficit all pass.

**Instead:** for every numeric public function, one _hand-checkable_ golden case with `pytest.approx(ref, rel=1e-6)`
and the reference derivation in the docstring (or a cited table value). Add `hypothesis` property tests for the
invariants you already know: mass/element balance closes, K(T) monotone, energy conserved to tolerance, round-trip
identities. Then run `mutmut` or `cosmic-ray` once on the kernel package: the surviving mutants are exactly the tests
that assert shape. Delete or fix those tests — a test that cannot fail is negative documentation.

### 2.3 Governance by Ratchet File, Where the Ratchet Never Moves

Patterns seen in every repo: a baseline JSON that equals the current state (coverage floor 20 = measured 20),
a size budget whose exception list names files that no longer exist, a test-count floor of 2,282 against 8,088
collected, mypy "baselines" that are 0-byte files, `--check` scripts that only verify someone re-ran `--regenerate`,
a shadow-module ledger whose own comments record that an agent emptied it, closed the issue, and added four new
shadows in the same commit; a CLAUDE.md that announces BLE001/F401/F841 are enforced while `extend-ignore` still lists
F401+F841 (1,372 unused imports hidden) and BLE001 is per-file-ignored for all 617k LOC of `src/shared/python/**`;
19,495 unmarked UpstreamDrift test IDs that no always-on lane runs. Each of these was _built_ to prevent regression and _configured_ so it cannot detect one.

**Instead:** a gate is only real if you can name the last PR it blocked. Audit every CI gate with that question;
delete the ones with no answer (the review pages list candidates). For the survivors, make the ratchet a _target_
that moves on a schedule (coverage floor +2 % per month, written as a dated table in the workflow) rather than a
baseline that is regenerated to match reality. Prefer three gates that bite to thirty that are green.

### 2.4 Documentation That Describes the Intended System

AGENTS.md in three repos documents ten Jules overnight workflows with a timetable; none of the ten exist in any of the
four repos. CLAUDE.md in Tools_Private describes "15 proprietary calculators"; seven are PyQt shells importing their
physics from public Tools. CLAUDE.md in AffineDrift promises interactive code cells; the corpus has zero executable
cells. UpstreamDrift's CLAUDE.md says Tools is the single source of truth for `ai/chat/sidekick`; UpstreamDrift carries
a 511-file fork of it, 160 files diverged. Agents read these files first and act on them — so stale instructions are
not harmless, they are a prompt-injection you wrote yourself.

**Instead:** keep CLAUDE.md/AGENTS.md to what is _mechanically true_: commands that run, paths that exist, rules a
hook enforces. Everything aspirational moves to an ADR with a status field. Add one CI check that every path and
workflow name mentioned in CLAUDE.md/AGENTS.md exists — that check would have caught all of the above.

### 2.5 Shared Code Shared by Copying

Byte-identical files between Tools and UpstreamDrift: 337 (62k LOC). Fleet scripts (`automerge_guard.py`,
`changes_fragment_*.py`, `requeue_stalled_merges.py`, …) exist 3–4× across repos, "synced" by copy; 5 of the 16
"synced" AGENTS.md sections already differ. The GAAI framework is vendored into three repos (77k LOC total). Tools
already publishes a versioned wheel (`ud_tools-1.15.0`) — the mechanism that makes all of this unnecessary exists and
is not used by the main consumer.

**Instead:** one direction of dependency, enforced by packaging, never by ledger. UpstreamDrift `pip install`s the
Tools wheel at a pinned version (a `requirements.lock` line, not a submodule and not a fork). Fleet scripts become a
tiny `fleet-tools` package installed the same way. A "divergence inventory" of 3,892 content hashes is what you
build when you have decided not to do this; do this instead and delete the inventory.

### 2.6 The Repository as a Filing Cabinet

UpstreamDrift is 901 MB tracked, 541 MB of it under `docs/` (CSV sweeps, `.mat` tables, GIFs, 29k LOC of Python
_inside docs/_). AffineDrift commits 371 shell exit-code files and 705 stdout logs (11 MB of session transcripts)
plus 31 MB of PDFs behind a dead LFS rule. Every agent clone, every CI job and every reviewer pays for this, and
`git blame`/`grep` get slower and noisier for everyone.

**Instead:** code in `src/`, prose in `docs/`, data in a data repo or release asset (or LFS that is actually
configured), evidence as a _hash + URL_ in a JSON ledger rather than the artifact itself. A one-off history rewrite
is not needed; stop adding, move the largest 20 files to releases, and add a pre-commit size hook (1 MB) so the
problem stops growing.

### 2.7 Two Implementations of One Quantity, One of Them Decorative

UpstreamDrift's `BallFlightSimulator` integrates in Rust (Reynolds-dependent drag, decaying spin) and then recomputes
every exported `acceleration`/`forces` in Python with a _different_ drag model and constant spin — so the exported
acceleration is not the derivative of the exported velocity (8–16 % off). Its MuJoCo→Pinocchio wrapper reorders the
quaternion but hands MuJoCo's world-frame free-flyer velocity to Pinocchio, which expects body frame — a mismatch the
repo's own `pose_interchange` adapters document. Eleven quaternion→matrix and seventeen RK4 copies exist in the same
repo while the DRY gate blesses 5,604 duplicate fingerprints.

**Instead:** one kernel, one owner, and a _parity test that compares numbers_: `assert np.allclose(python(x), rust(x),
rtol=1e-9)` on randomized inputs, run in CI. The Rust "numeric parity" test currently asserts `is not None`.

### 2.8 Convention Splits That Look Like Vocabulary

AffineDrift's manuscript defines ZTCF as a _pointwise_ drift evaluation; the repo's governed contract type admits only
`construction: "forward"`, the explorer widget forward-integrates, and NOTATION.md defines a four-member family.
Both constructions are legitimate objects — but they are different objects with the same name, and a journal
reviewer who follows the supplementary link will find the one the paper says it is not. The same shape appears in
Tools_Private (`equivalence_ratio` as fuel-φ replacing oxygen vs oxidant-ER adding to it).

**Instead:** when a term has two constructions, give them two names in code _and_ a one-page ADR that says which one
the publication means, with the equation. `ztcf_pointwise()` and `ztcf_forward()` can coexist; `ztcf()` cannot.

### 2.9 Figures That Are Not Computed

Four flagship AffineDrift figures are hand-parameterised curves (`48.2/(1+exp(−25(t−0.19)))`, `18t³+120t⁴`) under an
evidence label of "Simulation (Qualified Multi-Body)", beside a working `GolfModel` that could have produced them.
This is the finding most likely to end a peer review badly, and the cheapest to fix.

**Instead:** every figure script imports the model and the figure test asserts the plotted array equals the model
output. If a figure is a sketch, say "schematic" in the caption and the evidence-level field. Make "no figure
without a data provenance line" a content gate — it is one of the few gates that would earn its place.

## 3. What to Keep Doing (Seen Across the Fleet)

- Writing the physics as explicit Lagrangian/RNEA/Christoffel forms and checking identities — this is why the kernels pass.
- DbC at public boundaries (`require()`, narrow exception helpers) — the _idea_ is right; the enforcement needs to catch up
  (15 % adoption in Tools_Private; `require` never raises `TypeError`).
- Separating `core.py` from `gui.py` so math is testable headless.
- Fail-closed companion/manifest verification (AffineDrift's SHA-256 consumer, Tools' p1am auth) — model other I/O on these.
- Ruff + pre-commit — once the two repos that currently fail their own `ruff check .` are fixed, keep it exactly that strict.

## 4. A Suggested Order of Operations

1. **Numbers first (days):** fix the five value bugs that cost money or credibility — Tools WGS sizing, Lucas viscosity,
   acid-gas dew point model (public Tools; Tools_Private inherits), Tools_Private feed-oxygen handling and
   unconverged-Gibbs masking, AffineDrift figures. Each has a reproduction and a golden test in its finding page.
2. **Vocabulary (a week):** split ZTCF forward/pointwise in code and NOTATION.md; decide φ vs ER; write the two ADRs.
3. **Gates (a week):** delete gates that cannot fire; make three real ones (value goldens on numeric entry points,
   a CLAUDE.md path-existence check, a 1 MB file-size hook).
4. **Sharing (a month):** UpstreamDrift consumes the Tools wheel; delete the fork and the divergence apparatus; fleet
   scripts become a package.
5. **Weight (ongoing):** move data out of `docs/`, stop committing logs/exit codes, and let the repos get small enough
   that an agent can clone one in under a minute.
