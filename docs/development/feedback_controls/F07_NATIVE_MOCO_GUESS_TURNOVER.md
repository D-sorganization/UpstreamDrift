# Source-Bound Native Moco Numerical Guess Turnover

Issue #11990 extends the maintained #11968 OpenSim offline matching runner.
`moco-native-guess` consumes an exact SHA-bound source XML and a frozen TRC
clock, then writes `native_initial_guess.sto` and a local JSON receipt. The
source module is `tour_matching/native_moco_guess.py`; `moco_tracking.py`
rejects incomplete state-table names and compares the actual native Moco guess
state/control names with caller-declared bounds. `native_moco_runner.py`
audits the guess against the loaded source and original capture clock before
any solve. No new optimizer, capture parser, control profile or physiology
schema was added.

The preparation uses the actual state after native `initSystem` assembly and
`equilibrateMuscles`, with no modifications to the source model or actuator
policy. It writes a constant **numerical seed**, not measured motion, static
equilibrium, a physiologically accepted initial condition, or a complete
native restart. It never invents state/control bounds or actuator commands.
OpenSim's installed STO writer lost one binary64 ULP in a synthetic Thelen
state; the maintained writer emits 17 significant digits and verifies every
name, value and time through independent native `TimeSeriesTable` readback.
The source identity hashes only the entrypoint XML; missing external geometry
files and inherited resources remain open source-closure questions.

## Native Executed Evidence

OpenSim `4.6-2026-06-22-85aaf64` on the unchanged pinned 520-muscle XML
SHA-256 `a55c64341680551fb5a41be254bdfdb3b2be0ac789336ea44902b22fc9a83913`
loaded as model dump SHA-256
`cdb742504ee35c42cf049239fa0dbaee403f9e304817e1e3f62ece7395593ee6`.
It supplied 1,348 finite named states and 549 actuators; native muscle
equilibration changed 486 continuous values from post-initialization. The
private driver and iron observation clocks were preserved exactly, without
writing raw coordinates or marker identities to the public repository:

| Frozen Reference | Exact Knots |  STO Bytes | STO SHA-256                                                        |
| ---------------- | ----------: | ---------: | ------------------------------------------------------------------ |
| Driver           |         654 | 13,189,558 | `354c694b32e6e046ee67898aed357a554d3c14ddfd014f4a409ae26e7a9ed353` |
| Iron             |         657 | 13,250,183 | `5caff404b582442b1b7ec3c8b804b69e74b40f482802ea2c1af40ccddba4ee9d` |

The owned private staging script `moco_11990_native_guess/verify_real_preparation.py`
reused the original #11968 file-driven request, changing only its absent
guess path and digest. Both actual preparations changed from seven blockers
to six; the removed blocker was exactly `guess-sha256-mismatch`. Capture
registration, marker-binding coverage, passive policy, nonmuscle assistance,
native constraint policy and complete bounded native state remain blocked.
No real-source Moco problem or solve was constructed. The local aggregate
receipt is `moco_11990_native_guess/aggregate-guess-diagnostic.json`; private
STO, TRC and detailed logs remain outside this repository.

The synthetic TDD suite had a missing-module RED followed by native GREEN.
It checks source-byte mutation, non-advancing clock, missing and nonfinite
states, truncated STO clock, source/clock exact readback, actual native
MocoProblem state/control coverage, and the CLI route on both Millard and
Thelen fixtures. Its incomplete-state test first showed that native
`insertStatesTrajectory` silently accepted a partial table; the builder now
rejects that case before it can become a misleading optimization result.
The positive cases use a supplied native fixture model; no fixture shape or
muscle count is hard-coded in production source.

Reproduce a reviewed local seed from the repository root in the qualified
OpenSim environment, using the exact source/TRC and an empty owned output
directory:

```text
python -m src.engines.physics_engines.opensim.python.tour_matching.cli moco-native-guess --model SOURCE.osim --source-sha256 EXPECTED_SHA256 --trc FROZEN.trc --output-dir OWNED_OUT
```

To proceed toward a real match, a reviewer must provide explicit native
marker/body correspondence and registration, physiological state/control
bounds, passive and nonmuscle actuator policies, source constraint/contact
policy, and a complete replay-admissible native state. The private D02
protocol remains exploratory; adjacent frames and paired formats are not
independent trials. Solver success, this numerical seed and the six-blocker
reduction are not evidence of a muscular full-swing match or six-engine parity.
