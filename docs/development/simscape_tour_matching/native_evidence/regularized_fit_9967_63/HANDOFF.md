# Expanded Sextic Trial 63 — Terminal Numerical Gate Failure

Run63 terminated with exit 1 after 77.672 s and three forward evaluations:
`Sensitivity primal marker replay exceeds numerical agreement bound`.
This is an intentionally preserved gate failure, not an optimizer return or
an accepted improvement. No workaround, tolerance change or relaunch occurred.

Original runtime61 and baseline candidate19 were retained, with explicit
restart62. The only authorized changes were `--shaping bernstein456` and
max-nfev 5; analytic Jacobian, penalty weight 0.01, bounds and scales remained
unchanged. Thus B4/B5/B6 provide 81 variables while retaining one global
degree-six polynomial and the original initial state. The restart contract
passed. The first two sensitivity/primal marker audits passed with maximum
differences 4.59e-8 and 9.59e-10 m. The third audit raised at runtime61
`native_sensitivity.py:227`; its numerical discrepancy value was not returned
by that exception and must not be invented from the earlier passing records.

The failing third candidate's preceding forward metrics were whole RMS
28.614 mm, early 10.824 mm, terminal 66.637 mm and terminal club RMS 31.056 mm.
These do not qualify the failed derivative or justify advancing that candidate.
Its canonical hash is
`e057a7e977cfe2a163fe4c5e7e5767c17d5025b4ef7f33166672afbf658c358d`.
`best-unqualified.json` is only a checkpoint. No returned-candidate file exists.
Run62 remains the latest cleanly returned exploratory candidate, itself still
rejected by fit acceptance.

## Reproduction and Next Decision

`launch.json` contains exact command/environment, source/input hashes, original
PID and terminal status. `summary.json` preserves three forward records and two
successful Jacobian records. `raw-run.zip` contains every remote output, exact
stderr traceback, stdout, inputs and immutable runtime61 source archive.
Archive and every output hash were verified after download; the archive hash
is stored in `summary.json`. Force-add the ZIP when committing.

The next diagnostic should reproduce the failing candidate's primal consistency
comparison with explicit measured discrepancy and solver settings before
deciding whether derivative accuracy, propagation sensitivity or replay
tolerances cause the mismatch. Do not bypass the gate or infer that larger
polynomial freedom is ineffective from this aborted trial. Additional fitting
or numerical changes require a separate recorded experiment. No fitting job
from this assignment remains active; no production edits or commits were made.
