# Synthetic Adapter Golden Runtime Reference

## Scope and Calculation

This reference records a recovered **synthetic adapter test receipt**. It does not
represent another golfer simulation, measured-player result, or physical model
qualification. The test injects a deterministic synthetic physics implementation
and six delivery rows per pattern. Draw, Straight, and Fade each have four
supported rows and two unsupported rows. The reported support fraction is
`4 / 6`; `matched_runner.py` obtains its two-sided 95% Clopper–Pearson interval
with `scipy.stats.binomtest(4, 6).proportion_ci(confidence_level=0.95)`. In
beta-quantile notation, the endpoints are
`BetaInv(0.025; 4, 3)` and `BetaInv(0.975; 5, 2)`.

The complete JSON receipt covers core dispatch, matched populations, callbacks,
and delivery adapters. The historical golden
[`adapter_refactor_reference.json`](../../../../tests/tools/shot_pattern_analysis/fixtures/adapter_refactor_reference.json)
remains byte-for-byte unchanged at SHA-256
`0403629f476ae9f01b50afe9060fd2e6d0b41b4f463aad213fec032c6af245fa`.
The separate [SciPy 1.15.3 receipt](../../../../tests/tools/shot_pattern_analysis/fixtures/adapter_refactor_reference_scipy_1_15_3.json)
is bound to SHA-256
`2340fb22609bc17eac72ccd4f621bbe187edeff9be0bd26d062f89b1bce90d93`.
The independently recovered [Python 3.11 receipt](../../../../tests/tools/shot_pattern_analysis/fixtures/adapter_refactor_reference_py311_scipy_1_15_3.json)
is bound to SHA-256
`8922892320bea6ea9f10bedb0f6793fcd3d972d88149ffa48650cde06e8219fe`.
Its entire scientific receipt and 65-module source map equal the Python 3.12
SciPy 1.15.3 receipt; its provenance records the distinct interpreter run.
The SciPy 1.15.3 and 1.18.1 scientific receipts differ only at the six
interval endpoints, two for each pattern. At SciPy 1.15.3 they are
`[0.22277809550354957, 0.9567281317071105]`; at SciPy 1.18.1 they are
`[0.22277809550351219, 0.9567281317072583]`. All other receipt fields
match exactly. The differences are approximately `10^-13` in numerical
interval evaluation, with identical input rows and support counts.

## Recovered Producer and Environment

The complete receipt was reproduced from historical commit
`de9ece2b99121e6f0c5a18f85e9331012d80a7b0` (tree
`d0fca9395abdaca6d1075ba5ce2343ed0b361ba7`) using the pre-change
`test_adapter_refactor.py` harness from `45848d30a4`, SHA-256
`53a3ba0ef97e120afa778d0eb8e1e38849d8264e2f1bbcdbf1bd18d584fa6d4e`.
The historical fixture's four source-file hashes match both reproductions.
The SciPy 1.15.3 receipt records SHA-256 identities for all 65 loaded
historical source modules, the commit and tree, the harness, and the Python,
NumPy, and SciPy versions. The 65-module map is identical in the SciPy 1.18.1
reproduction. No current source module was substituted into that producer run.

The reproductions used Python 3.11.15 and 3.12.13 with NumPy 2.2.6. The
original full fixture matched exactly with SciPy 1.18.1 on Python 3.12.13.
The CI-lock full fixture matched exactly with SciPy 1.15.3 on both Python
releases. The test selects only the reviewed Python 3.11 or 3.12 minor release,
NumPy 2.2.6, and explicit SciPy version combinations. A Python patch release
update still has to match the full fixture exactly. The test fails for an
unreviewed combination; it checks the selected fixture hash and compares the
entire receipt as JSON without a numerical tolerance.

## Reproduction and Limits

To run the current adapter gate from the repository root with one of the
reviewed environments:

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/tools/shot_pattern_analysis/test_adapter_refactor.py
sha256sum tests/tools/shot_pattern_analysis/fixtures/adapter_refactor_reference*.json
```

For SciPy 1.15.3, the following command recreates the synthetic receipt from
the historical producer and exact pre-change harness. Run it from the current
repository root in an environment with Python 3.11 or 3.12, NumPy 2.2.6,
SciPy 1.15.3, pytest, and the pinned Tools package:

```bash
git worktree add --detach /tmp/adapter-golden-source de9ece2b99121e6f0c5a18f85e9331012d80a7b0
mkdir -p /tmp/adapter-golden-harness/fixtures
git show 45848d30a4:tests/tools/shot_pattern_analysis/test_adapter_refactor.py > /tmp/adapter-golden-harness/test_adapter_refactor.py
cp tests/tools/shot_pattern_analysis/fixtures/adapter_refactor_reference.json /tmp/adapter-golden-harness/fixtures/
UD_ADAPTER_REPO="$PWD" PYTHONPATH="$PWD/vendor/ud-tools/src" python - <<'PY'
import hashlib, json, os, pathlib, runpy, sys
import numpy, scipy
assert numpy.__version__ == "2.2.6" and scipy.__version__ == "1.15.3"
repo = pathlib.Path(os.environ["UD_ADAPTER_REPO"])
source = pathlib.Path("/tmp/adapter-golden-source")
harness = pathlib.Path("/tmp/adapter-golden-harness/test_adapter_refactor.py")
assert hashlib.sha256(harness.read_bytes()).hexdigest() == "53a3ba0ef97e120afa778d0eb8e1e38849d8264e2f1bbcdbf1bd18d584fa6d4e"
sys.path.insert(0, str(source))
os.chdir(source)
receipt = runpy.run_path(str(harness))["_adapter_receipt"]()
minor = ".".join(str(sys.version_info[i]) for i in (0, 1))
name = "adapter_refactor_reference_py311_scipy_1_15_3.json" if minor == "3.11" else "adapter_refactor_reference_scipy_1_15_3.json"
expected = json.loads((repo / "tests/tools/shot_pattern_analysis/fixtures" / name).read_text())
loaded = {}
for module in list(sys.modules.values()):
    file = getattr(module, "__file__", None)
    if file:
        path = pathlib.Path(file).resolve()
        if path.is_relative_to(source) and path.is_file():
            loaded[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
assert loaded == expected["provenance"]["loaded_historical_source_sha256"]
assert json.dumps(receipt) == json.dumps(expected["receipt"])
print("Full historical-source adapter receipt and 65 loaded module hashes match")
PY
```

The fixture's `provenance.source_sha256` and each recovered receipt's
`provenance.loaded_historical_source_sha256` retain the exact input identities.
This recovery verifies a synthetic regression fixture under two dependency
versions. It does not rerun the phase-two shot populations or renew scientific
acceptance of the published study.
