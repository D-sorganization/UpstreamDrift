# FreeMoCap sidecar pipeline

UpstreamDrift uses [FreeMoCap](https://freemocap.org/) for markerless 3D
motion capture from multiple-camera video. FreeMoCap is licensed AGPL,
so we wrap it as an **out-of-process subprocess** to keep our MIT-licensed
codebase from importing AGPL code at runtime.

## Architecture

```
+--------------------+         subprocess          +-------------------+
|  UpstreamDrift     |  -------------------------> |  FreeMoCap venv   |
|  (MIT licensed)    |    python -m freemocap      |  (AGPL licensed)  |
|                    |                             |                   |
|  src/tools/        |                             |  ~/.venvs/        |
|    freemocap_      |                             |    freemocap/     |
|    sidecar/        |                             |    bin/python     |
+--------------------+                             +-------------------+
        |                                                    |
        |    landmarks.csv  +  metadata.json (file system)   |
        |  <-------------------------------------------------|
        v
+--------------------+
|  motion_pipeline   |
|  reads outputs via |
|  stdlib only       |
+--------------------+
```

No symbol from `freemocap` is ever imported in our process. AGPL code stays
entirely on the other side of the subprocess boundary.

## Usage

### From the CLI

```bash
# Dry-run (no freemocap install needed) — writes stub artifacts
python3 -m src.tools.freemocap_sidecar.run_freemocap \
    --input ./session_recordings \
    --output ./landmarks_out \
    --dry-run

# Real run — points at the freemocap venv's python
python3 -m src.tools.freemocap_sidecar.run_freemocap \
    --input ./session_recordings \
    --output ./landmarks_out \
    --env-python ~/.venvs/freemocap/bin/python
```

### Programmatically

```python
from src.tools.freemocap_sidecar import run_freemocap_sidecar

result = run_freemocap_sidecar(
    input_dir="./session_recordings",
    output_dir="./landmarks_out",
    freemocap_env_python="~/.venvs/freemocap/bin/python",
)

if result.success:
    print(f"Wrote landmarks to {result.landmarks_csv}")
else:
    print(f"FreeMoCap failed (rc={result.return_code}): {result.stderr_tail}")
```

## Output contract

After a successful run the output directory contains at minimum:

- **`landmarks.csv`** — frame-by-frame 3D landmark positions:
  ```
  frame,landmark_id,x,y,z
  0,0,0.123,-0.045,1.677
  0,1,...
  ```

- **`metadata.json`** — session metadata:
  ```json
  {
      "freemocap_version": "1.6.0",
      "n_frames": 1234,
      "n_landmarks": 33,
      "fps": 60,
      "duration_s": 20.5
  }
  ```

When freemocap is unavailable (no env, not installed, dry-run), the
sidecar still writes minimum stub artifacts following the same shape so
downstream consumers can integration-test without needing the real
install. Stub metadata carries `"stub": true` and
`"freemocap_version": "stub"`.

## Failure modes

`run_freemocap_sidecar()` returns a `FreeMoCapResult` rather than
raising on subprocess failure. Inspect:

| Field | Meaning |
|---|---|
| `success` | True iff subprocess exited 0 AND output files exist |
| `used_real_freemocap` | True if real freemocap was invoked, False if stub |
| `return_code` | subprocess exit code (-1 = timeout, 127 = interpreter missing) |
| `stderr_tail` | last 4 KB of subprocess stderr |
| `landmarks_csv`, `metadata_json` | absolute paths if produced, else None |

Common cases:

- **`success=True, used_real_freemocap=True`** — real run worked.
- **`success=True, used_real_freemocap=False`** — dry-run or fallback;
  stub artifacts written.
- **`success=False, used_real_freemocap=False, return_code=127`** —
  user passed a `freemocap_env_python` that does not exist; stub written.
- **`success=False, return_code=-1`** — subprocess timeout.

## Setting up the freemocap venv

```bash
# One-time install in an isolated environment
python3 -m venv ~/.venvs/freemocap
~/.venvs/freemocap/bin/pip install --upgrade pip
~/.venvs/freemocap/bin/pip install freemocap

# Then point the sidecar at this venv's python
export FREEMOCAP_PY=~/.venvs/freemocap/bin/python
```

## Why subprocess and not just `import freemocap`?

1. **License hygiene.** AGPL contamination of our MIT codebase would
   force the rest of UpstreamDrift to be AGPL-distributed too. Keeping
   freemocap on the other side of an OS-process boundary keeps the
   licenses cleanly separated.

2. **Dependency isolation.** FreeMoCap pulls in heavy ML stacks
   (mediapipe, OpenCV, etc.) that conflict with versions used elsewhere
   in UpstreamDrift. The isolated venv keeps the dependency graph
   untangled.

3. **Crash isolation.** A crash inside freemocap (it does have lengthy
   numerical pipelines that can OOM or segfault) does not take down
   UpstreamDrift's process.

## Tests

The sidecar has full unit-test coverage in
`tests/unit/tools/freemocap_sidecar/test_run_freemocap.py`:

- Dry-run path produces stub artifacts.
- Missing interpreter → graceful 127 with stub fallback.
- Missing freemocap module in subprocess → graceful fallback with stub.
- Genuine subprocess failure → reported, no stub.
- Subprocess timeout → reported.
- Real success path with outputs → reported.
- Real success path without outputs → reported as failure.
- CLI entrypoint dry-run and failure paths.

Run them with:

```bash
python3 -m pytest tests/unit/tools/freemocap_sidecar/ -v
```

## Future work

1. **motion_pipeline adapter** — a `FreeMoCapSource` adapter under
   `src/shared/python/motion_pipeline/sources/` that consumes the
   `landmarks.csv` / `metadata.json` outputs and emits the canonical
   `BodyTarget` schema. Tracked as a follow-up.

2. **Multi-camera calibration helper** — a small CLI that wraps
   freemocap's calibration step so users don't have to run multiple
   subprocesses by hand.

3. **GUI tile** — a launcher tile in the desktop UI that lets users
   point at a session directory and run the sidecar without dropping to
   the command line.
