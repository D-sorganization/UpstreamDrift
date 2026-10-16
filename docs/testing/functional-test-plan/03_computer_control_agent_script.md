## Computer-Control Agent Script

This section is the executable contract. An agent that follows it produces the
same evidence set every run, with no fixed sleeps and no human judgement calls.

### 0. Run Workspace and Environment

```bash
export RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
export EVIDENCE=scratch/functional-qa-$RUN_ID
mkdir -p $EVIDENCE/{screenshots,logs,api,surface}
git rev-parse HEAD > $EVIDENCE/commit.txt
python3 -V > $EVIDENCE/env.txt; uname -a >> $EVIDENCE/env.txt 2>/dev/null
export UD_TMP=$EVIDENCE/tmp && mkdir -p $UD_TMP   # the only writable output path
```

All product output goes to `$UD_TMP`. Nothing outside `$EVIDENCE` is written.

### 1. Launch Commands (Canonical — Do Not Improvise)

| Surface             | Command                                                                              | Ready signal                                                                         | Timeout |
| ------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ | ------- |
| Native PyQt6        | `python3 launch_upstream_drift.py --classic`                                         | A top-level window titled `UpstreamDrift` is enumerable via the accessibility bridge | 90 s    |
| Backend only        | `python3 launch_upstream_drift.py --api-only`                                        | `GET /api/health` → 200 **and** `GET /readyz` → 200                                  | 120 s   |
| Web dev shell       | `npm --prefix ui run dev`                                                            | `GET http://localhost:<vite-port>/` → 200 and `/api/launcher/manifest` proxies → 200 | 120 s   |
| Surface enumeration | `QT_QPA_PLATFORM=offscreen python3 scripts/testing/enumerate_test_surface.py --json` | process exit 0                                                                       | 120 s   |

Redirect every process: `> $EVIDENCE/logs/<name>.stdout.log 2> $EVIDENCE/logs/<name>.stderr.log`.

### 2. Readiness Detection — Never `sleep N`

Poll on a **condition**, at 1 s intervals, up to the row's timeout. On timeout
the row is `BLOCKED` (environment) only if the ready signal never appeared _and_
no error was logged; if the process exited or logged a traceback it is `FAIL`.

| Thing being waited for  | Poll                                                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Backend up              | `curl -fsS $API/api/health` then `curl -fsS $API/readyz`                                                                                    |
| Vite up                 | `curl -fsS $VITE/` returns 200                                                                                                              |
| Native window up        | accessibility enumeration returns a window whose title matches the expected name                                                            |
| Native tool launched    | launcher status text changes to `* <Tool> Running` **and** a new child PID appears under the launcher                                       |
| Tool window closed      | the window handle is gone **and** its child PID has exited                                                                                  |
| Web page settled        | the route's primary landmark is in the accessibility tree **and** the network request list has no in-flight request for 2 consecutive polls |
| Long operation finished | the operation's own progress/terminal-state indicator, never elapsed time                                                                   |

If a required poll cannot be expressed for a row, the row is `NOT RUN` with the
reason `no deterministic ready signal` — that itself is a testability defect to
file against this plan.

### 3. Evidence Capture (Mandatory per Row)

Write one JSON line per attempted row to `$EVIDENCE/results.jsonl`:

```json
{
  "id": "NAT-34",
  "result": "FAIL",
  "class": "A",
  "started": "2026-07-25T18:04:11Z",
  "duration_s": 22.4,
  "commit": "c523c07b5",
  "screenshots": [
    "screenshots/NAT-34-before.png",
    "screenshots/NAT-34-after.png"
  ],
  "logs": ["logs/classic-launcher.stderr.log"],
  "observed": "tile double-click produced no window; launcher status unchanged",
  "expected": "pendulum simulator window within 30 s",
  "issue": 8065
}
```

Rules:

- Screenshot **before** the state-changing action and **after** the ready
  signal (or after the timeout). Name them `<ID>-before.png` / `<ID>-after.png`.
- Capture the process-table snapshot before and after each tile row; the diff
  is `REL-09` evidence.
- For `API-*` rows, save the raw response to `$EVIDENCE/api/<ID>.json` with the
  status code — the response body **is** the evidence.
- Never paste log contents into an issue when a path will do; always record the
  path.

### 4. Deterministic Verdicts

Evaluate in order; the first matching rule wins.

| Verdict     | Rule                                                                                                                                                                                                                  |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FAIL**    | Launcher/child process exited unexpectedly, a traceback or crash modal appeared, an assertion of the row's class was not met, or a success indicator appeared without the corresponding state change (false success). |
| **BLOCKED** | The row's precondition is an optional dependency the product's own probe (`PF-04`/`API-14`) reports absent, **or** a needed sibling model-pack repo is absent per `PF-07`. Requires the probe output as evidence.     |
| **NOT RUN** | The control surface could not target the element (offscreen card, owned file dialog not exposed by the accessibility bridge), or no deterministic ready signal exists. **Never** record this as PASS or FAIL.         |
| **PASS**    | Every clause of the row's class assertion was observed, with before/after evidence.                                                                                                                                   |

A `NOT RUN` caused by a control-surface limitation must never be used to close a
defect issue. Record the limitation verbatim in the execution record.

### 5. Execution Order

1. `PF-06` … `PF-09` (surface enumeration and environment contract) — a failure
   here invalidates coverage claims and stops the campaign.
2. `SH-08` … `SH-13` — static inventory integrity. These are cheap, fully
   automatable, and historically found `#8086`, `#8087`, `#8089`–`#8092`.
3. `PF-01` … `PF-05`, then `QT-01`, `REL-07`.
4. `QT-13` … `QT-34` (menu surface) — cheap, high defect yield.
5. `NAT-*` class A tiles, then class B, then `PRV-*`.
6. `QT-02` … `QT-12`, `REL-01` … `REL-11`.
7. `API-01` … `API-19`.
8. `WEB-*` only after the equivalent native evidence exists, then `PAR-*`.

Within each block, run P0 before P1 before P2.

### 6. Defect Filing

- Reproduce once from a clean process state before filing, unless the defect is
  a crash or risks data loss.
- One issue per independently actionable defect, titled
  `[Functional QA][<matrix ID>] <observed failure>`, linked to the campaign epic.
- Before filing, grep the defect verification ledger below — if the row already
  has an open issue, add an observation comment instead of a duplicate.
- Do not install software, change permissions, use production or personal data,
  or send any communication outside the authorized GitHub issues.

### 7. Run Closeout

Emit `$EVIDENCE/summary.md` containing: totals by verdict and priority, P0 pass
rate, every filed issue link, every `BLOCKED` dependency with its probe output,
every `NOT RUN` with its reason, and the exact next matrix slice. Append one row
to the execution record in this document.

## GitHub Issue Templates

### Epic

`[Epic] Functional product testing — <release/build>`

Body sections: scope and build, acceptance criteria, matrix document link,
execution checklist grouped by matrix area, confirmed-defect checklist,
blocked dependency checklist, and sign-off criteria.

### Defect

`[Functional QA][<matrix ID>] <concise observed failure>`

Body sections: environment/build, matrix ID, preconditions/fixture,
reproduction steps, expected result, actual result, evidence (screenshots/log
path), reproducibility, impact/severity, and `Parent epic: #<number>`.
