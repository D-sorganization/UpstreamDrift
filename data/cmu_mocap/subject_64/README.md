# CMU MoCap — Subject 64 (Golf)

Motion-capture data for **Subject #64 (golf)** from the CMU Graphics Lab Motion
Capture Database, trials **01–15** (the swing and putt motions).

- **Source:** https://mocap.cs.cmu.edu/search.php?subjectnumber=64
- **Capture rate:** 120 fps
- **Markers:** ~44 (CMU standard full-body marker set)

## Files

Each trial `64_NN` is provided in three complementary representations:

| Extension | Contents                                       | Notes                                                  |
| --------- | ---------------------------------------------- | ------------------------------------------------------ |
| `.c3d`    | Raw 3D marker trajectories                     | Self-contained; opens in `run_c3d_viewer.py` / `ezc3d` |
| `.amc`    | Per-trial joint angles                         | Solved onto the skeleton below                         |
| `.asf`    | Skeleton definition (bone hierarchy + lengths) | Single shared file: `64.asf`                           |

`.amc` files are only meaningful together with `64.asf`.

## Trial → motion mapping

| Trials          | Motion     |
| --------------- | ---------- |
| `64_01`–`64_10` | Golf swing |
| `64_11`–`64_15` | Putt       |

(Subject 64 on CMU also has trials 16–30 — placing tee / placing ball /
picking up ball — which were intentionally **not** included here.)

## Per-clip summary

| Trial | Motion | Markers | Frames | Duration |
| ----- | ------ | ------- | ------ | -------- |
| 64_01 | swing  | 45      | 448    | 3.7 s    |
| 64_02 | swing  | 44      | 493    | 4.1 s    |
| 64_03 | swing  | 44      | 444    | 3.7 s    |
| 64_04 | swing  | 44      | 412    | 3.4 s    |
| 64_05 | swing  | 44      | 387    | 3.2 s    |
| 64_06 | swing  | 44      | 529    | 4.4 s    |
| 64_07 | swing  | 44      | 376    | 3.1 s    |
| 64_08 | swing  | 44      | 364    | 3.0 s    |
| 64_09 | swing  | 44      | 452    | 3.8 s    |
| 64_10 | swing  | 44      | 562    | 4.7 s    |
| 64_11 | putt   | 44      | 593    | 4.9 s    |
| 64_12 | putt   | 44      | 504    | 4.2 s    |
| 64_13 | putt   | 44      | 437    | 3.6 s    |
| 64_14 | putt   | 44      | 559    | 4.7 s    |
| 64_15 | putt   | 44      | 415    | 3.5 s    |

## Attribution / usage

The data used here was obtained from [mocap.cs.cmu.edu](https://mocap.cs.cmu.edu).
The database was created with funding from NSF EIA-0196217 and is free to use for
research. Please retain this acknowledgment in any derivative datasets or publications.
