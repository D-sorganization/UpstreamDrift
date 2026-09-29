#!/usr/bin/env python3
"""Generate SciPy golden vectors for `tests/parity_filter.rs` (issue #7661).

The `upstream-mocap-preproc` filter kernels claim machine-precision parity with
SciPy's `butter`/`lfilter`/`lfilter_zi`/`medfilt`/`gaussian_filter1d`. This
script emits the reference outputs so the Rust tests can assert that parity
against goldens, mirroring the `upstream-muscle/tests/parity_hill.csv` pattern.

Requires SciPy + NumPy. Regenerate from this directory with:

    python3 generate_parity_filter.py

Coverage:
  * butter(order, Wn, 'low')            -> b,a coefficient parity
  * lfilter_zi(b, a)                    -> steady-state warm-start parity
  * lfilter(b, a, x)                    -> forward IIR parity (exercised via the
                                          filtfilt golden, which composes
                                          lfilter + lfilter_zi)
  * filtfilt(b, a, x)                   -> zero-phase fwd/bwd parity
  * medfilt(x, k)                       -> median-filter parity
  * gaussian_filter1d(x, sigma)         -> Gaussian smoothing parity
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, signal


def fmt(values) -> list[str]:
    return [repr(float(v)) for v in np.atleast_1d(values)]


def main() -> int:
    here = Path(__file__).resolve().parent
    rng = np.random.default_rng(7661)

    # A representative noisy mocap-like signal sampled at 100 Hz.
    n = 64
    t = np.arange(n) / 100.0
    x = (
        np.sin(2 * np.pi * 1.5 * t)
        + 0.3 * np.sin(2 * np.pi * 8.0 * t)
        + 0.05 * rng.standard_normal(n)
    )

    rows: list[list] = []

    # ── butter coefficients + lfilter_zi (warm start) ─────────────────────────
    for order, wn in [(2, 0.2), (4, 0.1), (4, 0.35), (6, 0.25)]:
        b, a = signal.butter(order, wn, btype="low")
        rows.append(["butter_b", order, repr(float(wn))] + fmt(b))
        rows.append(["butter_a", order, repr(float(wn))] + fmt(a))
        zi = signal.lfilter_zi(b, a)
        rows.append(["lfilter_zi", order, repr(float(wn))] + fmt(zi))

    # ── filtfilt (composes lfilter + lfilter_zi warm-start) ───────────────────
    for order, wn in [(2, 0.2), (4, 0.1)]:
        b, a = signal.butter(order, wn, btype="low")
        y = signal.filtfilt(b, a, x)
        rows.append(["filtfilt_input", order, repr(float(wn))] + fmt(x))
        rows.append(["filtfilt_output", order, repr(float(wn))] + fmt(y))

    # ── lfilter (forward only) ────────────────────────────────────────────────
    b, a = signal.butter(3, 0.25, btype="low")
    y = signal.lfilter(b, a, x)
    rows.append(["lfilter_input", 3, "0.25"] + fmt(x))
    rows.append(["lfilter_output", 3, "0.25"] + fmt(y))
    # The b,a used for the lfilter case, so the Rust side can reproduce it.
    rows.append(["lfilter_b", 3, "0.25"] + fmt(b))
    rows.append(["lfilter_a", 3, "0.25"] + fmt(a))

    # ── medfilt ───────────────────────────────────────────────────────────────
    for k in [3, 5, 7]:
        y = signal.medfilt(x, kernel_size=k)
        rows.append(["medfilt_input", k, "0"] + fmt(x))
        rows.append(["medfilt_output", k, "0"] + fmt(y))

    # ── gaussian_filter1d ─────────────────────────────────────────────────────
    for sigma in [1.0, 2.0, 3.5]:
        y = ndimage.gaussian_filter1d(x, sigma)
        rows.append(["gaussian_input", 0, repr(float(sigma))] + fmt(x))
        rows.append(["gaussian_output", 0, repr(float(sigma))] + fmt(y))

    out_path = here / "parity_filter.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "int_param", "float_param", "values..."])
        w.writerows(rows)
    sys.stderr.write(f"wrote {out_path} ({len(rows)} rows)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
