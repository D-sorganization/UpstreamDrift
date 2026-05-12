"""Generate the Hill-curve parity fixture for `tests/parity_hill.rs`.

Runs the Python source-of-truth (`HillMuscleModel.force_length_active`,
`force_velocity`, `tendon_force`) on a 100-point sweep per curve and writes
``tests/parity_hill.csv`` as ``(curve, input, expected_output)`` rows.

Usage::

    cd <repo-root>
    python rust_core/upstream-muscle/scripts/generate_parity_fixture.py

Re-run whenever the Python source's curve definitions change.

Slice 1 of UD#5216.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.biomechanics.hill_muscle import (  # noqa: E402
    HillMuscleModel,
    MuscleParameters,
)


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n < 2:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def main() -> int:
    # The model's curves are independent of the parameters once you pass
    # in normalized inputs, so any non-degenerate parameter set works.
    muscle = HillMuscleModel(
        MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20, v_max=10.0)
    )

    n = 100
    rows: list[tuple[str, float, float]] = []

    # f_l: sweep normalized length around the optimum.
    for x in _linspace(0.20, 1.80, n):
        rows.append(("f_l", x, muscle.force_length_active(x)))

    # f_v: sweep normalized velocity across both branches, including past
    # the -0.99 clamp on the concentric side.
    for x in _linspace(-1.50, 1.50, n):
        rows.append(("f_v", x, muscle.force_velocity(x)))

    # f_t: sweep normalized tendon length around slack.
    for x in _linspace(0.50, 1.50, n):
        rows.append(("f_t", x, muscle.tendon_force(x)))

    out = Path(__file__).resolve().parent.parent / "tests" / "parity_hill.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["curve", "input", "expected"])
        for curve, x, y in rows:
            writer.writerow([curve, repr(x), repr(y)])

    sys.stdout.write(f"wrote {len(rows)} rows to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
