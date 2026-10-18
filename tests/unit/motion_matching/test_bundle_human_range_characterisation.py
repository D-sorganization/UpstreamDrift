"""Report-only: which bundle coordinates leave ``HUMAN_RANGES_DEG`` (HRG-00, #13161).

Set ``HRG_BEST_FIT_BUNDLE`` to a matching bundle (``.npz`` holding ``spec`` and
``reference_q``) to print, per coordinate, the wrapped minimum and maximum in
degrees against its human range, and the worst excess. The test never fails on
a violation: it exists so that later HRG children can promote it to a hard
gate. Without the variable it is skipped. Private captures stay outside the
repository, so nothing here embeds their values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.range_of_motion import HUMAN_RANGES_DEG

pytestmark = pytest.mark.unit

ENV_VAR = "HRG_BEST_FIT_BUNDLE"


@dataclass(frozen=True)
class RangeRow:
    coordinate: str
    minimum_deg: float
    maximum_deg: float
    low_deg: float
    high_deg: float

    @property
    def excess_deg(self) -> float:
        return max(
            self.low_deg - self.minimum_deg, self.maximum_deg - self.high_deg, 0.0
        )


def characterise(q: np.ndarray, names: list[str]) -> list[RangeRow]:
    """Per-coordinate min/max (degrees, wrapped to +-180) against the ranges.

    Precondition: ``q`` is (frames, len(names)) in radians. Unavailable
    ranges are skipped, never zero-filled.
    """
    values = np.asarray(q, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("q must be (frames, coordinates) matching names")
    rows = []
    for index, name in enumerate(names):
        if name not in HUMAN_RANGES_DEG:
            continue
        deg = (np.degrees(values[:, index]) + 180.0) % 360.0 - 180.0
        low, high = HUMAN_RANGES_DEG[name]
        rows.append(RangeRow(name, float(deg.min()), float(deg.max()), low, high))
    return rows


def test_characterise_flags_only_excess() -> None:
    q = np.radians([[-50.0, 0.0], [30.0, 10.0]])
    rows = characterise(q, ["LWInputX", "LFInput"])
    by_name = {r.coordinate: r for r in rows}
    assert by_name["LWInputX"].excess_deg == pytest.approx(10.0)
    assert by_name["LFInput"].excess_deg == 0.0
    assert characterise(q[:, :1], ["unranged"]) == []
    with pytest.raises(ValueError, match="frames, coordinates"):
        characterise(q, ["LWInputX"])


def test_bundle_human_range_report(capsys: pytest.CaptureFixture[str]) -> None:
    location = os.environ.get(ENV_VAR)
    if not location:
        pytest.skip(f"{ENV_VAR} is not set")
    path = Path(location)
    if not path.is_file():
        pytest.skip(f"{ENV_VAR} does not point to a file")
    with np.load(path, allow_pickle=False) as bundle:
        spec = json.loads(bytes(bundle["spec"]).decode("utf-8"))
        q = np.asarray(bundle["reference_q"], dtype=float)
    names = list(spec["coordinate_order"])
    rows = characterise(q, names)
    over = [r for r in rows if r.excess_deg > 0.0]
    with capsys.disabled():
        print(
            f"\nHRG human-range report: {len(q)} frames, {len(rows)} ranged coordinates"
        )
        for r in rows:
            flag = "EXCEEDS" if r.excess_deg > 0.0 else "ok"
            print(
                f"  {r.coordinate:18s} {r.minimum_deg:8.1f} .. {r.maximum_deg:7.1f}"
                f"   range {r.low_deg:7.1f} .. {r.high_deg:6.1f}   {flag}"
            )
        print(f"  {len(over)} coordinates exceed their range (report only)")
    assert len(names) == q.shape[1]
