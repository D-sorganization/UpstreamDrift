"""Engine-neutral container for one recorded native swing run (#12328).

Both MuJoCo and Drake hand their native output over as arrays of world-frame
clubhead positions (``mj_data.site_xpos`` / a Drake frame-pose history).
Adapters only re-shape these; a speed the engine did not supply is derived by
differencing the engine's own positions, never by re-simulating.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..adapters_base import QuantityData

POSITION_DIM = 3


@dataclass(frozen=True)
class NativeSwingRun:
    """One engine's recorded swing, in SI units and a Z-up world frame.

    Attributes:
        time_s: strictly increasing sample times.
        clubhead_position_m: ``(len(time_s), 3)`` world positions; a non-finite
            entry marks a missing sample.
        clubhead_speed_mps: optional engine-supplied speed, same length as
            ``time_s``; derived from positions when omitted.
        kinematic_sequence: optional pre-computed advertised series.
        engine_version / model_file_hash: recorded provenance, if known.
        run_class / force_provenance: mirror the schema ``EngineRun`` enums.
    """

    time_s: tuple[float, ...]
    clubhead_position_m: tuple[tuple[float, float, float], ...]
    clubhead_speed_mps: tuple[float, ...] | None = None
    kinematic_sequence: QuantityData | None = None
    engine_version: str | None = None
    model_file_hash: str | None = None
    run_class: str = "forward_dynamics"
    force_provenance: str | None = None

    def __post_init__(self) -> None:
        n = len(self.time_s)
        if n == 0:
            raise ValueError("time_s must be non-empty")
        if not all(math.isfinite(t) for t in self.time_s):
            raise ValueError("time_s must be finite")
        if any(b <= a for a, b in zip(self.time_s, self.time_s[1:], strict=False)):
            raise ValueError("time_s must be strictly increasing")
        if len(self.clubhead_position_m) != n:
            raise ValueError("clubhead_position_m must match time_s length")
        if any(len(p) != POSITION_DIM for p in self.clubhead_position_m):
            raise ValueError("clubhead_position_m rows must have 3 components")
        if self.clubhead_speed_mps is not None and len(self.clubhead_speed_mps) != n:
            raise ValueError("clubhead_speed_mps must match time_s length")


def derive_speed(run: NativeSwingRun) -> tuple[float, ...] | None:
    """Return the speed series: engine-supplied, else differenced positions.

    A sample whose position neighbourhood contains a non-finite entry comes
    back ``nan`` (the caller maps it to ``None``); fewer than two samples give
    ``None`` (no speed can be formed).
    """
    if run.clubhead_speed_mps is not None:
        return run.clubhead_speed_mps
    if len(run.time_s) < 2:
        return None
    pos = np.asarray(run.clubhead_position_m, dtype=float)
    pos[~np.isfinite(pos)] = np.nan
    vel = np.gradient(pos, np.asarray(run.time_s, dtype=float), axis=0)
    return tuple(float(s) for s in np.linalg.norm(vel, axis=1))
