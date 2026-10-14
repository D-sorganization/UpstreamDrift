"""Cross-engine bar-path parity of a lift same-input bundle (LIFT-5, #11745).

An engine replays the bundle open loop (the bundle's start state and efforts,
the shared ZOH-RK4 integrator of :mod:`~src.shared.python.motion_matching.\
same_input`) and places the bar landmark with its own forward kinematics
(:func:`~src.shared.python.lifting.bundles.bar_path`).  The deviation is
measured against the MuJoCo bar path recorded in the bundle provenance
(``window_bar_path_m``), so it combines dynamics and kinematics parity.

The committed full-window record (:data:`BAR_PATH_PARITY_RECORD`) is written
by ``scripts/measure_lift_bar_path_parity.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.lifting.bundles import bar_path
from src.shared.python.motion_matching.same_input import (
    ALL_ENGINES,
    PARITY_ENGINES,
    InputBundle,
    VectorPlant,
    open_loop,
)

Array = NDArray[np.float64]

_REPO = Path(__file__).resolve().parents[4]
BAR_PATH_PARITY_RECORD = (
    _REPO / "docs/development/lifting/evidence/back_squat/bar_path_parity.json"
)
# Acceptance of #11745: replay parity in at least these engines.
REQUIRED_ENGINES: tuple[str, ...] = PARITY_ENGINES


@dataclass(frozen=True)
class BarPathReplay:
    """One engine's open-loop replay of a bundle, seen through the bar path.

    ``bar_m`` (steps + 1, 3) is the replay engine's bar landmark;
    ``deviation_m`` its distance from the recorded MuJoCo bar path and
    ``coordinate_error_rad`` the largest coordinate error per state against
    the bundle reference and ``final_coordinate_error_rad`` (nv,) the
    per-coordinate error of the last reached state.  ``failure`` is the integrator's failure reason, if
    the replay stopped early.
    """

    engine: str
    dt_s: float
    bar_m: Array
    deviation_m: Array
    coordinate_error_rad: Array
    final_coordinate_error_rad: Array
    failure: str | None

    @property
    def steps(self) -> int:
        """Integration steps actually reached."""
        return int(self.bar_m.shape[0] - 1)

    @property
    def max_bar_deviation_m(self) -> float:
        """Largest bar-landmark distance from the recorded path, metres."""
        return float(self.deviation_m.max())

    @property
    def max_coordinate_error_rad(self) -> float:
        """Largest coordinate error against the bundle reference."""
        return float(self.coordinate_error_rad.max())

    def summary(self) -> dict[str, Any]:
        """Strict-JSON scalars for the parity record."""
        return {
            "engine": self.engine,
            "steps": self.steps,
            "duration_s": self.steps * self.dt_s,
            "failure": self.failure,
            "max_bar_deviation_m": self.max_bar_deviation_m,
            "final_bar_deviation_m": float(self.deviation_m[-1]),
            "max_coordinate_error_rad": self.max_coordinate_error_rad,
        }


def recorded_bar_path(bundle: InputBundle) -> Array:
    """The MuJoCo bar path stored with the bundle, (steps + 1, 3) metres.

    Raises:
        ValueError: When the bundle carries no matching ``window_bar_path_m``.
    """
    raw = bundle.provenance.get("window_bar_path_m")
    if raw is None:
        raise ValueError("bundle provenance has no window_bar_path_m")
    path = np.asarray(raw, dtype=float)
    if path.shape != (bundle.steps + 1, 3) or not np.isfinite(path).all():
        raise ValueError(
            f"window_bar_path_m must be {(bundle.steps + 1, 3)} finite values"
        )
    return path


def replay_bar_path(
    engine: str, bundle: InputBundle, *, steps: int | None = None
) -> BarPathReplay:
    """Replay ``steps`` bundle steps open loop in ``engine``; score the bar path.

    Args:
        engine: One of the same-input engines (``ALL_ENGINES``).
        bundle: A lift bundle carrying ``window_bar_path_m`` provenance.
        steps: Leading steps to replay (default: all).

    Postconditions: ``bar_m`` has one row per reached state; MuJoCo, the
    reference engine, reproduces the recorded path exactly.

    Raises:
        ValueError: For an unknown engine, a step count outside
            ``[1, bundle.steps]`` or a bundle without a recorded bar path.
    """
    if engine not in ALL_ENGINES:
        raise ValueError(f"engine must be one of {ALL_ENGINES}, got {engine!r}")
    count = bundle.steps if steps is None else steps
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("steps must be an integer")
    if not 1 <= count <= bundle.steps:
        raise ValueError(f"steps must lie in [1, {bundle.steps}], got {count}")
    recorded = recorded_bar_path(bundle)
    plant = VectorPlant(engine, bundle.spec_bytes)
    rollout = open_loop(
        plant,
        bundle.q0,
        bundle.v0,
        bundle.replay_efforts(count),
        dt_s=bundle.dt_s,
        policy=bundle.step_policy(),
    )
    reached = rollout.q.shape[0]
    frame = bundle.provenance.get("bar_frame", "Hub")
    bar = bar_path(plant, rollout.q, frame=frame)
    errors = np.abs(rollout.q - bundle.reference_q[:reached])
    return BarPathReplay(
        engine=engine,
        dt_s=float(bundle.dt_s),
        bar_m=bar,
        deviation_m=np.linalg.norm(bar - recorded[:reached], axis=1),
        coordinate_error_rad=errors.max(axis=1),
        final_coordinate_error_rad=errors[-1],
        failure=rollout.failure,
    )
