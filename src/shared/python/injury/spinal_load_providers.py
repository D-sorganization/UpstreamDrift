"""Spinal load providers: heuristic and model-based behind one interface (SPN-12).

``SpinalLoadAnalyzer`` historically computed lumbar loads from a closed-form
heuristic (body-weight share above the level, torque over a fixed 5 cm moment
arm, a centrifugal term). Those numbers are a screening estimate, not a joint
reaction. This module separates the two sources of per-level loads:

* :class:`HeuristicSpinalLoadProvider` wraps the unchanged heuristic and labels
  every output ``heuristic`` (provenance ``solver``, ``distribution_source``
  and ``provider_label``), so a heuristic series cannot be mistaken for a model
  load;
* :class:`ModelSpinalLoadProvider` wraps a precomputed
  :class:`~src.shared.python.spine.loads.SpinalLoadSeries` (for example the
  OpenSim pipeline output) and passes it through with provenance intact.

Both satisfy :class:`SpinalLoadProvider`. Sign convention is that of
``spine.loads`` (ADR-0055 item 5): force on the superior vertebra by the
inferior one, inferior ISB frame, compression ``+F_y``, AP shear ``+F_x``,
ML shear ``+F_z``. The heuristic risk thresholds apply to heuristic output
only; the analyzer does not apply them to model output.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.shared.python.spine.levels import SpinalLevel, parse_level
from src.shared.python.spine.loads import (
    REQUIRED_PROVENANCE_KEYS,
    LoadObservable,
    SpinalLoadSeries,
)

__all__ = [
    "HEURISTIC_PROVIDER_ID",
    "HeuristicSpinalLoadProvider",
    "ModelSpinalLoadProvider",
    "SpinalLoadProvider",
]

#: Identifier of the heuristic provider; also surfaced by the analyzer result.
HEURISTIC_PROVIDER_ID = "heuristic_v1"

_TIME_MATCH_TOL_S = 1e-9
_ArrayDict = Mapping[str, np.ndarray]


@runtime_checkable
class SpinalLoadProvider(Protocol):
    """Source of per-level spinal loads.

    Attributes:
        provider_id: Stable identifier surfaced on analysis results.
        is_model_based: ``True`` only for loads from a musculoskeletal model.
    """

    provider_id: str
    is_model_based: bool

    def loads(
        self,
        *,
        time_s: np.ndarray,
        joint_angles: _ArrayDict | None = None,
        joint_velocities: _ArrayDict | None = None,
        joint_torques: _ArrayDict | None = None,
        levels: tuple[SpinalLevel, ...] | None = None,
    ) -> SpinalLoadSeries:
        """Return the per-level load series for the given time base."""
        ...


def _heuristic_provenance() -> dict[str, str]:
    provenance = dict.fromkeys(REQUIRED_PROVENANCE_KEYS, "n/a (heuristic)")
    provenance["solver"] = HEURISTIC_PROVIDER_ID
    provenance["cost_exponent"] = "n/a"
    provenance["distribution_source"] = (
        "heuristic closed-form estimate, not a distributed model load"
    )
    provenance["external_loads"] = "not modelled (heuristic)"
    provenance["provider_label"] = "heuristic"
    return provenance


class HeuristicSpinalLoadProvider:
    """The legacy analyzer heuristic exposed as a labelled load series.

    ``analyzer`` supplies ``lumbar_segments`` (level names) and
    ``compute_segment_loads``; the numbers are exactly the analyzer's. Torsion
    is reported as the axial moment ``M_y``.

    Raises (from :meth:`loads`): ``ValueError`` if an analyzed segment name is
    not a known :class:`SpinalLevel` or ``time_s`` is not a valid time base.
    """

    provider_id = HEURISTIC_PROVIDER_ID
    is_model_based = False

    def __init__(self, analyzer: Any) -> None:
        for attr in ("lumbar_segments", "compute_segment_loads"):
            if not hasattr(analyzer, attr):
                raise TypeError(f"analyzer must provide {attr!r}")
        self._analyzer = analyzer

    def loads(
        self,
        *,
        time_s: np.ndarray,
        joint_angles: _ArrayDict | None = None,
        joint_velocities: _ArrayDict | None = None,
        joint_torques: _ArrayDict | None = None,
        levels: tuple[SpinalLevel, ...] | None = None,
    ) -> SpinalLoadSeries:
        """Return the heuristic loads as a series labelled ``heuristic``."""
        time = np.asarray(time_s, dtype=float)
        angles = {} if joint_angles is None else dict(joint_angles)
        velocities = {} if joint_velocities is None else dict(joint_velocities)
        torques = {} if joint_torques is None else dict(joint_torques)
        names = list(self._analyzer.lumbar_segments)
        parsed = tuple(parse_level(name) for name in names)
        keep = [i for i, lv in enumerate(parsed) if levels is None or lv in levels]
        if not keep:
            raise ValueError("no analyzed segment matches the requested levels")
        force = np.zeros((time.size, len(keep), 3))
        moment = np.zeros_like(force)
        for out, i in enumerate(keep):
            seg = self._analyzer.compute_segment_loads(
                names[i], angles, velocities, torques, time
            )
            force[:, out, 0] = seg.ap_shear
            force[:, out, 1] = seg.compression
            force[:, out, 2] = seg.lateral_shear
            moment[:, out, 1] = seg.torsion
        return SpinalLoadSeries(
            time_s=time,
            levels=tuple(parsed[i] for i in keep),
            observable=LoadObservable.GENERALIZED_EFFORT,
            force_n=force,
            moment_nm=moment,
            provenance=_heuristic_provenance(),
        )


class ModelSpinalLoadProvider:
    """Pass-through of a precomputed model :class:`SpinalLoadSeries`.

    Preconditions: ``series`` is a ``SpinalLoadSeries``; ``provider_id`` is a
    non-empty string different from :data:`HEURISTIC_PROVIDER_ID`.

    Postcondition of :meth:`loads`: the returned series carries the original
    arrays and provenance unchanged (the same object when no level subset is
    requested). A requested time base must equal the series' time base and
    every requested level must be present, otherwise ``ValueError``.
    """

    is_model_based = True

    def __init__(
        self, series: SpinalLoadSeries, provider_id: str = "model_precomputed"
    ) -> None:
        if not isinstance(series, SpinalLoadSeries):
            raise TypeError("series must be a SpinalLoadSeries")
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("provider_id must be a non-empty string")
        if provider_id == HEURISTIC_PROVIDER_ID:
            raise ValueError("provider_id must not reuse the heuristic provider id")
        self._series = series
        self.provider_id = provider_id

    def loads(
        self,
        *,
        time_s: np.ndarray,
        joint_angles: _ArrayDict | None = None,
        joint_velocities: _ArrayDict | None = None,
        joint_torques: _ArrayDict | None = None,
        levels: tuple[SpinalLevel, ...] | None = None,
    ) -> SpinalLoadSeries:
        """Return the stored series (optionally a level subset)."""
        time = np.asarray(time_s, dtype=float)
        stored = self._series.time_s
        if time.shape != stored.shape or not np.allclose(
            time, stored, rtol=0, atol=_TIME_MATCH_TOL_S
        ):
            raise ValueError(
                f"requested time base {time.shape} does not match the model "
                f"series time base {stored.shape}"
            )
        if levels is None:
            return self._series
        missing = [lv.name for lv in levels if lv not in self._series.levels]
        if missing:
            raise ValueError(f"levels not in the model series: {missing}")
        idx = [self._series.levels.index(lv) for lv in levels]
        return SpinalLoadSeries(
            time_s=stored,
            levels=tuple(levels),
            observable=self._series.observable,
            force_n=self._series.force_n[:, idx, :],
            moment_nm=self._series.moment_nm[:, idx, :],
            provenance=dict(self._series.provenance),
            force_unit=self._series.force_unit,
        )
